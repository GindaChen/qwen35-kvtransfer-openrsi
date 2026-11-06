"""Harbor Judge adapter for protocol 0.3 (isolated candidate, trusted controller).

The Judge process and the two controller processes it starts are trusted and
never import candidate code. Each controller (/opt/kv-transfer/isolated_benchmark.py)
runs the implementation's transfer() only inside an isolated worker started
through the Harness `rsi-isolate` (/opt/rsi-isolate), which cannot read or
write /logs/verifier, the Judge's or controller's memory, or reach any socket
endpoint. Result rows are written only by the controllers. If the isolation
facility is missing or reports an unconfined mode, the run is unscored.

Development hosts without a Harness mount may set KV_ISOLATOR (a trusted
rsi-isolate library directory), KV_TASK_ROOT, KV_LOG_DIR and KV_CANDIDATE.
"""

import json
import math
import os
import secrets
import statistics
import subprocess
import sys
from pathlib import Path

LOG = Path(os.environ.get('KV_LOG_DIR', '/logs/verifier'))
TASK = Path(os.environ.get('KV_TASK_ROOT', '/opt/kv-transfer'))
CANDIDATE = Path(os.environ.get('KV_CANDIDATE', '/workspace/candidate.py'))
ISOLATOR = os.environ.get('KV_ISOLATOR')  # None: the Harness mount /opt/rsi-isolate
PROTOCOL = '0.3.0'
CASES = {
    (p, d, bs, ctx, l)
    for p, d in [(1, 1), (1, 2), (2, 1), (2, 2)]
    for bs in [1, 4]
    for ctx in [4096, 8192, 32768, 65536, 4097, 10034, 32749, 65521]
    for l in ['contiguous', 'fragmented', 'fragmented_runs']
}


def load(path):
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    keyed = {(r["p"], r["d"], r["bs"], r["ctx"], r["layout"]): r for r in rows}
    if (
        len(rows) != 192
        or set(keyed) != CASES
        or not all(
            r.get("correct") is True
            and type(r["median_us"]) in (int, float)
            and math.isfinite(r["median_us"])
            and r["median_us"] > 0
            and len(r["samples_us"]) == 50
            and all(type(t) in (int, float) and math.isfinite(t) and t > 0
                    for t in r["samples_us"])
            and r["median_us"] == statistics.median(r["samples_us"])
            for r in rows
        )
    ):
        raise RuntimeError("Invalid/incomplete suite")
    return keyed


def require_isolation():
    """Refuse to score unless the Harness isolator reports a confined mode."""
    library = Path(ISOLATOR) if ISOLATOR else Path('/opt/rsi-isolate')
    script = library / 'rsi_isolate.py'
    if not script.is_file():
        raise RuntimeError('Harness rsi-isolate is unavailable; refusing to score')
    env = dict(os.environ, RSI_JUDGE_ISOLATION_FALLBACK='none')
    out = subprocess.run([sys.executable, '-I', str(script), 'isolation'], env=env,
                         capture_output=True, text=True, timeout=60)
    mode = json.loads(out.stdout or '{}') if out.returncode == 0 else {}
    if mode.get('mode') not in ('landlock', 'uid', 'userns') or mode.get('fallback') not in (None, 'none'):
        raise RuntimeError('Unconfined isolation mode; refusing to score: ' + (out.stdout + out.stderr)[-300:])
    return mode


def run_suite(label, candidate, seed):
    output = LOG / (label + '.jsonl')
    cmd = [sys.executable, str(TASK / 'isolated_benchmark.py'), '--output', str(output),
           '--warmup', '10', '--samples', '50']
    if candidate:
        cmd += ['--candidate', str(candidate)]
    if ISOLATOR:
        cmd += ['--isolator', ISOLATOR]
    # The seed travels in the controller's environment, never on a command line
    # (/proc/<pid>/cmdline is world-readable; environ is not).
    env = dict(os.environ, PYTHONPATH=str(TASK), KV_MAP_SEED=str(seed))
    with (LOG / (label + '.log')).open('w') as stream:
        subprocess.run(cmd, env=env, cwd=str(TASK), stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=5400)
    meta = json.loads(output.with_suffix('.environment.json').read_text())
    if meta.get('protocol_version') != PROTOCOL or meta.get('scoring_hardware') is not True:
        raise RuntimeError('Result was not produced by the protocol-0.3 controller on GB200')
    return load(output)


def protect_self():
    if not sys.platform.startswith('linux'):
        return  # unit tests on development machines; the Judge is Linux
    import ctypes
    if ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0):  # PR_SET_DUMPABLE = 0
        raise OSError(ctypes.get_errno(), 'Cannot protect Judge process memory')


def main():
    protect_self()
    LOG.mkdir(parents=True, exist_ok=True)
    try:
        mode = require_isolation()
        if not CANDIDATE.is_file():
            raise RuntimeError('Missing candidate')
        map_seed = secrets.randbits(63)  # identical map sequence for both implementations
        b = run_suite('reference', None, map_seed)
        c = run_suite('candidate', CANDIDATE, map_seed)
        ratios = {k: b[k]["median_us"] / c[k]["median_us"] for k in b}
        reward = math.exp(statistics.mean(math.log(v) for v in ratios.values()))
        (LOG / "summary.json").write_text(json.dumps({
            "protocol": PROTOCOL,
            "cases": 192,
            "geometric_mean_speedup": reward,
            "reference": "Torch baseline in the same isolated worker path",
            "isolation": mode,
            "scope": "verified isolated transfer latency (publish + request + transfer + ack + capture)",
        }, indent=2))
        (LOG / "reward.json").write_text(json.dumps({"reward": reward}))
    except Exception as exc:
        (LOG / "failure.json").write_text(json.dumps({"status": "unscored", "error": str(exc)}))
        raise


if __name__ == "__main__":
    main()
