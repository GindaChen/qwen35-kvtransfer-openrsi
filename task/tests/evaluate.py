"""Harbor Judge adapter. Historical six-hour run did not use this adapter."""

import json
import math
import os
import secrets
import statistics
import subprocess
import sys
from pathlib import Path

LOG = Path("/logs/verifier")


def load(path):
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    keyed = {(r["p"], r["d"], r["bs"], r["ctx"], r["layout"]): r for r in rows}
    expected = {
        (p, d, bs, ctx, l)
        for p, d in [(1, 1), (1, 2), (2, 1), (2, 2)]
        for bs in [1, 4]
        for ctx in [4096, 8192, 32768, 65536, 4097, 10034, 32749, 65521]
        for l in ["contiguous", "fragmented", "fragmented_runs"]
    }
    if (
        len(rows) != 192
        or set(keyed) != expected
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


def main():
    LOG.mkdir(parents=True, exist_ok=True)
    try:
        env = dict(os.environ, PYTHONPATH="/opt/kv-transfer")
        map_seed = secrets.randbits(
            63
        )  # Identical map sequence for the paired implementations.
        for label, candidate in [
            ("reference", None),
            ("candidate", "/workspace/candidate.py"),
        ]:
            cmd = [
                sys.executable,
                "/opt/kv-transfer/benchmark.py",
                "--output",
                str(LOG / (label + ".jsonl")),
                "--warmup",
                "10",
                "--samples",
                "50",
                "--seed",
                str(map_seed),
            ]
            if candidate:
                cmd += ["--candidate", candidate]
            with (LOG / (label + ".log")).open("w") as stream:
                subprocess.run(
                    cmd,
                    env=env,
                    cwd="/workspace",
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1700,
                )
        b, c = load(LOG / "reference.jsonl"), load(LOG / "candidate.jsonl")
        ratios = {k: b[k]["median_us"] / c[k]["median_us"] for k in b}
        reward = math.exp(statistics.mean(math.log(v) for v in ratios.values()))
        (LOG / "summary.json").write_text(
            json.dumps(
                {
                    "cases": 192,
                    "geometric_mean_speedup": reward,
                    "reference": "Torch baseline",
                    "scope": "transfer only",
                },
                indent=2,
            )
        )
        (LOG / "reward.json").write_text(json.dumps({"reward": reward}))
    except Exception as exc:
        (LOG / "failure.json").write_text(
            json.dumps({"status": "unscored", "error": str(exc)})
        )
        raise


if __name__ == "__main__":
    main()
