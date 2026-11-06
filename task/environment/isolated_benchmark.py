"""Protocol 0.3: a trusted controller measures and checks an isolated candidate.

Authority split (see docs/protocol03/README.md):

* Controller (this process, trusted, never imports candidate code): owns map
  seeds, payload generation, private source snapshots, the clock, output
  capture, every correctness check and the result file. It makes itself
  non-dumpable, so no same-uid process can read or write its memory through
  /proc or ptrace.
* Worker (untrusted): `isolated_worker.py` runs inside the Harness
  `rsi-isolate` (via gpu_isolate.py) with a seccomp guard installed before
  torch or the candidate is imported. It can read only the task environment,
  the candidate directory and the Python/CUDA runtime, write only its private
  HOME/TMPDIR, has no network, and is killed with everything it started.
* Channel: one AF_UNIX SOCK_SEQPACKET pair. Controller -> worker: bounded JSON
  plus dedicated CUDA VMM allocation FDs. Worker -> controller: bounded JSON
  only; the only accepted message is {'kind':'done','nonce':<controller nonce>}.
  No worker clock, score, file or Python object is ever consumed.

Per case, outside timing: dedicated VMM buffers (never torch allocator slabs)
are allocated and published once, as in protocol 0.2's per-case allocation.
Per invocation, outside timing: fresh maps, destination poison, a
controller-private staging copy of fresh payload for every source byte (also
the expected snapshot) and the unselected state slots. Timed: (1) publish --
the controller writes the fresh KV pages and selected state slots into the
exported source buffers and synchronises the source devices, so no transfer
work can precede the clock; (2) the request with the maps; (3) the candidate
transfer, the worker's device synchronisation and its acknowledgement; (4) a
controller-private copy (capture) of the destination KV buffers and selected
destination state slots and the controller's device synchronisation. A worker that
acknowledges early either has its late work inside the capture (timed) or is
caught by the check. After timing: the independent bit-exact checker runs on
the capture (selected state and untouched holes) and source preservation is
checked on the live buffers (late writes there can only fail the check).
"""
import argparse
import json
import math
import os
import secrets
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import benchmark as b
from isolation.guard import protect_parent
from isolation.vmm import allocate_plan, describe_allocations, encode_maps
from isolation.wire import receive, send

PROTOCOL_VERSION = '0.3.0'
ROOT = Path(__file__).resolve().parent
SCRATCH_LIMIT = 2 * 1024 ** 3
TRANSFER_TIMEOUT = 600  # seconds per invocation; a timeout is an unscored failure
PROFILE = os.environ.get('KV_WORKER_PROFILE') == '1'


class Worker:
    """One isolated worker process serving every case of one suite."""

    def __init__(self, candidate, log_path, isolator=None, timeout=1800):
        protect_parent()
        self.socket, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.socket.settimeout(timeout)
        self.process = None
        self.log = open(log_path, 'xb')
        candidate = Path(candidate).resolve(strict=True)
        command = [sys.executable, '-I', str(ROOT / 'gpu_isolate.py'), '--controller-pid', str(os.getpid())]
        if isolator is not None:
            # Trusted development option for hosts without the Harness mount;
            # the official adapter supplies none and uses /opt/rsi-isolate.
            command += ['--library', str(Path(isolator).resolve(strict=True))]
        command += ['run', '--no-network', '--timeout', '7000',
                    '--read', str(ROOT), '--read', str(candidate.parent),
                    '--read', str(Path(sys.prefix).resolve()),
                    '--pass-fd', str(child.fileno())]
        variables = {'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3'),
                     'MAX_JOBS': '16', 'CMAKE_BUILD_PARALLEL_LEVEL': '16',
                     'NVCC_THREADS': '1', 'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8',
                     'TORCH_CUDA_ARCH_LIST': os.environ.get('TORCH_CUDA_ARCH_LIST', '10.0'),
                     'PYTHONNOUSERSITE': '1',
                     'KV_WORKER_PROFILE': os.environ.get('KV_WORKER_PROFILE', '0')}
        # Development hosts may keep site-packages outside sys.prefix; expose
        # exactly the import path this trusted interpreter already uses.
        extra = [p for p in sys.path if p and os.path.isdir(p)
                 and not Path(p).resolve().is_relative_to(Path(sys.prefix).resolve())
                 and Path(p).resolve() != ROOT]
        for path in extra:
            command += ['--read', str(Path(path).resolve())]
        variables['PYTHONPATH'] = ':'.join([str(ROOT)] + extra)
        cuda_home = os.environ.get('CUDA_HOME')
        if cuda_home and Path(cuda_home).is_dir():
            variables['CUDA_HOME'] = str(Path(cuda_home).resolve())
            command += ['--read', variables['CUDA_HOME']]
        variables['PATH'] = ':'.join(filter(None, [
            str(Path(sys.executable).parent),
            str(Path(cuda_home) / 'bin') if cuda_home else None,
            '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin']))
        for key, value in variables.items():
            command += ['--env', f'{key}={value}']
        command += ['--', sys.executable, str(ROOT / 'isolated_worker.py'),
                    '--fd', str(child.fileno()), '--candidate', str(candidate)]
        env = dict(os.environ, RSI_JUDGE_ISOLATION_FALLBACK='none')
        env.pop('RSI_JUDGE_ALLOW_MISSING_LANDLOCK_SIGNAL', None)
        env.pop('KV_MAP_SEED', None)
        try:
            self.process = subprocess.Popen(command, env=env, pass_fds=(child.fileno(),),
                                            stdin=subprocess.DEVNULL,
                                            stdout=self.log, stderr=subprocess.STDOUT)
            child.close()
            message, _ = receive(self.socket)
            if message != {'kind': 'ready', 'protocol': PROTOCOL_VERSION}:
                raise RuntimeError('Invalid worker readiness message')
        except BaseException:
            child.close()
            self.close()
            raise

    def _expect(self, kind, nonce, spin=None):
        message, _ = receive(self.socket, spin=spin)
        if message != {'kind': kind, 'nonce': nonce}:
            raise RuntimeError('Invalid worker message; expected ' + kind)

    def prepare(self, plan, allocations):
        """Untimed: publish this case's dedicated buffers (no maps, no fresh payload)."""
        nonce = secrets.token_hex(16)
        fds = []
        try:
            for allocation in allocations:
                fds.append(allocation.export())
            send(self.socket, describe_allocations(plan, allocations, nonce), fds)
        finally:
            for fd in fds:
                os.close(fd)
        self._expect('prepared', nonce)

    def transfer(self, maps):
        nonce = secrets.token_hex(16)
        send(self.socket, {'kind': 'transfer', 'nonce': nonce, 'maps': maps})
        self._expect('done', nonce, spin=TRANSFER_TIMEOUT)

    def release(self):
        nonce = secrets.token_hex(16)
        send(self.socket, {'kind': 'release', 'nonce': nonce})
        self._expect('released', nonce)

    def pid(self):
        return None if self.process is None else self.process.pid

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    send(self.socket, {'kind': 'stop'})
                    self.process.wait(timeout=30)
                except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
                    self.process.kill()  # rsi-isolate kills all descendants on exit
                    self.process.wait()
            self.process = None
        self.socket.close()
        self.log.close()


SLOT_STATE = ('conv', 'recurrent')


def stage(plan, generation):
    """Untimed preparation of one invocation.

    Creates a controller-private fresh payload for every source byte (never
    exported; after the timed publish it is also the expected source state),
    pre-writes the parts of it that are not selected for transfer (unselected
    state slots), and writes fresh destination poison. Selected slots keep
    stale bytes until the timed publish; KV pages are published in full."""
    staged, index = [], []
    for rank, source in enumerate(plan['src']):
        torch.cuda.set_device(rank)
        dev = torch.device('cuda', rank)
        rows = {name: torch.empty_like(t) for name, t in source.items()}
        for t in rows.values():
            b.random_finite_payload(t)
        slots = plan['src_slots'].to(dev)
        keep = torch.ones(source['conv'].shape[1], dtype=torch.bool, device=dev)
        keep[slots] = False
        unselected = keep.nonzero().flatten()
        for name in SLOT_STATE:
            source[name].index_copy_(1, unselected, rows[name].index_select(1, unselected))
        staged.append(rows)
        index.append({'src_slots': slots,
                      'selected': {name: rows[name].index_select(1, slots) for name in SLOT_STATE}})
    poison = -777.0 if generation % 2 == 0 else 333.0
    for target in plan['dst']:
        for t in target.values():
            t.fill_(poison)
        dev = target['kv'].device
        slots = plan['dst_slots'].to(dev)
        out = {'kv': torch.empty_like(target['kv'])}
        for name in SLOT_STATE:
            shape = list(target[name].shape)
            shape[1] = slots.numel()
            out[name] = torch.empty(shape, dtype=target[name].dtype, device=dev)
        index.append({'dst_slots': slots, 'out': out})
    return staged, index, poison


def publish(plan, staged, index):
    """Timed: write the fresh payload into the exported source buffers (all KV
    pages, the selected state slots) and synchronise the source devices.

    Before this point the selected bytes are stale, so no transfer work can be
    done ahead of the clock."""
    for rank, source in enumerate(plan['src']):
        source['kv'].copy_(staged[rank]['kv'])
        slots = index[rank]['src_slots']
        for name in SLOT_STATE:
            source[name].index_copy_(1, slots, index[rank]['selected'][name])
    for rank in range(plan['p']):
        torch.cuda.synchronize(rank)


def capture(plan, index):
    """Timed: controller-private copies of every destination KV buffer and of
    every selected destination state slot, then device synchronisation.

    Never exported. Late worker writes are either inside this timed capture or
    absent from the checked state."""
    out = []
    for rank, target in enumerate(plan['dst']):
        slots = index[plan['p'] + rank]['dst_slots']
        row = index[plan['p'] + rank]['out']  # preallocated outside timing
        row['kv'].copy_(target['kv'])
        for name in SLOT_STATE:
            torch.index_select(target[name], 1, slots, out=row[name])
        out.append(row)
    b.sync()
    return out


def check_captured(plan, captured, before, poison):
    """The independent checker on the capture: selected KV pages, KV holes and
    selected state slots. Unselected state-slot holes are checked live."""
    view = {k: v for k, v in plan.items() if k != 'dst'}
    view['dst'] = captured
    view['dst_slots'] = torch.arange(plan['dst_slots'].numel())
    b.check(view, before, poison)
    for target in plan['dst']:
        dev = target['kv'].device
        keep = torch.ones(target['conv'].shape[1], dtype=torch.bool)
        keep[plan['dst_slots']] = False
        holes = keep.nonzero().flatten().to(dev)
        for name in SLOT_STATE:
            if holes.numel() and not bool((target[name].index_select(1, holes) == poison).all()):
                raise AssertionError('overwritten hole ' + name)


def run_case(worker, p, d, bs, ctx, layout, warmup, samples, seed):
    plan, allocations = allocate_plan(p, d, bs, ctx, layout, b.derive_seed(seed, 'allocation'))
    times = []
    try:
        worker.prepare(plan, allocations)
        for generation in range(2 + warmup + samples):
            # Untimed: fresh maps, full-buffer refresh (unselected source bytes and
            # destination poison), private fresh payload and expected snapshots.
            plan.update(b.maps(bs, ctx, layout, b.derive_seed(seed, generation)))
            staged, index, poison = stage(plan, generation)
            request = encode_maps(plan)
            b.sync()
            start = time.perf_counter_ns()
            publish(plan, staged, index)
            published = time.perf_counter_ns()
            worker.transfer(request)
            acked = time.perf_counter_ns()
            captured = capture(plan, index)
            end = time.perf_counter_ns()
            elapsed = (end - start) / 1000
            if PROFILE:
                print('CONTROLLER_PROFILE publish_us=%.1f request_to_ack_us=%.1f capture_us=%.1f' % (
                    (published - start) / 1e3, (acked - published) / 1e3, (end - acked) / 1e3), flush=True)
            check_captured(plan, captured, staged, poison)
            b.check_source(plan, staged)
            if generation >= 2 + warmup:
                times.append(elapsed)
            del staged, captured
    finally:
        try:
            if worker.process is not None and worker.process.poll() is None:
                worker.release()
        finally:
            del plan
            for allocation in reversed(allocations):
                allocation.close()
    return dict(p=p, d=d, bs=bs, ctx=ctx, layout=layout, correct=True,
                median_us=statistics.median(times), samples_us=times,
                useful_bytes=bs * (ctx * 20480 + 64471040),
                page_payload_bytes=bs * (math.ceil(ctx / b.BLOCK) * b.BLOCK * 20480 + 64471040))


def cases(smoke):
    for p, d in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        for ctx in [4097] if smoke else b.LENGTHS:
            for bs in [1] if smoke else [1, 4]:
                for layout in b.LAYOUTS:
                    yield p, d, bs, ctx, layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default=str(ROOT / 'reference/candidate.py'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int,
                        help='Development only. The Judge passes KV_MAP_SEED in the environment: '
                             'command lines are world-readable through /proc, environments are not')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--isolator', help='Trusted development-only rsi-isolate library directory')
    parser.add_argument('--allow-hardware', action='store_true',
                        help='Development only: permit non-GB200 GPUs; the result is marked non-scoring')
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        raise ValueError('Invalid sampling counts')
    if torch.cuda.device_count() != 4:
        raise RuntimeError('Exactly four delegated GPUs are required')
    names = [torch.cuda.get_device_name(i) for i in range(4)]
    if not args.allow_hardware and not all('GB200' in n for n in names):
        raise RuntimeError('Require NVIDIA GB200; do not mix hardware results')
    for i in range(4):
        for j in range(4):
            if i != j and not torch.cuda.can_device_access_peer(i, j):
                raise RuntimeError('Peer access required')
    seed_provided = bool(os.environ.get('KV_MAP_SEED')) or args.seed is not None
    protect_parent()  # before the seed is read: no same-uid /proc or ptrace access
    if os.environ.get('KV_MAP_SEED'):
        seed = int(os.environ.pop('KV_MAP_SEED'))
    elif args.seed is not None:
        seed = args.seed
    else:
        seed = secrets.randbits(63)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Preserve earlier results; choose a new output path')
    metadata = dict(protocol_version=PROTOCOL_VERSION, metric='verified_transfer_latency_us',
                    timing='fresh selected-source publish + request + transfer + ack + private selected-destination capture',
                    hardware=names, scoring_hardware=all('GB200' in n for n in names),
                    torch=torch.__version__, cuda=torch.version.cuda,
                    mode='torch-reference' if Path(args.candidate).resolve() == (ROOT / 'reference/candidate.py').resolve() else 'candidate',
                    seed_mode='provided' if seed_provided else 'os-random',
                    started_at=time.time())
    output.with_suffix('.environment.json').write_text(json.dumps(metadata, indent=2))
    worker = Worker(args.candidate, output.with_suffix('.worker.log'), args.isolator)
    try:
        with output.open('x') as stream:
            for p, d, bs, ctx, layout in cases(args.smoke):
                row = run_case(worker, p, d, bs, ctx, layout, args.warmup, args.samples,
                               b.derive_seed(seed, f'{p}/{d}/{bs}/{ctx}/{layout}'))
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                print(json.dumps({k: v for k, v in row.items() if k != 'samples_us'}), flush=True)
                torch.cuda.empty_cache()
    finally:
        worker.close()
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
