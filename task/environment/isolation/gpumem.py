"""Trusted enforcement of the 2 GiB/GPU candidate scratch cap.

The candidate may run arbitrary native code (cudaMalloc, cuMemCreate, child
processes), so the cap cannot be enforced inside the worker. Instead a trusted
monitor process -- started by the controller before the worker, never sharing
memory with it, non-dumpable -- reads the driver's per-process accounting
through NVML (`nvmlDeviceGetComputeRunningProcesses`) for each of the four
delegated GPUs:

* PIDs present before the worker starts (the controller and anything else
  already on the GPUs) are excluded. Every other PID on those GPUs belongs to
  the worker's process tree: rsi-isolate gives the Judge exclusive GPUs and
  ends all descendants with the worker. PIDs are compared in NVML's own
  namespace, so a container without host PID visibility works too.
* The worker initialises a CUDA context on every device *before* the candidate
  is imported and reports nothing; the controller asks the monitor for the
  worker tree's usage at that point (the context baseline, trusted code only).
* Scratch on a device = worker-tree usage - context baseline. Dedicated
  source/destination pools are created by the controller and only imported by
  the worker, so NVML charges them to the controller, not the worker.
* The monitor polls continuously (default every 5 ms) and keeps the peak; the
  controller also takes a synchronous sample at the end of every case. A peak
  above the cap makes the case -- and so the suite -- unscored.

Limits (documented in docs/protocol03/README.md): a transient allocation that
is both made and freed between two polls (< ~5 ms) is not observed; lazily
loaded kernels, library workspaces and the caching allocator's reserve after
the baseline count as scratch; if NVML cannot report per-process usage the
monitor fails closed and the run is unscored.
"""
import ctypes
import json
import os
import sys
import threading
import time

SCRATCH_LIMIT = 2 * 1024 ** 3
NOT_AVAILABLE = (1 << 64) - 1  # NVML_VALUE_NOT_AVAILABLE for unsigned long long


class ScratchError(RuntimeError):
    """The candidate exceeded the scratch cap, or usage could not be measured."""


def tree_usage(processes, excluded):
    """Bytes used by every PID not in ``excluded``.

    ``processes`` is a list of (pid, used_bytes) for one device. A PID may be
    listed more than once (MIG/compute instances); entries are summed. An
    unavailable value is never treated as zero."""
    total = 0
    for pid, used in processes:
        if pid in excluded:
            continue
        if used is None or used == NOT_AVAILABLE or used < 0:
            raise ScratchError('GPU memory of process %d is not reportable; cannot enforce scratch cap' % pid)
        total += used
    return total


class Ledger:
    """Pure accounting: baseline, per-device peaks, and the cap decision."""

    def __init__(self, devices, limit=SCRATCH_LIMIT):
        self.devices = list(devices)
        self.limit = limit
        self.excluded = set()
        self.context = None
        self.peak = [0] * len(self.devices)
        self.last = [0] * len(self.devices)

    def exclude(self, snapshot):
        """Record every PID currently on the delegated GPUs as not the worker's."""
        for processes in snapshot:
            for pid, _ in processes:
                self.excluded.add(pid)

    def set_context(self, snapshot):
        self.context = [tree_usage(p, self.excluded) for p in snapshot]

    def observe(self, snapshot):
        if self.context is None:
            raise ScratchError('Scratch observed before the worker context baseline')
        for i, processes in enumerate(snapshot):
            scratch = tree_usage(processes, self.excluded) - self.context[i]
            self.last[i] = scratch
            if scratch > self.peak[i]:
                self.peak[i] = scratch

    def take(self):
        """Return and reset the peaks since the last take."""
        peak, self.peak = self.peak, [0] * len(self.devices)
        return peak

    def violations(self, peak):
        return [(self.devices[i], value) for i, value in enumerate(peak) if value > self.limit]


# -- NVML (ctypes; the driver ships libnvidia-ml.so.1) ---------------------------

class _Process(ctypes.Structure):  # nvmlProcessInfo_v2_t (== v3 layout)
    _fields_ = [('pid', ctypes.c_uint), ('usedGpuMemory', ctypes.c_ulonglong),
                ('gpuInstanceId', ctypes.c_uint), ('computeInstanceId', ctypes.c_uint)]


class NVML:
    INSUFFICIENT_SIZE = 7

    def __init__(self, uuids):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        self._check(self.lib.nvmlInit_v2(), 'nvmlInit')
        self.handles = []
        for uuid in uuids:
            handle = ctypes.c_void_p()
            self._check(self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(handle)),
                        'nvmlDeviceGetHandleByUUID ' + uuid)
            self.handles.append(handle)
        name = 'nvmlDeviceGetComputeRunningProcesses_v3'
        if not hasattr(self.lib, name):
            name = 'nvmlDeviceGetComputeRunningProcesses_v2'
        self.query = getattr(self.lib, name)

    def _check(self, code, what):
        if code:
            self.lib.nvmlErrorString.restype = ctypes.c_char_p
            raise ScratchError('%s failed: %s' % (what, self.lib.nvmlErrorString(code).decode()))

    def processes(self, handle):
        size = 64
        while True:
            count = ctypes.c_uint(size)
            buffer = (_Process * size)()
            code = self.query(handle, ctypes.byref(count), buffer)
            if code == self.INSUFFICIENT_SIZE:
                size = max(size * 2, count.value + 8)
                continue
            self._check(code, 'compute process query')
            return [(buffer[i].pid, buffer[i].usedGpuMemory) for i in range(count.value)]

    def snapshot(self):
        return [self.processes(h) for h in self.handles]


# -- monitor process -----------------------------------------------------------

def serve(uuids, interval):
    """Line protocol on stdin/stdout. Requests: exclude, context, take, stop.
    Replies: one JSON object per request."""
    import ctypes as _c
    _c.CDLL(None).prctl(4, 0, 0, 0, 0)  # non-dumpable
    nvml = NVML(uuids)
    ledger = Ledger(range(len(uuids)))
    lock = threading.Lock()
    failure = []
    polling = threading.Event()

    def poll():
        while True:
            time.sleep(interval)
            if not polling.is_set():
                continue
            try:
                snap = nvml.snapshot()
                with lock:
                    ledger.observe(snap)
            except Exception as exc:  # fail closed: reported on the next take
                failure.append(str(exc))
                polling.clear()

    threading.Thread(target=poll, daemon=True).start()
    for line in sys.stdin:
        request = line.strip()
        try:
            with lock:
                if request == 'exclude':
                    ledger.exclude(nvml.snapshot())
                    reply = {'ok': True, 'excluded': sorted(ledger.excluded)}
                elif request == 'context':
                    ledger.set_context(nvml.snapshot())
                    polling.set()
                    reply = {'ok': True, 'context': ledger.context}
                elif request == 'take':
                    if failure:
                        raise ScratchError(failure[0])
                    ledger.observe(nvml.snapshot())  # synchronous end-of-case sample
                    peak = ledger.take()
                    reply = {'ok': True, 'peak': peak, 'last': list(ledger.last)}
                elif request == 'stop':
                    print(json.dumps({'ok': True}), flush=True)
                    return
                else:
                    raise ValueError('unknown request')
        except Exception as exc:
            reply = {'ok': False, 'error': '%s: %s' % (type(exc).__name__, exc)}
        print(json.dumps(reply), flush=True)


class Monitor:
    """Controller-side handle on the trusted monitor process."""

    def __init__(self, uuids, interval=None, limit=SCRATCH_LIMIT):
        import subprocess
        interval = interval or float(os.environ.get('KV_SCRATCH_POLL_MS', '5')) / 1000
        self.limit = limit
        self.process = subprocess.Popen(
            [sys.executable, '-I', os.path.abspath(__file__), '--interval', str(interval)] + list(uuids),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.history = []

    def _ask(self, request):
        self.process.stdin.write(request + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise ScratchError('Scratch monitor exited')
        reply = json.loads(line)
        if not reply.get('ok'):
            raise ScratchError('Scratch monitor: ' + reply.get('error', '?'))
        return reply

    def exclude_present(self):
        return self._ask('exclude')

    def set_context(self):
        return self._ask('context')

    def check(self, label):
        """End-of-case: raise ScratchError (unscored) if any device exceeded the cap."""
        reply = self._ask('take')
        self.history.append({'case': label, 'peak_bytes': reply['peak']})
        over = [(i, v) for i, v in enumerate(reply['peak']) if v > self.limit]
        if over:
            raise ScratchError('Candidate scratch cap exceeded (%s): %s' % (
                label, ', '.join('cuda:%d %.3f GiB > %.3f GiB' % (i, v / 2 ** 30, self.limit / 2 ** 30)
                                 for i, v in over)))
        return reply['peak']

    def close(self):
        if self.process.poll() is None:
            try:
                self._ask('stop')
            except Exception:
                pass
            try:
                self.process.wait(timeout=10)
            except Exception:
                self.process.kill()
                self.process.wait()


def device_uuids(torch):
    """NVML UUIDs of the visible CUDA devices, in CUDA order."""
    out = []
    for i in range(torch.cuda.device_count()):
        uuid = str(torch.cuda.get_device_properties(i).uuid)
        out.append(uuid if uuid.startswith(('GPU-', 'MIG-')) else 'GPU-' + uuid)
    return out


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 3 or args[0] != '--interval':
        raise SystemExit('usage: gpumem.py --interval SECONDS UUID...')
    serve(args[2:], float(args[1]))
