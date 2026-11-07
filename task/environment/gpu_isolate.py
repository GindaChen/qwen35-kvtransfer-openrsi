"""Stock Harness `rsi-isolate` plus two narrow CUDA-driver compatibility rules.

Measured on B200 (Linux 6.17, Landlock ABI 7) and GB200 (Linux 6.8, ABI 4):
the unmodified isolator makes cuInit() return 304 (CUDA_ERROR_OPERATING_SYSTEM).
The driver names its helper threads through glibc, which opens
/proc/self/task/<tid>/comm for writing; Landlock grants no write under /proc,
so the open fails with EACCES and the driver aborts initialisation (strace:
docs/protocol03/cuinit-landlock.md). Two rules are added to the ruleset that
rsi-isolate itself builds, before its fork, and nothing else changes:

1. WRITE_FILE|TRUNCATE beneath /proc. A Landlock rule on an ephemeral procfs
   pid/task dentry does not survive re-instantiation, so the rule is anchored at
   the permanent /proc root. Writes there remain subject to the kernel's own
   checks: comm/attr only for the caller's own thread group, /proc/<pid>/mem
   needs ptrace access (denied: Landlock forbids ptrace out of the domain, the
   worker seccomp guard denies ptrace, and the controller is non-dumpable),
   sysctl and other global files need root (the worker never is).
2. MAKE_DIR on /dev, only if /dev is root-owned and not group/other writable:
   CUDA probes mkdir('/dev/nvidia-caps') and Landlock is checked before the
   EEXIST/EACCES DAC result. DAC still forbids creation by the unprivileged
   worker.

3. Only on kernels whose Landlock cannot scope abstract sockets (ABI < 6) and
   whose Judge shares a network namespace, rsi-isolate --no-network refuses
   socket(AF_UNIX); the CUDA driver must bind/listen its own abstract
   "cuda-uvmfd-<ns>-<pid>" SEQPACKET socket during cuInit (strace on GB200),
   so cuInit fails with 304. There socket(AF_UNIX) creation is kept, and the
   worker guard (isolation/guard.py) -- installed in the worker before CUDA or
   the candidate is loaded -- refuses connect(), sendto() with an address and
   AF_UNIX datagram sockets, so the worker still cannot reach any endpoint.
   IP and other socket families stay refused by rsi-isolate.

No read grant, IP-network rule, signal scope, uid choice or cleanup behaviour
of rsi-isolate is changed. No unconfined fallback exists. The worker must
install isolation.guard before importing torch (isolated_worker.py does, and
the controller only runs that entrypoint).
"""
import ctypes
import os
import signal
import stat
import sys
from pathlib import Path

LANDLOCK_MAKE_DIR = 1 << 7


def main():
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == '--controller-pid':
        parent = int(args[1])
        args = args[2:]
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0):  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), 'Cannot bind worker lifetime to controller')
        if os.getppid() != parent:
            raise RuntimeError('Controller exited during worker setup')
    library = Path('/opt/rsi-isolate')
    if len(args) >= 2 and args[0] == '--library':
        # Trusted development CLI only (hosts without a Harness mount).
        library = Path(args[1]).resolve(strict=True)
        args = args[2:]
    if not (library / 'rsi_isolate.py').is_file() or not (library / 'judge_isolation.py').is_file():
        raise RuntimeError('Harness isolation library is required')
    if os.environ.get('RSI_JUDGE_ISOLATION_FALLBACK', 'none') != 'none':
        raise RuntimeError('Unconfined fallback is forbidden')
    device = os.stat('/dev', follow_symlinks=False)
    if not stat.S_ISDIR(device.st_mode) or device.st_uid != 0 or device.st_mode & 0o022:
        raise RuntimeError('CUDA isolation requires root-owned, non-writable /dev')
    sys.path.insert(0, str(library))
    import rsi_isolate
    import judge_isolation

    original = rsi_isolate._ruleset

    def gpu_ruleset(*a, **k):
        ruleset = original(*a, **k)
        if ruleset is not None:
            ruleset.allow('/proc', judge_isolation.FS_WRITE_FILE | judge_isolation.FS_TRUNCATE)
            ruleset.allow('/dev', LANDLOCK_MAKE_DIR)
        return ruleset

    rsi_isolate._ruleset = gpu_ruleset

    original_deny_unix = rsi_isolate._deny_unix

    def gpu_deny_unix(no_network, ruleset):
        if not original_deny_unix(no_network, ruleset):
            return False
        # Kernel without Landlock abstract-socket scoping, shared network
        # namespace: the stock rule refuses socket(AF_UNIX) and cuInit fails.
        # The worker guard (isolation/guard.py), installed before CUDA or the
        # candidate loads, refuses connect(), addressed sendto() and AF_UNIX
        # datagram sockets, so no endpoint outside the worker is reachable.
        sys.stderr.write('gpu_isolate: AF_UNIX creation kept for cuInit; '
                         'worker guard refuses connect/addressed send/datagram\n')
        return False

    rsi_isolate._deny_unix = gpu_deny_unix
    return rsi_isolate.main(args)


if __name__ == '__main__':
    raise SystemExit(main())
