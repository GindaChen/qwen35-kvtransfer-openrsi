# Protocol 0.3: isolated candidate, trusted controller (2026-11-06)

Protocol 0.3 resolves the evaluation-integrity failure recorded in
[the 2026-11-03 audit](../evaluator-audit.md) and in both rubric reviews of
OpenRSI-Index discussion 316: the candidate no longer shares a process,
address space, clock, checker or writable result path with the evaluator.
Official scoring is re-enabled through this boundary only.

## Authority split

| Role | Process | Trusted | Owns |
| --- | --- | --- | --- |
| Judge adapter | `tests/evaluate.py` | yes | isolation-mode check, map seed, reward, `/logs/verifier` |
| Controller | `isolated_benchmark.py` (one per implementation) | yes, never imports candidate code | VMM buffers, payloads, snapshots, clock, capture, checks, result rows |
| Worker | `isolated_worker.py` under `gpu_isolate.py` → Harness `rsi-isolate` | **no** | imports `candidate.py`, runs `transfer(plan)` |

Channel: one AF_UNIX `SOCK_SEQPACKET` socketpair created by the controller.
Controller → worker: bounded JSON (`prepare` with dedicated CUDA VMM allocation
FDs, `transfer` with base64 int64 maps, `release`, `stop`). Worker → controller:
bounded JSON only; FDs are refused; the only accepted replies are the literal
`ready`/`prepared`/`done`/`released` objects echoing a controller nonce. No worker
clock, score, file or Python object is consumed (never pickle).

## What the worker cannot do (enforced, and probed on GB200)

| Attack class (from the audit) | Enforcement | Probe result (`tests/validate_isolation.py`) |
| --- | --- | --- |
| Torch dispatch override `aten::equal -> True` | checker runs in the controller process | `dispatch_override`: rejected |
| Native monkeypatch of checker/sync/timer (`PyRun_SimpleString` from a C++ extension) | same; worker clocks never read | `native_authority`: measured 42.1 ms ≥ forced 40 ms; follow-up no-transfer call rejected |
| Read controller memory (`/proc/<pid>/mem`, `environ`, `process_vm_readv/writev`) | controller `PR_SET_DUMPABLE=0`; Landlock ptrace scope; worker seccomp | EACCES / EPERM |
| Forge or overwrite result/reward files; edit evaluator code | Landlock: worker writes only its private HOME/TMPDIR; reads only task env, candidate dir, runtime | EACCES on private canary, reward path, `benchmark.py` |
| Signal the controller | worker seccomp: signals only to itself | EPERM |
| Reach any socket endpoint (IP, abstract or path AF_UNIX) | rsi-isolate `--no-network`; worker seccomp refuses `connect`, addressed `sendto`, AF_UNIX datagrams | EPERM; controller listeners saw no connection |
| Mutate maps/descriptors; skip a family; write source; stale output | controller-owned maps and buffers, independent bit-exact checker | all rejected |
| Acknowledge before GPU work completes | timed controller capture after the ack | `early_ack_unsynchronised`: 98.8 ms timed ≥ 97.7 ms queued sleep |
| Start work before the clock (pre-staging) | fresh payload is written into the source buffers only after the clock starts (timed publish) | structural; stale bytes are useless |
| Escape cleanup (double fork, new session) | rsi-isolate is the child subreaper and kills every descendant | n/a |
| Exceed the 2 GiB/GPU scratch cap with native allocations or a child process | trusted NVML monitor process (below) | `scratch_over_cap` (2.25 GiB native) and `scratch_child_process` (2.5 GiB in a child): rejected; `scratch_within_cap` (1.5 GiB native): passed |

## Two driver-compatibility rules (`gpu_isolate.py`)

The stock isolator makes `cuInit()` return 304 on both B200 and GB200 (measured):

1. The CUDA driver names helper threads by opening `/proc/self/task/<tid>/comm`
   for writing; Landlock denied it. Fix: `WRITE_FILE|TRUNCATE` beneath `/proc`
   (anchored at the permanent `/proc` root; per-pid dentries do not keep rules).
   Kernel checks still apply (own thread group only; `mem` needs ptrace access).
2. On kernels whose Landlock cannot scope abstract sockets (GB200: Linux 6.8,
   ABI 4) with a shared network namespace, `--no-network` refuses
   `socket(AF_UNIX)`; the driver must bind/listen its own abstract
   `cuda-uvmfd-<ns>-<pid>` socket. Fix: creation is kept, and the worker guard
   refuses every way to reach another endpoint (row above). Where the Judge has
   a private network namespace or Landlock ABI ≥ 6, the stock rule already
   allows this and nothing is relaxed.

No unconfined fallback exists: `gpu_isolate.py` refuses
`RSI_JUDGE_ISOLATION_FALLBACK` other than `none`, and `evaluate.py` refuses to
score unless `rsi-isolate isolation` reports `landlock`, `uid` or `userns`.

## Scratch cap: 2 GiB per GPU, enforced (`isolation/gpumem.py`)

The candidate may allocate at most 2 GiB per GPU beyond the fixed
source/destination pools. Native code (`cudaMalloc`, `cuMemCreate`, child
processes) bypasses any in-process allocator limit, so the cap is enforced
from outside the worker:

1. Before the worker starts, the controller creates its own CUDA context on
   all four GPUs and starts a trusted **monitor process** (non-dumpable, never
   shares memory with the worker). The monitor reads the driver's per-process
   GPU memory through NVML (`nvmlDeviceGetComputeRunningProcesses`) and records
   every PID already on the GPUs (the controller, anything pre-existing) as
   excluded. Every PID that appears later is the worker's process tree
   (rsi-isolate ends all descendants with the worker). PIDs are compared in
   NVML's own namespace, so this works in a container without host PIDs.
2. The worker creates a CUDA context on every device **before** importing the
   candidate and says `context`; the monitor records that usage as the
   worker's baseline (trusted code only). Only then does the controller send
   `load`.
3. Scratch on a device = worker-tree usage − baseline. The pools are created
   by the controller and only imported by the worker, so NVML charges them to
   the controller (measured: largest case, pools ≈ 5.6 GB per side, worker
   scratch 1.1 GiB for the Torch reference).
4. The monitor polls every 5 ms (`KV_SCRATCH_POLL_MS`) through every case and
   takes one synchronous sample at the end of the case. A peak above 2 GiB on
   any GPU raises `ScratchError`: the suite is unscored. Every result row
   carries `scratch_peak_bytes`, and `evaluate.py` re-checks them.

Limits, stated plainly: an allocation made and freed within one poll interval
(< ~5 ms) is not observed; lazily loaded kernels, library workspaces and the
caching allocator's reserve after the baseline count as scratch (the Torch
reference peaks at about 1.1 GiB on the largest cases); if NVML cannot report
per-process usage the monitor fails closed (unscored).

Preflight cases: `scratch_within_cap` (native `cuMemAlloc` of 1.5 GiB per GPU,
kept for the run) must pass; `scratch_over_cap` (2.25 GiB per GPU, native) and
`scratch_child_process` (a child process the candidate starts holds 2.5 GiB)
must be rejected with "scratch cap exceeded".

## Root (`uid`) mode in the task image

The Harness runs the Judge as root in the task container; `rsi-isolate` then
runs the worker as its own unprivileged uid (62000+) and also applies
Landlock where the kernel has it. The image was built from
`task/environment/Dockerfile` and `/tests/test.sh` was run in it as root, with
the Harness `rsi-isolate` mounted read-only at `/opt/rsi-isolate`,
`--network none` and four GPUs. This found and fixed three defects that would
have failed every official run: NumPy was missing from the image (the VMM
buffers are wrapped through `__cuda_array_interface__`), `python3-dev` was
missing (no `Python.h`, so no candidate could build a C++/CUDA extension), and
the preflight wrote its hostile candidates under `/logs/verifier`, which
`rsi-isolate` never grants. The only host available with Docker and root in a
container is an x86_64 B200 node: the GB200 node has no Docker, Podman,
Apptainer or sudo and runs jobs as a non-root cluster account (probed
2026-11-06; only Enroot is installed, which was not tried), so the image used
the x86_64 torch wheel and the results are marked non-scoring. Result:
`test.sh` exit 0 as root, mode `uid` (Landlock ABI 7 also applied), preflight
passed in 93 s, 192/192 for both implementations. See the evidence file for
what ran where.

## Timing boundary (changed from 0.2; numbers are not comparable)

Outside timing: per-case buffer allocation/publication, per-invocation maps,
payload generation, destination poison, expected snapshots, checks, JIT.
Inside timing, identical for reference and candidate except step 3:

1. publish: controller writes fresh payload into the source KV pages and
   selected state slots and synchronises source devices;
2. request with the maps (busy-polled socket on both sides);
3. `transfer(plan)`, the worker's four-device synchronisation, acknowledgement;
4. capture: controller-private copy of destination KV buffers and selected
   state slots, then synchronisation.

Steps 1, 2 and 4 add a fixed cost to both implementations, so protocol-0.3
speedups are lower than in-process 0.2 speedups for the same kernels,
especially for small cases. This is the price of a trustworthy clock.

## Evidence

See `protocol03-validation-summary.json` (repository root) for the GB200
adversarial preflight, the full matched 192-case pair and hashes, and its
`followup_20261106` section for the scratch-cap run on GB200 (preflight 124 s
wall, 192/192 for both, 2.12x, peak scratch reference 1.09 GiB / candidate
4 MiB) and the root `uid`-mode run of the task image.
