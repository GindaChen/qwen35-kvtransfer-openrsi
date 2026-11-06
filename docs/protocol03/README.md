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
adversarial preflight, the full matched 192-case pair and hashes.
