# Evaluator integrity audit (2026-11-03; resolution 2026-11-06)

> **Status 2026-11-06:** the boundary failures below are resolved for scoring by protocol 0.3 (trusted controller, candidate in an `rsi-isolate` worker). See [Protocol 0.3 resolution](#protocol-03-resolution-2026-11-06) at the end and [protocol03/README.md](protocol03/README.md). The in-process runner described below still exists for development only and is never scored; `python tools/audit_boundary.py --in-process` still exits 1 by design.

This audit separates functional checking from containment of an adversarial native candidate. Passing the former does not establish the latter. No official Harness acceptance or execution is claimed.

## Reproduced Python bypasses

At source revision `08b4fbb8d5f3841dae14c277300a214936fb3b93`, an import alias bypassed the module-root assignment check:

```python
import torch as api
api.equal = lambda a, b: True
```

The source policy accepted this code; executing it in a disposable CPU Python process made `bit_equal` accept unequal FP32 payloads. Local aliases, container aliases, CUDA synchronization aliases and tensor-class assignments had the same structural weakness. Attribute assignments/deletions are now rejected regardless of the spelling of their root. Tensor methods and indexing still support payload writes.

A second executed proof used `torch.serialization.pickle.loads` to execute a Python monkeypatch. Import-root restrictions did not exclude Torch's public host API re-exports. The policy now rejects those routes, serialization entry points and wildcard imports. Regression tests preserve the exact executed pickle example. The private saved candidate passed the revised policy without being added to the public task.

## Functional checker and result parsing

The existing seven plan tests protect against empty/rebound/resized/changed maps, scalar mutation, tensor descriptor changes and output rebinding. Five layout tests cover valid independent maps, reproducibility, fragmented schedules, finite broad payloads and signed-zero equality.

New CPU checker tests exercise a complete transfer, omission of each of the three state families, stale destination data after a new source generation, source mutation and writes to destination holes. They use the real head/channel counts with small CPU payload dimensions; they do not replace full GPU checks.

The result loader now requires literal `correct=true`, 50 finite positive samples and a median consistent with those samples. Tests reject duplicate/missing cases, truthy strings, forged medians, nonfinite/negative samples and wrong sample counts. This validates result structure, not the authenticity of a candidate-writable file.

## Remaining concrete boundary failures

A third **executed** CPU and GB200 CUDA proof uses `torch.library.Library("aten", "IMPL", dispatch).impl("equal", lambda a, b: True)`, with dispatch set to `CPU` or `CUDA`. The revised source policy accepts this public dispatch-registration API. Keeping the registry alive makes `bit_equal` accept unequal tensors. `tools/audit_boundary.py` reproduces this in a disposable process (`--device cuda:0` on a delegated GB200 GPU) and exits 1 when breached. It is an explicit remaining failing integrity probe, not a passing regression or a GPU transfer test. Native registration provides the same class of attack. This remains unresolved; no claim of evaluation-integrity validation is made.

`load_candidate` executes the candidate module in the same process as `check`, `check_source`, `sync` and the timer. Private Python references and detached tensor metadata do not separate native address spaces. The candidate may compile a C++/CUDA extension using the supplied extension API.

Two static probes still pass the bounded source policy: C++ calling `PyRun_SimpleStringFlags`, and POSIX `open`/`write` inside a native source string. These were not compiled or executed. They illustrate why a list of Python AST patterns and C API names cannot establish containment: native module initialization can reach Python state, and native code can access whatever files its process can write. Adding more regular expressions would not fix that boundary.

The outer `evaluate.py` process starts separate reference/candidate subprocesses, but the candidate subprocess contains its own checker, timer and result writer. Its JSONL is therefore not an independently authenticated measurement. In a development run where the reference JSONL shares a writable result directory, native candidate code can also replace the already-measured reference file with plausible rows; checking sample count and median consistency cannot establish who measured them. Read-only Work snapshots protect source files; they do not protect Python objects within the candidate process or automatically make its writable results trustworthy. Likewise, fresh images and no-network policies do not separate an imported native module from its own checker.

The declared 2 GiB/GPU scratch limit is also not enforced: Torch peak allocation statistics are recorded but not compared to a bound, and direct native CUDA allocations are not covered by that counter.

The task currently does not invoke the Harness's `rsi-isolate` facility. That facility can isolate a child command from a trusted Judge; wrapping the entire present benchmark would still place its checker and timer inside the untrusted child. A complete fix needs an evaluator-owned process retaining snapshots, checking outputs and measuring completion across an explicitly designed candidate boundary, plus denied access to result files and Judge memory. The actual site must demonstrate filesystem, process, GPU-sharing and cleanup behavior with adversarial probes. No such boundary is claimed here.

The official adapter now refuses scoring before candidate import and writes an unscored failure, with a regression test requiring no reward file. Development benchmark runs remain available. This is a fail-closed stop, not a completed isolation implementation.

A process/IPC refactor can affect the current callable latency metric and cache/setup contract. Its transfer and timing semantics need to be explicit before those measurements are treated as the accepted scientific protocol. Until then, full-suite runs are development functional/performance evidence only, and the integrity issue remains open.

## Matched protocol-0.2 development evidence

Both the reference and unchanged saved candidate passed all 192 cases on four GB200 GPUs, with matched map seeds, 10 warmups and 50 measured samples per case. The observed geometric-mean speedup was 3.649695744x over naive Torch; all 192 case medians improved. The pair took 673.96 seconds, including candidate setup and correctness checking. Raw timing rows and hardware-only metadata are preserved under `docs/protocol02/`; hashes and limitations are in `protocol02-validation-summary.json`. The benchmark bytes match tested commit `dd6ea94fdfcb4e71d5a26af88e8169cc1527365c`. These measurements do not resolve the integrity failures above.

The earlier 3.8063x result and learning curves remain the historical 128-case protocol-0.1 pilot. No new research learning curve or official Harness trajectory has been generated. Raw native transcripts and operator logs remain withheld for privacy review.

The GB200 follow-up also passed all three result-parser/fail-closed tests. A read-only target capability probe found no Docker, Apptainer, RSI-Harness executable or Docker socket; a supported operator site profile has not been established. No extra allocation or unsupported Ray Harness backend was introduced.

## Protocol 0.3 resolution (2026-11-06)

Each remaining failure above, and its status under the scored protocol-0.3 path:

| 2026-11-03 failure | Protocol 0.3 | Evidence on 4×GB200 |
| --- | --- | --- |
| `torch.library` dispatch override makes `bit_equal` accept unequal tensors | the checker runs in the controller, which never imports candidate code | `dispatch_override` rejected; `tools/audit_boundary.py` (isolated) exits 0 |
| `load_candidate` executes native code beside checker/timer/result writer | `load_candidate` runs only in the isolated worker; clock, payload, snapshots, checks and rows are controller-owned | native `PyRun_SimpleString` monkeypatch of checker/sync/timer had no effect: forced 40 ms wait measured, no-transfer call rejected |
| Native file access can forge result rows / overwrite the reference file | Landlock: worker writes only its private HOME/TMPDIR; `/logs/verifier` is never granted; rows written by controllers | write to private reward path and to `benchmark.py`: EACCES |
| Native code can reach Judge memory | controller and Judge are non-dumpable; worker seccomp denies ptrace/process_vm_*; Landlock ptrace scope | `/proc/<controller>/mem`, `environ`: EACCES; `process_vm_readv/writev`: EPERM |
| `PyRun_SimpleStringFlags` and POSIX `open/write` in native sources pass the policy | irrelevant to scoring: executed inside the worker by the probe above | contained |
| Harness `rsi-isolate` not invoked | invoked for every candidate and reference worker through `gpu_isolate.py` (two measured CUDA compatibility rules) | Landlock mode, no fallback |
| Early acknowledgement / work before the clock | timed fresh-payload publish before the request and timed private capture after the ack | queued 97.7 ms GPU sleep measured as 98.8 ms |

Still open: the 2 GiB/GPU scratch cap is a contract, not enforced; the actual
Harness site profile (container, root `uid` mode) has not been exercised, only
the Landlock mode on the delegated GB200 node; the image has not been built.
