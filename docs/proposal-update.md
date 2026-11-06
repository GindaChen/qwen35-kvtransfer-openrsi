# Proposal field edits for OpenRSI-Index discussion 316 (after the 2026-11-06 PASS)

Not posted. For the contributor to review and apply. Each entry is the full
new text of one field of the proposal table (Section | Field). Fields not
listed stay as they are. `<COMMIT>` = the head commit of branch
`isolated-evaluator-protocol03` that contains this file.

Evidence for every number: `protocol03-validation-summary.json`, section
`followup_20261106`.

---

**Status line (top of proposal)**

Continuation update (2026-11-06, evening): the obligations from the PASS
review are closed. The 2 GiB/GPU scratch cap is enforced by the Judge; the task
image was built and `/tests/test.sh` ran in it as root with the Harness
`rsi-isolate` in `uid` mode; stale protocol-0.2 text is relabelled. No
official Harness result is claimed.

**Research Question | Repository URL**

[LLMServe/DistServe](https://github.com/LLMServe/DistServe), with a public
task-owned Qwen3.5 transfer overlay:
[`<COMMIT>`](https://github.com/GindaChen/qwen35-kvtransfer-openrsi/tree/<COMMIT>)
(protocol 0.3). Public source includes the reference, isolated evaluator
(`task/environment/isolated_benchmark.py`, `isolated_worker.py`,
`gpu_isolate.py`, `isolation/` including the scratch monitor `gpumem.py`),
development evaluator, tests, adversarial preflight
(`task/tests/validate_isolation.py`), integrity audit, image recipe and
official adapter. No private infrastructure launcher or winning candidate is
included.

**Research Question | Exact commit/tag**

DistServe `82831f1604cc6b10bebd360f6c437a07790dde9f`; SwiftTransformer
`9948e31b371aac93caff73b8c0dc50cf77d890cb`; vLLM
`40837c5d0e5840c3143f917017cbfd31d07755e0`. Current task overlay `<COMMIT>`
(protocol 0.3). The GB200 and container validation runs executed exactly the
committed evaluator files (per-file SHA256 in
`protocol03-validation-summary.json`). The scored workload (cases, layouts,
sampling) is unchanged since protocol 0.2.

**Research Question | Files/components the agent may modify**

Candidate CUDA/C++/Triton/Python source, build configuration, transfer
descriptors, stream scheduling, batched copies, push/pull strategy and bounded
scratch allocation. Scratch: at most 2 GiB per GPU in addition to the fixed
source/destination pools, **enforced by the Judge**: a trusted monitor process
charges everything the candidate worker's process tree holds on each GPU
(NVML per-process accounting, so native allocations and child processes
count) above its CUDA context created before the candidate is imported; above
2 GiB on any GPU the run is unscored. Persistent setup may depend on
allocation topology, not request contents or future mappings.

**Reference Baseline | Baseline evaluation path**

Official: `/tests/test.sh` → unit tests → `tests/validate_isolation.py`
(hostile candidates, including scratch-cap violations, must be contained on
the actual site) → `tests/evaluate.py`, which refuses to score unless
`rsi-isolate isolation` reports a confined mode, then runs
`isolated_benchmark.py` for the Torch reference and for the candidate with one
trusted map seed passed by environment. Both run serially on the same
four-GB200 node. Development: `benchmark.py` (in-process, never scored) or
`isolated_benchmark.py`. No official Harness execution is claimed.

**Reference Baseline | Baseline metric reported by the repository**

Historical protocol-0.1 pilot: 128 cases, 3.80626658x over naive Torch
(custom-run evidence). Historical protocol-0.2 development pair (in-process
timing): 3.649696x, 192/192. Protocol 0.3 (isolated; timing includes a fixed
controller publish, request and capture for both implementations): the same
saved candidate measured 2.24x and, in a second full run on the same GB200 node
with the scratch cap enforced, **2.12x** (192/192 correct for both, all 192
faster); the reference was about 6% faster in the second run, so about ±6% is
run-to-run drift between separate matched pairs. 0.3 and 0.2 speedups are not
comparable. Not an RSI-Harness or NIXL result.

**Evaluation | Fixed evaluation protocol**

Protocol 0.3: four disjoint TP pairs `(1,1),(1,2),(2,1),(2,2)` × contexts
`4096,8192,32768,65536,4097,10034,32749,65521` × batches `1,4` × three layouts
= **192 cases**. Layouts: contiguous; independent random source/destination
fragmentation; independently fragmented runs with shuffled logical page pairs.
State-slot maps are independently seeded. The candidate's `transfer(plan)`
runs only in an isolated worker (Harness `rsi-isolate` plus a seccomp guard);
a trusted controller owns the dedicated GPU buffers, maps, fresh broad finite
BF16/FP32 payloads, alternating destination poison, snapshots, clock, checks
and result rows. Raw-bit checks cover selected outputs, unchanged sources and
unchanged holes; signed zero matters. All attention, convolution and
recurrent layers are required; selected page tails transfer in full. Two
initial correctness calls and every one of 10 warmups + 50 timed invocations
are checked. Timed: controller publish of fresh source payload, the request,
`transfer` through all-device synchronisation and acknowledgement, and a
controller-private capture of the destination; identical fixed parts for
reference and candidate. Untimed: allocation, payload generation, snapshots,
checks, first-time compilation. Scratch above 2 GiB/GPU, compile/crash/
illegal-memory/OOM/correctness/timeout/incomplete and infrastructure outcomes
are unscored. One submission requires the entire suite; no case replacement,
dropping or resampling.

**Evaluation | Reward or score definition**

Higher is better: geometric mean of reference_median_us/candidate_median_us
across all **192 equally weighted cases**, both measured by trusted
controllers in the same isolated path. Reference nominal value 1. Every case
and measurement must complete. Invalid, over-scratch, infrastructure and
incomplete outcomes are unscored, not a finite penalty scalar. Result rows are
written only by the controllers; the adapter additionally checks literal
correctness flags, 50 finite positive samples, exact case coverage, median
consistency and per-GPU scratch peaks ≤ 2 GiB before writing the reward.

**Evaluation | Measures preventing reward hacking**

Append to the current text: "The 2 GiB/GPU scratch cap is enforced by a
trusted NVML monitor process outside the worker; on 4×GB200 the preflight
passes a native 1.5 GiB/GPU allocation and rejects a native 2.25 GiB/GPU
allocation and a candidate-started child process holding 2.5 GiB." Delete the
sentence "Remaining gap: the 2 GiB/GPU scratch cap is a contract, not
enforced."

**Evaluation | Noise handling and meaningful improvement**

Reference and candidate run serially on the same held GB200 node with matched
10/50 sampling and identical map seeds. Protocol-0.3 run of 2026-11-06
(evening): median within-case relative sample IQR 0.82% (reference) and 0.52%
(candidate). Two separate matched pairs of the same candidate on the same node
gave 2.24x and 2.12x (reference about 6% faster in the second), so differences
below roughly ±6% between separate evaluations are not meaningful. Repeat
matched evaluations when variability could change a conclusion; preserve
incomplete or noisy outcomes.

**Workspace | Leakage and reward-hacking safeguards**

No runtime network downloads; no winning pilot kernel or native logs in
starting task artifacts. Candidate code runs only in the isolated worker; it
cannot read evaluator state, write results or the reward, reach any network
or socket endpoint, or exceed the scratch cap unnoticed. Validated natively
(Landlock mode) on one 4×GB200 node, and in the task image as root (`uid`
mode, `--network none`) on an x86_64 B200 host: `/tests/test.sh` exit 0,
preflight passed, 192/192 for both implementations. That run found and fixed
four defects (missing NumPy, missing `python3-dev`, preflight fixtures under
`/logs/verifier`, a test path). The arm64 image on the GB200 site has not been
run in a container; the Judge re-runs the adversarial preflight on the actual
site before scoring and stops unscored if any hostile case is not contained.

**Compute Feasibility | Compute resources per single experiment run**

Work and Judge each: one physical node, at most 4 NVIDIA GB200 GPUs, 16 CPU
cores (if applicable), 256 GiB RAM (if applicable), builds capped at 16 jobs.
Serial phases have a four-GPU peak. TP1→TP1 uses 2 active devices, unequal
pairs 3, TP2→TP2 four. Candidate scratch is capped at 2 GiB/GPU beyond the
fixed pools and enforced by the Judge.

**Compute Feasibility | Estimated runtime per single experiment run**

The historical six-hour custom pilot remains separate. Judge on one 4×GB200
node (protocol 0.3, measured 2026-11-06): unit tests seconds; adversarial
preflight **124 s** wall; matched 192-case pair **846 s** (about 7 min per
implementation, including JIT and checks); about 17 min in total. Prior
provisional bounds: up to 30 min Work plus 60 min Judge, serial. End-to-end
supported Harness timing remains Estimate incomplete. The contributor
explicitly retains a **six-hour Agent budget**; no automatic 24/48-hour
expansion is authorized.
