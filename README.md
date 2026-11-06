# Qwen3.5 hybrid state transfer on GB200

OpenRSI contribution source by Junda Chen. Apache-2.0.

The starting reference, evaluator, and proposed Harbor1.4 task live in `task/`. This is a transfer-only benchmark derived from Qwen3.5-35B-A3B state shapes: full-attention KV, GDN convolution, and GDN recurrent state. It needs no model weights or forward pass.

Protocol 0.2 has 192 cases: four TP pairs (1→1,1→2,2→1,2→2), eight context lengths, batches 1/4, and three layouts (contiguous, independently random fragmentation, and independently fragmented runs with shuffled logical page pairs). All selected physical pages are transferred in full; sources and unselected holes must remain unchanged. See task/instruction.md and task/environment/LAYOUT.md.

A six-hour custom Codex/Ray pilot passed128 cases and reached3.8063x geometric-mean speedup over the Torch reference, with3.8056x confirmation. This is preliminary engineering evidence, not an official RSI-Harness result. Historical reference used10 samples; final candidates50. The proposed Harbor evaluator uses50 samples for both. Native proposal/research traces remain withheld pending an approved privacy-compatible protocol; no original native transcript has been uploaded.

The image recipe and Harbor adapter have passed static syntax/layout checks. Docker build, full dependency/base-image digest lock, official validator, and genuine Harness trajectory remain pending. Scoring now uses protocol 0.3 (below): the candidate runs only in an isolated worker. Source and task layout are public so the task builder can materialize the exact baseline. The winning kernel is not a starting artifact.

Sources: [DistServe](https://github.com/LLMServe/DistServe/tree/82831f1604cc6b10bebd360f6c437a07790dde9f), [SwiftTransformer](https://github.com/LLMServe/SwiftTransformer/tree/9948e31b371aac93caff73b8c0dc50cf77d890cb), [vLLM](https://github.com/vllm-project/vllm/tree/40837c5d0e5840c3143f917017cbfd31d07755e0), [Qwen configuration](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json).

## Review and privacy

The proposal review found that candidate-visible plan mutation could alter correctness selections. The revised evaluator keeps independent private maps/tensor descriptors, validates the exposed metadata, and checks original output allocations. Regression tests cover empty/rebound maps, in-place resize/value mutation, scalar mutation and output rebinding.

[Post-initial-jump research curve](docs/post-initial-jump.png) and [gain timeline](docs/gain-timeline.md) show measured pilot progress. Raw native transcripts contain operator connection details and are not published in this repository. A privacy-compatible trajectory delivery policy is being requested from maintainers. The historical pilot remains distinct from an official RSI-Harness run.

## Protocol 0.2 validation

The revised suite has192 cases with three allocation regimes and broader finite payloads. The complete matched GB200 development pair passed 192 cases for each implementation and measured **3.649696x** over naive Torch, with 10 warmups and 50 samples for each. This is not an official Harness score or a NIXL result. It retains evaluator-owned maps/descriptors, rejects candidate mutations and compares payload bits. The historical pilot and post-jump figure still describe protocol0.1's128 cases. See task/tests/test_plan_integrity.py and test_layouts.py. Full matched GB200 development checks are tracked in `protocol02-validation-summary.json`. Functional transfer checks are separate from evaluation integrity. [The integrity audit](docs/evaluator-audit.md) documents the reproduced protocol-0.2 bypasses; they apply to the in-process development runner only (`python tools/audit_boundary.py --in-process` exits 1 by design). No official validator or Harness run is claimed.

## Protocol 0.3: isolated evaluation (2026-11-06)

The scored path no longer imports candidate code into the evaluator. A trusted, non-dumpable controller (`task/environment/isolated_benchmark.py`) owns payloads, snapshots, the clock, the bit-exact checks and the result rows; the candidate's `transfer(plan)` runs only in a worker started through the Harness `rsi-isolate` (Landlock, no network, private HOME, subreaper cleanup) plus a worker seccomp guard. The two share only dedicated CUDA VMM buffers and a bounded JSON channel. `tests/evaluate.py` refuses to score without a confined isolation mode. Design, threat table and two measured CUDA-driver compatibility rules: [docs/protocol03/README.md](docs/protocol03/README.md).

On four GB200 GPUs (one delegated node, Landlock mode): the adversarial preflight `task/tests/validate_isolation.py` rejects or times every hostile candidate (dispatch override, native checker/timer monkeypatch, map mutation, omitted families, source writes, stale output, unsynchronised early acknowledgement) and every native attempt to read private files, write the reward or evaluator code, read controller memory, signal it or reach a socket is denied; `tools/audit_boundary.py` exits 0. The full matched 192-case pair through `tests/evaluate.py` passed 192/192 for the Torch reference and for the unchanged saved candidate, with a geometric-mean speedup of **2.24x** (all 192 cases faster; per TP pair 1.76x, 2.19x, 2.22x, 2.93x) (10 warmups, 50 samples). The timing boundary now includes a fixed controller publish, request round trip and capture for both implementations, so this number is lower than, and not comparable with, protocol 0.2's 3.65x. Evidence and hashes: `protocol03-validation-summary.json`. Remaining gaps: the 2 GiB/GPU scratch cap is not enforced; the Harness root (`uid`) mode and the container image have not been exercised.
