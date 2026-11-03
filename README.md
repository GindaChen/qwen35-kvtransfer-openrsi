# Qwen3.5 hybrid state transfer on GB200

OpenRSI contribution source by Junda Chen. Apache-2.0.

The starting reference, evaluator, and proposed Harbor1.4 task live in `task/`. This is a transfer-only benchmark derived from Qwen3.5-35B-A3B state shapes: full-attention KV, GDN convolution, and GDN recurrent state. It needs no model weights or forward pass.

Four TP pairs (1→1,1→2,2→1,2→2), eight context lengths, batches1/4, contiguous and independently fragmented page maps form128 cases. All selected physical pages are transferred in full; sources and unselected holes must remain unchanged. See task/instruction.md and task/environment/LAYOUT.md.

A six-hour custom Codex/Ray pilot passed128 cases and reached3.8063x geometric-mean speedup over the Torch reference, with3.8056x confirmation. This is preliminary engineering evidence, not an official RSI-Harness result. Historical reference used10 samples; final candidates50. The proposed Harbor evaluator uses50 samples for both. Native proposal/research traces will be delivered to the generated private task repository through the official workflow.

The image recipe and Harbor adapter have passed static syntax/layout checks. Docker build, full dependency/base-image digest lock, official validator, and genuine Harness trajectory remain pending. Source and task layout are public so the task builder can materialize the exact baseline. The winning kernel is not a starting artifact.

Sources: [DistServe](https://github.com/LLMServe/DistServe/tree/82831f1604cc6b10bebd360f6c437a07790dde9f), [SwiftTransformer](https://github.com/LLMServe/SwiftTransformer/tree/9948e31b371aac93caff73b8c0dc50cf77d890cb), [vLLM](https://github.com/vllm-project/vllm/tree/40837c5d0e5840c3143f917017cbfd31d07755e0), [Qwen configuration](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json).

## Review and privacy

The proposal review found that candidate-visible plan mutation could alter correctness selections. The revised evaluator keeps independent private maps/tensor descriptors, validates the exposed metadata, and checks original output allocations. Regression tests cover empty/rebound maps, in-place resize/value mutation, scalar mutation and output rebinding.

[Post-initial-jump research curve](docs/post-initial-jump.png) and [gain timeline](docs/gain-timeline.md) show measured pilot progress. Raw native transcripts contain operator connection details and are not published in this repository. A privacy-compatible trajectory delivery policy is being requested from maintainers. The historical pilot remains distinct from an official RSI-Harness run.
