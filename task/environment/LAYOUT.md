# Qwen3.5 hybrid transfer kernel experiment

Scope: device-to-device state transfer only. No weights, model forward, inference server or NIXL service. Baseline is naive Torch, not a community performance claim. One Python process addresses disjoint source/destination GPU sets; CUDA IPC process overhead is excluded.

## Physical layouts and mapping

Each prefill rank owns a separate allocation; each decode rank has its own allocation on another GPU.
- KV: `[10 layers, physical page, local KV head, 128 token slots, K256|V256]`, BF16.
- Convolution: `[30 layers, physical state slot, 3 history steps, local Q|K|V channels]`, BF16.
- Recurrent: `[30 layers, physical state slot, local value head, 128,128]`, FP32.

For batch 2, ctx 129, each request has two logical pages. A legal example:

```
logical pages:       req0:0  req0:1  req1:0  req1:1
prefill page table:       8       2      11       5
decode page table:        3      12       1       9
prefill state slots: req0=4 req1=0
decode state slots:  req0=1 req1=6
```

The transfer maps 8→3, 2→12, 11→1, 5→9. It must not copy the intervening holes. Prefill and decode maps are independently sampled without replacement, with about 50% extra page capacity plus 17 guard pages. State pools have `2*batch+7` slots, independent maps and holes. The contiguous control has different source/destination offsets. Logical ordering remains per-request token order; physical ordering mixes requests. Fresh maps and payloads are generated for every invocation. Two initial generations explicitly test slot reuse and stale outputs.

Selected physical pages, including the unused final-token tail, must copy byte-exactly. The tail is explicit arbitrary payload, not meaningful model tokens; bandwidth reporting separates useful-token bytes from rounded page payload bytes. The benchmark does not model vLLM hybrid-manager page padding or aliasing. It uses source-derived tensor layouts, not live vLLM cache allocations. All indices/strides are supplied in the plan.

TP2 convolution must shard each Q/K/V segment independently. At TP1: Q2048|K2048|V4096. At TP2 each rank: Q1024|K1024|V2048. Recurrent heads32 split into16; full-attention KV heads2 split into1.

## Evaluation

128 cases =4 TP pairs ×8 lengths ×2 batches ×2 page layouts. Lengths:4096,8192,32768,65536,4097,10034,32749,65521. Batches1,4. Source snapshots, every destination state element, source immutability, and untouched holes are checked on every invocation. Deliberate recurrent corruption was rejected. Timing includes candidate metadata upload and all four device synchronizations, excludes payload generation/snapshots/checking/initial JIT. Initial full baseline:3 warmups+10 samples; final comparisons should use10+50. Both reference and candidates use the same timing boundary. Actual scratch accounting is a research contract, not a hardened malicious-code sandbox.

## Code for review

- `benchmark.py`: mapping generation, allocation, Torch reference, independent checker and timer.
- `remote.py`: submits a bounded job to the existing ticket dispatcher, fetches artifacts.
- `dispatch.py`: verifies holder PID and fresh ticket status, pins to the delegated node, kills its own subprocess on timeout/ticket loss; preserves the outer hold.
- `AGENTS.md`: allowed research scope and commands.

GPU work must run only under an operator-provided valid allocation. Machine identities, SSH settings, connection addresses and ticket records belong in private operator logs, not published task artifacts.

## Evaluator plan integrity

The evaluator retains private original maps and tensor references. Each candidate call receives cloned CPU maps and detached tensor aliases with independent shape metadata. After the call, validate every map, scalar and tensor descriptor against private originals, then check payloads using only evaluator-owned maps/references and private source snapshots. Changing candidate-visible selections cannot shrink the checked work. The mutable-plan bypass found by proposal review is covered by regression tests. This remains practical correctness enforcement, not a sandbox against arbitrary native-code memory introspection.
