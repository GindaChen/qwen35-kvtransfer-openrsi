# Gain through time — six-hour Qwen3.5 transfer-kernel pilot

Times are elapsed since 02:58:55 PST on November 3, 2026. Scores are geometric means across 128 cases against the fixed Torch reference. The table reports running-best observed scores; small record increments are not proof of a causal gain.

| Elapsed | Best observed score | What changed / what was learned |
|---|---:|---|
| 0 min — Baseline | 1.0000× | Torch gather → peer copy → scatter. |
| 5 min — First CUDA kernel | 3.4315× | Direct peer reads and destination writes remove payload-sized staging buffers and many framework operations. |
| 16.5 min — Metadata and fusion | 3.7445× | Reusable metadata buffers; fuse convolution/recurrent work, then fuse all three state families into one destination kernel. |
| 60 min — TP and tile tuning | 3.7689× | Specialize TP paths, tune threads and KV/recurrent tile sizes; reject source-side remote writes and unsuitable cache policies. |
| 120 min — CUDA graphs and metadata packing | 3.7999× | Reuse launch graphs, pack maps once, and use write-combined metadata staging for the two-destination path. |
| 180 min — Validation and rejected alternatives | 3.8013× | Higher-sample checks; test ordering, separate graph branches and other designs. Most changes fail to beat the retained kernel. |
| 240 min — Mapped metadata exploration | 3.8050× | Investigate mapped host metadata and topology-specific choices. Some early peaks shrink on replay. |
| 300 min — Final dispatch refinements | 3.8063× | Fixed graph-plan keys, exact cache-key comparisons and cached binding callables; final selected kernel measured with 50 samples. |
| 360 min — Confirmation and final rejections | 3.8063× | Independent selected-kernel replay gives 3.8056×. Later alternatives do not improve the selected 3.8063× kernel. |

The largest gain was the initial change from naive Torch to direct peer-copy CUDA. The first CUDA implementation scored 3.4315×; six hours of research raised the best to 3.8063×, an additional 10.92% in the score. This ratio corresponds to about 9.85% lower geometric-mean candidate latency relative to the first CUDA kernel.

These stages group multiple changes. They do not attribute the whole interval gain to one instruction or code change. Baseline used 10 samples; research trials used 10/20/50. Noise and repeated tests explain why some record scores were not adopted as new designs.
