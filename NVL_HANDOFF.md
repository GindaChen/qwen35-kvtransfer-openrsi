# Continue natively on the allocated NVL node

## Current state
- OpenRSI Discussion: https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/316
- Original review: rejected only Evaluation Integrity, specifically mutable candidate-visible maps influencing the checker. All other proposal gates passed.
- Reported map bypass is corrected: independent evaluator-owned selections/descriptors; cloned input maps; detached payload aliases; post-call descriptor validation; source snapshots and output checking use private originals.
- Protocol0.2 adds broad finite bit-pattern payloads, bitwise equality, alternating guards and three allocation regimes. There are192 scored cases (four TP pairs x eight contexts x two batch sizes x three layouts). Old128-case results/curves remain historical evidence.
- A bounded candidate-source policy rejects direct timer/checker monkeypatching, reflection, non-transfer imports and common host file/process APIs before import. It is not proof against obfuscated arbitrary native code. Validate the real process/filesystem/isolation boundary before calling the task hardened.
- Post-initial-jump plot and trajectory privacy question were posted: https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/316#discussioncomment-19382928
- No raw native trace has been uploaded to GitHub. Do not upload one now.

## Completed checks
- Plan integrity:7 CPU regression tests passed on target runtime.
- Random layout/payload:5 CPU tests passed on target runtime, including signed-zero bit equality.
- Source policy:8 explicit acceptance/rejection cases checked; the preserved best candidate passes the policy.
- Protocol0.2 GPU smoke: reference and saved best pass all12 cases (all TP pairs, three layouts, batch1/context4097,1 warmup/2 samples). This is NOT a full192-case validation or score.
- Corrected-selection checker on the original128 shapes: both implementations pass a new50-sample run; matched geometric-mean speedup3.68881x. This is distinct from the historical3.8063x pilot result and from protocol0.2.
- Full192-case checks and official Harbor/Harness execution remain pending.

## Commands in a prepared target environment
Read the live nvl72 shared-Ray guide first. Use only a valid user-delegated allocation and its documented placement/runtime. Never start Ray, touch the infrastructure allocation, reuse unrelated GPU holds or publish operator connection details.

```bash
python task/tests/test_plan_integrity.py
python task/tests/test_layouts.py
python task/tests/test_candidate_policy.py
python task/environment/benchmark.py --warmup 10 --samples 50 --output reference-v02.jsonl
PYTHONPATH="$PWD/task/environment" python task/environment/benchmark.py --candidate /path/to/private/candidate.py --warmup 10 --samples 50 --output candidate-v02.jsonl
```

For paired map reproducibility, generate one random seed in the trusted controller and pass the same `--seed` to reference/candidate; keep all data fresh. For official scoring use task/tests/evaluate.py through the Harness, not a fabricated reward or converted pilot trace. Review the source policy and candidate import/compile boundary; add negative tests for any additional concrete bypass found. Do not claim a Python in-process policy is a native-code security sandbox.

## Official workflow after full validation
1. Fetch the current Discussion using proposal-agent's status helper before mutation. Reuse Discussion316; do not create another proposal.
2. Update the existing proposal with the latest immutable source commit and full protocol0.2 contract (192 cases, three layouts, new data/checker policy). Explain the review fix and actual validation, without changing historical pilot claims. Route any new scientific or budget decision to the user. Use the original approved six-hour agent budget unless the user agrees to change it.
3. Update the same Discussion with proposal-agent's helper and wait for that review to complete. A normal PASS starts task construction automatically; do not post `/task` just to start or hurry it. Answer only genuine task-defining questions.
4. At ACCEPTED, verify the generated private repository. Use its bundled validator and runner skills, and an operator-approved site profile. Do not improvise an unsupported Ray backend or owner-only scheduler commands. Actual container/isolation support on this site still needs validation.
5. Obtain an acceptable trajectory privacy protocol before upload. The standard helper copies original native bytes. Do not silently redact/convert a log and label it original; do not upload SSH configuration, connection endpoints, private addresses/paths or credentials. Native proposal logs remain on the original controller; the new NVL experiment log is separate. Preserve local originals and hashes. Never copy SSH keys or Codex auth files between machines for this contribution.
6. Run the actual task through RSI-Harness, collect its real results/trajectory, publish only approved privacy-safe artifacts, then share final workflow feedback. The custom six-hour pilot is supporting evidence, not retroactively a Harness run.

## Public source versus private operation
This repository contains no SSH launcher, raw native trace, runtime environment dump or operator ticket binding. A private operator handoff is available separately on the user's account. Some earlier public source revisions contained internal node/ticket identifiers; current documentation removes those identifiers. Do not infer that a current-file cleanup purges GitHub history/caches.
