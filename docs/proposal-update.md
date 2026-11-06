# Draft: proposal field updates for OpenRSI-Index discussion 316

Not posted. For the contributor to review and paste. Replace `<COMMIT>` with
the merged commit of this branch. Only fields affected by protocol 0.3 change.

**Status line (top of proposal).** Protocol 0.3 implements the evaluator
authority boundary: the candidate runs only in a Harness `rsi-isolate` worker;
a trusted controller owns payloads, snapshots, clock, checks and results.
Official scoring is re-enabled through that boundary and refuses to run
without it. Adversarial preflight and a full matched 192-case pair pass on
4×GB200. No official Harness execution is claimed.

**Research Question / Exact commit.** Task overlay `<COMMIT>` (protocol 0.3).

**Reference Baseline / Baseline evaluation path.** Official: `/tests/test.sh`
→ unit tests → `tests/validate_isolation.py` (hostile candidates must be
contained on the actual site) → `tests/evaluate.py`, which refuses to score
unless `rsi-isolate isolation` reports a confined mode, then runs
`isolated_benchmark.py` for the Torch reference and for the candidate with one
map seed passed by environment. Development: `benchmark.py` (in-process,
never scored) or `isolated_benchmark.py`.

**Evaluation / Fixed evaluation protocol.** Unchanged workload (192 cases,
10 warmups + 50 samples, every invocation bit-exactly checked). Timing
boundary (both implementations): timed controller publish of fresh source
payload; request; `transfer(plan)` through all-device synchronisation and
acknowledgement; timed controller-private capture of the destination. Untimed:
allocation, payload generation, poison, snapshots, checks, compilation.

**Evaluation / Evaluation information hidden from the agent.** Expected
snapshots, maps before each request, the map seed (environment only, not
argv), controller memory (non-dumpable) and `/logs/verifier` (never granted)
are inaccessible to the candidate process.

**Evaluation / Measures preventing reward hacking.** Replace the "Integrity
remains blocked" sentence with: The 2026-11-03 dispatch-override breach is
contained: the candidate executes only in an isolated worker (Landlock
read-only task/candidate/runtime, private HOME, no `/logs/verifier`, no
network; seccomp: no ptrace/process_vm_*, no signals to other processes, no
`connect`/addressed `sendto`/AF_UNIX datagrams, no io_uring/bpf/namespaces;
subreaper cleanup), sharing only dedicated CUDA VMM buffers and a JSON
channel with a non-dumpable controller that never imports candidate code.
On 4×GB200 the adversarial preflight rejects dispatch override, empty maps,
omitted families, source writes and stale output; a native
`PyRun_SimpleString` monkeypatch of checker/sync/timer has no effect; native
reads of the private canary, writes to the reward path and evaluator code,
`/proc/<controller>/mem|environ`, `process_vm_readv/writev`, signals and
socket connections are all denied; an unsynchronised early acknowledgement is
timed (queued 97.7 ms GPU sleep measured 98.8 ms). Two CUDA-driver
compatibility rules for `rsi-isolate` are documented with strace evidence.
Remaining gap: the 2 GiB/GPU scratch cap is a contract, not enforced.

**Workspace / Leakage and reward-hacking safeguards.** Replace "The current
shared-process checker/result boundary has a reproduced bypass; official
scoring is disabled" with the boundary summary above and "Actual container
(root `uid` mode) validation on the operator site remains to be run; the
Judge re-runs the adversarial preflight there before scoring."

**Data and Network Boundaries.** Candidate result forging is prevented: result
rows and the reward are written only by trusted processes to a directory the
worker cannot read or write; the worker cannot reach any socket endpoint.

**Compute Feasibility / Estimated runtime.** On one 4×GB200 node the matched
192-case pair took 885 s (about 7.4 min per implementation, including JIT and
checks); preflight wall time was not recorded. Measured geometric-mean
speedup of the saved candidate under protocol 0.3: 2.24x (192/192 correct,
192 wins).
