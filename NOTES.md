# Notes: protocol 0.3 isolation fix (2026-11-06)

## Judge criteria being addressed

OpenRSI-Index discussion 316, latest rubric review (Reject). Passing gates:
expertise, source repository, scope, traceable baseline, objective/metric,
action space. Failing gates, all from one root cause:

| Gate | Reviewer's reason |
| --- | --- |
| Evaluation Integrity | reproduced Torch dispatch override; candidate native code shares checker/timer memory; candidate-writable result paths; adapter refuses scoring |
| Data and Network Boundaries | candidate can subvert comparisons in-process and forge result rows |
| Task-Generation Readiness | official entrypoint terminates unscored; no implemented candidate/checker authority separation |

The reviewer asks for "a trusted evaluator process and genuinely isolated
candidate boundary, with protected snapshots/clocks/results and adversarial
runtime probes" (audit wording). The judge re-runs only when the proposal text
in the discussion is edited; this branch supplies the implementation and
evidence, and `docs/proposal-update.md` drafts the text changes.

## Root cause

`benchmark.py --candidate` imports candidate code into the process that holds
the checker, the clock and the result writer. No source policy can contain
native code there.

## Fix (design in docs/protocol03/README.md)

Trusted controller + untrusted worker under the Harness `rsi-isolate`:
dedicated CUDA VMM buffers shared by FD, JSON-only channel, controller-owned
clock/payload/snapshots/checks/results, timed fresh-payload publish and timed
private capture, worker seccomp guard. Two driver-compatibility rules were
needed (CUDA thread naming under /proc; CUDA's abstract uvmfd socket on
Landlock ABI 4), both measured by strace and documented in `gpu_isolate.py`.

Prior unfinished attempt (2026-11-03, not published) used per-invocation VMM
export inside timing: 8 ms import overhead per call against ~0.2–2 ms
transfers, and CUDA could not initialise under the isolator. Both resolved.

## Decisions taken without the user (asleep)

- Used the prior attempt's VMM/wire/guard design as the base.
- Timing boundary changed (publish + request + capture are timed for both
  implementations). Speedups are therefore lower than protocol 0.2; this is
  stated, not hidden.
- Allowed `setpgid/setsid` in the worker guard (ninja needs them; rsi-isolate
  ends descendants as subreaper; signals stay confined).
- Map seed passed via environment (cmdline is world-readable via /proc).
- Scratch cap (2 GiB/GPU) is still not enforced; recorded as remaining gap.
- Driver debugging also used a separate B200 host (no B200 number is
  evidence). All GB200 evidence comes from one newly queued four-GB200 node;
  the contributor's existing holds were in use by other work and untouched.
- Nothing was posted to the discussion.

## Evidence

See protocol03-validation-summary.json.

## Follow-up after the judge's PASS (2026-11-06, evening)

The PASS review on discussion 316 listed open obligations; this round closes them.

| Obligation from the review | Done |
| --- | --- |
| 2 GiB/GPU scratch cap "explicitly not yet enforced" | Enforced from outside the worker: trusted NVML monitor process, per-PID accounting of the worker tree above its pre-candidate CUDA context (`task/environment/isolation/gpumem.py`); above the cap the suite is unscored. Preflight adds a native 1.5 GiB/GPU case (must pass), a native 2.25 GiB/GPU case and a 2.5 GiB child-process case (must be rejected). |
| root/uid-mode container validation pending | Image built from `task/environment/Dockerfile`; `/tests/test.sh` run in it as root with the Harness `rsi-isolate` mounted, `--network none`: mode `uid`. Only host with root-in-container: x86_64 B200 (x86_64 torch wheel; non-scoring). GB200 site has no Docker/Podman/root. |
| stale protocol-0.2 labels | 0.2 sections labelled historical; `task.toml` version 0.3.0; instruction text. |
| preflight wall time not recorded | `validate_isolation.py` now records `wall_seconds` and per-case seconds. |

Defects the container run found (each would have failed every official run):
1. NumPy missing from the image (`torch.as_tensor(__cuda_array_interface__)` for the VMM buffers).
2. `python3-dev` missing: no `Python.h`, so no candidate could build a C++/CUDA extension.
3. The preflight wrote hostile candidate sources under `/logs/verifier`, which `rsi-isolate` never grants (refusal on `--read`). Fixtures now go to a separate world-readable temp dir; private canaries stay under the verifier dir.
4. `test_guard.py` set `PYTHONPATH` to the repository layout path, absent in the image.

Decisions taken without the user: the cap counts everything the worker tree holds above its pre-import context (lazily loaded kernels and allocator reserve included) -- the Torch reference peaks around 1.1 GiB on the largest cases, leaving the cap meaningful; poll interval 5 ms; NVML unavailable means unscored (fail closed).
