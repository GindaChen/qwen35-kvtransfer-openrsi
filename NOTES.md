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
