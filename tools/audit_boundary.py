"""Integrity probe: the dispatch-registration override that breached protocol 0.2.

The probe registers `aten::equal -> True` for CPU and CUDA and transfers nothing.

  python tools/audit_boundary.py --in-process
      Legacy protocol-0.2 development runner (candidate shares the checker's
      process). Exits 1: the override makes unequal tensors compare equal.
      This runner is development-only and never produces a score.

  python tools/audit_boundary.py [--isolator DIR] [--allow-hardware]
      Protocol-0.3 scored path: the same source runs in an rsi-isolate worker,
      the trusted controller checks the output. Exits 0 when the candidate is
      rejected (contained), 1 if it were accepted. Needs four GPUs.

The full adversarial suite is task/tests/validate_isolation.py.
"""
import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'task/environment'))
import torch
import benchmark

SOURCE = '''import torch
registries = []
for dispatch in ["CPU", "CUDA"]:
    registry = torch.library.Library("aten", "IMPL", dispatch)
    registry.impl("equal", lambda a, b: True)
    registries.append(registry)
def transfer(plan):
    pass
'''


def in_process():
    benchmark.validate_candidate_source(SOURCE)  # the policy accepts it
    a, b = torch.tensor([1.0]), torch.tensor([2.0])
    before = benchmark.bit_equal(a, b)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        exec(SOURCE, {})
    after = benchmark.bit_equal(a, b)
    print(json.dumps({'runner': 'in-process development (protocol 0.2)', 'source_policy': 'accepted',
                      'unequal_payloads_equal_before': before, 'unequal_payloads_equal_after': after,
                      'integrity_probe': 'breached' if after else 'not_reproduced'}))
    return 1 if after else 0


def isolated(isolator):
    from isolated_benchmark import Worker, run_case
    with tempfile.TemporaryDirectory() as tmp:
        fixture = Path(tmp) / 'candidates'
        fixture.mkdir()
        path = fixture / 'candidate.py'
        path.write_text(SOURCE)
        worker = Worker(path, Path(tmp) / 'worker.log', isolator)
        try:
            run_case(worker, 1, 1, 1, 129, 'fragmented', 0, 1, 12345)
        except AssertionError as exc:
            print(json.dumps({'runner': 'isolated controller (protocol 0.3)', 'source_policy': 'accepted',
                              'integrity_probe': 'contained', 'rejection': str(exc)}))
            return 0
        finally:
            worker.close()
    print(json.dumps({'runner': 'isolated controller (protocol 0.3)', 'integrity_probe': 'breached'}))
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-process', action='store_true')
    parser.add_argument('--isolator')
    args = parser.parse_args()
    return in_process() if args.in_process else isolated(args.isolator)


if __name__ == '__main__':
    raise SystemExit(main())
