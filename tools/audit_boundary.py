"""Disposable CPU proof: an accepted candidate can corrupt an in-process checker.

Exit 1 means the current boundary is breached. This is an integrity probe,
not a transfer benchmark or a native trajectory. Never import untrusted code
in the caller's process. Run this script in a disposable development process.
"""
import json
import sys
import warnings
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'task/environment'))
import benchmark

source='''import torch
registry = torch.library.Library("aten", "IMPL", "CPU")
registry.impl("equal", lambda a, b: True)
def transfer(plan):
    pass
'''
try:
    benchmark.validate_candidate_source(source)
except ValueError:
    print(json.dumps({'source_policy':'rejected','integrity_probe':'blocked'}))
    raise SystemExit(0)
a=torch.tensor([1.0]); b=torch.tensor([2.0])
before=benchmark.bit_equal(a,b)
# Registry lifetime must span the comparison. This process exits immediately.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    exec(source,globals())
after=benchmark.bit_equal(a,b)
print(json.dumps({'source_policy':'accepted','unequal_payloads_equal_before':before,
                  'unequal_payloads_equal_after':after,
                  'integrity_probe':'breached' if after else 'not_reproduced'}))
raise SystemExit(1 if after else 0)
