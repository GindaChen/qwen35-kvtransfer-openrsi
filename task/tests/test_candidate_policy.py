"""Direct timer/checker/host-I/O tampering must fail before module import."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment"))
sys.path.insert(0, "/opt/kv-transfer")
import benchmark

CASES = [
    ("baseline", "from benchmark import reference as transfer\n", True),
    (
        "transfer",
        "import torch\nfrom torch.utils.cpp_extension import load_inline\ndef transfer(plan):\n pass\n",
        True,
    ),
    ("timer", "import time\ntime.perf_counter_ns=lambda:0\n", False),
    ("checker", "import sys\nsys.modules['__main__'].check=lambda *args:None\n", False),
    ("torch_equality", "import torch\ntorch.equal=lambda a,b:True\n", False),
    ("native_host_io", "src='system(\"do not run\");'\n", False),
    (
        "indirect_timer",
        "from torch.utils.cpp_extension import time as timer\ntimer.perf_counter_ns=lambda:0\n",
        False,
    ),
    ("builtin_introspection", "api=__builtins__['__import__']\n", False),
]


class CandidatePolicy(unittest.TestCase):
    def test_policy_cases(self):
        for name, source, allowed in CASES:
            with self.subTest(case=name):
                if allowed:
                    benchmark.validate_candidate_source(source)
                else:
                    with self.assertRaises(ValueError):
                        benchmark.validate_candidate_source(source)


if __name__ == "__main__":
    unittest.main()
