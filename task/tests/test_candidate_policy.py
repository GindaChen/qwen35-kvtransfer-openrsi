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
    ("import_alias", "import torch as api\napi.equal=lambda a,b:True\n", False),
    ("local_alias", "import torch\napi=torch\napi.equal=lambda a,b:True\n", False),
    ("container_alias", "import torch\na=[torch]\na[0].equal=lambda a,b:True\n", False),
    ("sync_alias", "from torch import cuda as api\napi.synchronize=lambda *a:None\n", False),
    ("tensor_class", "from torch import Tensor as T\nT.view=lambda a,*args:a\n", False),
    ("pickle_execution", 'import torch\ntorch.serialization.pickle.loads(b"cbuiltins\\nexec\\n(Vimport torch; torch.equal=lambda a,b:True\\ntR.")\n', False),
    ("pickle_import", "from torch.serialization import pickle as p\n", False),
    ("torch_save", "import torch\ntorch.save({}, 'reward.json')\n", False),
    ("torch_load", "import torch\ntorch.load('payload.pt', weights_only=False)\n", False),
    ("triton_load", "import triton.language as tl\nx=tl.load(ptr)\n", True),
    ("torch_load_alias", "from torch import load as run\n", False),
    ("wildcard", "from torch import *\n", False),
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
