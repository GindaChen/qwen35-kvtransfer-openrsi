"""Structural allocation and opaque-payload regression tests; CPU-only."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment"))
sys.path.insert(0, "/opt/kv-transfer")
import benchmark


class Layouts(unittest.TestCase):
    def test_layouts_are_valid_unique_and_leave_holes(self):
        for layout in benchmark.LAYOUTS:
            for bs, ctx in [(1, 4097), (4, 10034), (4, 65536)]:
                p = benchmark.maps(bs, ctx, layout, 1234)
                n = bs * ((ctx + 127) // 128)
                for name, cap, count in [
                    ("src_pages", p["cap"], n),
                    ("dst_pages", p["cap"], n),
                    ("src_slots", p["slotcap"], bs),
                    ("dst_slots", p["slotcap"], bs),
                ]:
                    x = p[name]
                    self.assertEqual(x.numel(), count)
                    self.assertEqual(x.unique().numel(), count)
                    self.assertGreaterEqual(x.min().item(), 0)
                    self.assertLess(x.max().item(), cap)
                    self.assertGreater(cap, count)

    def test_reproducible_seed_and_independent_sides(self):
        for layout in ["fragmented", "fragmented_runs"]:
            a = benchmark.maps(4, 10034, layout, 42)
            b = benchmark.maps(4, 10034, layout, 42)
            c = benchmark.maps(4, 10034, layout, 43)
            for name in ["src_pages", "dst_pages", "src_slots", "dst_slots"]:
                self.assertTrue(torch.equal(a[name], b[name]))
            self.assertFalse(torch.equal(a["src_pages"], a["dst_pages"]))
            self.assertFalse(torch.equal(a["src_pages"], c["src_pages"]))

    def test_fragmented_schedules_are_not_sequential(self):
        for layout in ["fragmented", "fragmented_runs"]:
            p = benchmark.maps(4, 10034, layout, 5)
            for name in ["src_pages", "dst_pages"]:
                x = p[name]
                self.assertFalse(bool((x[1:] > x[:-1]).all()))
                self.assertGreater((x[1:] - x[:-1]).abs().max().item(), 8)

    def test_all_payloads_are_finite_and_nontrivial(self):
        for dtype in [torch.bfloat16, torch.float32]:
            t = torch.empty((10000,), dtype=dtype)
            benchmark.random_finite_payload(t)
            self.assertTrue(bool(torch.isfinite(t).all()))
            self.assertTrue(bool((t < 0).any()))
            self.assertTrue(bool((t > 0).any()))
            bits = t.view(torch.int16 if dtype == torch.bfloat16 else torch.int32)
            self.assertGreater(bits.unique().numel(), 1000)

    def test_signed_zero_is_checked_by_bits(self):
        for dtype in [torch.bfloat16, torch.float32]:
            a = torch.tensor([0.0], dtype=dtype)
            b = torch.tensor([-0.0], dtype=dtype)
            self.assertTrue(torch.equal(a, b))
            self.assertFalse(benchmark.bit_equal(a, b))


if __name__ == "__main__":
    unittest.main()
