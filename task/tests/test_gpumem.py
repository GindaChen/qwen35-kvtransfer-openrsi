"""Scratch-cap accounting: which processes count, the baseline, the cap, fail-closed."""
import sys
import unittest
from pathlib import Path
_root = Path(__file__).resolve().parents[1] / 'environment'
sys.path.insert(0, str(_root if _root.is_dir() else Path('/opt/kv-transfer')))
from isolation.gpumem import NOT_AVAILABLE, SCRATCH_LIMIT, Ledger, ScratchError, tree_usage

GiB = 2 ** 30


class Accounting(unittest.TestCase):
    def ledger(self):
        ledger = Ledger(range(2))
        ledger.exclude([[(10, 5 * GiB)], [(10, 3 * GiB), (11, GiB)]])  # controller + a pre-existing tenant
        ledger.set_context([[(10, 5 * GiB), (20, 500 << 20)], [(10, 3 * GiB), (11, GiB), (20, 400 << 20)]])
        return ledger

    def test_excluded_and_context_subtracted(self):
        ledger = self.ledger()
        ledger.observe([[(10, 9 * GiB), (20, (500 << 20) + GiB)], [(10, 3 * GiB), (11, 7 * GiB), (20, 400 << 20)]])
        self.assertEqual(ledger.take(), [GiB, 0])  # controller/tenant growth never counts

    def test_child_process_counts(self):
        ledger = self.ledger()
        ledger.observe([[(20, 500 << 20)], [(20, 400 << 20), (31, int(2.5 * GiB))]])
        peak = ledger.take()
        self.assertEqual(peak[1], int(2.5 * GiB))
        self.assertEqual(ledger.violations(peak), [(1, int(2.5 * GiB))])

    def test_peak_survives_free_and_take_resets(self):
        ledger = self.ledger()
        ledger.observe([[(20, (500 << 20) + 3 * GiB)], [(20, 400 << 20)]])
        ledger.observe([[(20, 500 << 20)], [(20, 400 << 20)]])  # freed before the end of the case
        self.assertEqual(ledger.take()[0], 3 * GiB)
        ledger.observe([[(20, 500 << 20)], [(20, 400 << 20)]])
        self.assertEqual(ledger.take(), [0, 0])

    def test_limit_is_inclusive(self):
        ledger = self.ledger()
        self.assertEqual(ledger.violations([SCRATCH_LIMIT, SCRATCH_LIMIT + 1]), [(1, SCRATCH_LIMIT + 1)])

    def test_unreportable_usage_fails_closed(self):
        with self.assertRaises(ScratchError):
            tree_usage([(20, NOT_AVAILABLE)], set())
        with self.assertRaises(ScratchError):
            tree_usage([(20, None)], set())
        self.assertEqual(tree_usage([(10, NOT_AVAILABLE)], {10}), 0)  # excluded PIDs are never read

    def test_observe_requires_baseline(self):
        with self.assertRaises(ScratchError):
            Ledger(range(1)).observe([[(20, 1)]])

    def test_duplicate_entries_summed(self):
        self.assertEqual(tree_usage([(20, GiB), (20, GiB)], set()), 2 * GiB)


if __name__ == '__main__':
    unittest.main()
