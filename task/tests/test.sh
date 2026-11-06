#!/usr/bin/env bash
set -euo pipefail
python /tests/test_plan_integrity.py
python /tests/test_layouts.py
python /tests/test_candidate_policy.py
python /tests/test_checker.py
python /tests/test_results.py
python /tests/test_wire.py
python /tests/test_guard.py
# Adversarial preflight on this site: hostile candidates must be contained by
# the actual rsi-isolate + worker guard boundary before anything is scored.
PYTHONPATH=/opt/kv-transfer python /tests/validate_isolation.py --output /logs/verifier/isolation-preflight
exec python /tests/evaluate.py
