#!/usr/bin/env bash
set -euo pipefail
python /tests/test_plan_integrity.py
python /tests/test_layouts.py
python /tests/test_candidate_policy.py
exec python /tests/evaluate.py
