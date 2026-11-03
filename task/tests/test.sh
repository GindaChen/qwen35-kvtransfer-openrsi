#!/usr/bin/env bash
set -euo pipefail
python /tests/test_plan_integrity.py
exec python /tests/evaluate.py
