#!/usr/bin/env bash
set -euo pipefail
# Materialize the traceable baseline only; do not train, tune, or evaluate here.
install -m 0644 /opt/kv-transfer/reference/candidate.py /workspace/candidate.py
