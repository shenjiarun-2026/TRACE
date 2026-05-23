#!/usr/bin/env bash
set -euo pipefail

python -m trace.eval_lomo \
  --rankings "${RANKINGS:-examples/toy_rankings.json}" \
  --output results/lomo_plan.json
