#!/usr/bin/env bash
set -euo pipefail

# Example usage. Replace paths with public/local model IDs and data paths.
MODEL_PATHS="${MODEL_PATHS:-model_a,model_b,model_c}"
DATA_PATHS="${DATA_PATHS:-examples/toy_dataset.jsonl}"
RANKINGS="${RANKINGS:-examples/toy_rankings.json}"

python -m curate.train \
  --model "$MODEL_PATHS" \
  --data "$DATA_PATHS" \
  --rankings "$RANKINGS" \
  --output artifacts/metrics_by_model.json \
  --weights_output artifacts/curate_global_weights.json \
  --stability_output results/curate_stability_report.json \
  --pairwise_target combined \
  --safety_lambda 0.7 \
  --pairwise_features concat \
  --compute_compliance
