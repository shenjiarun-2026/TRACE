#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-path/to/unseen-model}"
DATASET="${DATASET:-examples/toy_dataset.jsonl}"
WEIGHTS="${WEIGHTS:-artifacts/curate_global_weights.json}"

python -m curate.filter \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --weights "$WEIGHTS" \
  --output outputs/curate_top.jsonl \
  --scores_output outputs/curate_scores.json \
  --top_n 1000 \
  --compute_compliance
