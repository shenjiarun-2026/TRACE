#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-path/to/unseen-model}"
DATASET="${DATASET:-examples/toy_dataset.jsonl}"
WEIGHTS="${WEIGHTS:-artifacts/trace_global_weights.json}"

python -m trace.filter \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --weights "$WEIGHTS" \
  --output outputs/trace_top.jsonl \
  --scores_output outputs/trace_scores.json \
  --top_n 1000 \
  --compute_compliance
