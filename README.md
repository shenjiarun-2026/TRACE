# CURATE

**CURATE**: CUrating safety tuning data via RObust and TrAnsferable dEscriptors.

CURATE is a gradient-free data curation pipeline for safety SFT. It learns an interpretable linear scorer over lightweight dataset-level descriptors from benchmark-induced pairwise dataset preferences pooled across multiple model families.

## Highlights

- Gradient-free: no target-model gradients are required.
- Interpretable: outputs one reusable global weight vector.
- Transferable: designed for leave-one-model-out and zero-shot transfer across model families.
- Lightweight descriptors: perplexity, toxicity/safety, self-BLEU diversity, information density, and compliance/refusal style.

## Installation

```bash
pip install -e .
# Optional extras:
pip install -e '.[toxicity,embeddings,data,dev]'
```

## Data format

CURATE expects JSON/JSONL records with at least:

```json
{"prompt": "...", "output": "..."}
```

The loader also normalizes common aliases such as `question -> prompt`, `instruction -> prompt`, and `response -> output`.

## Ranking format

Benchmark-induced rankings are stored as:

```json
{
  "dataset_a.json": {
    "models": [
      {"name": "model_a", "general_rank": 1, "safety_rank": 2, "num_datasets": 5},
      {"name": "model_b", "general_rank": 2, "safety_rank": 1, "num_datasets": 5}
    ]
  }
}
```

## Train global CURATE weights

```bash
MODEL_PATHS="model_a,model_b,model_c" \
DATA_PATHS="data/dataset_a.jsonl,data/dataset_b.jsonl" \
RANKINGS="configs/benchmark_rankings.json" \
bash scripts/train_curate_global.sh
```

This computes descriptors for each model and dataset, constructs model-specific pairwise dataset preferences, pools all pairwise rows, and trains a single global logistic regression scorer.

## Filter data for a new model

```bash
MODEL="path/to/unseen-model" \
DATASET="data/candidate_pool.jsonl" \
WEIGHTS="artifacts/curate_global_weights.json" \
bash scripts/filter_with_curate.sh
```

## Repository layout

```text
curate/         Core CURATE library
scripts/        Reproduction scripts
examples/       Toy data/ranking/weight examples
artifacts/      Released global CURATE weights
baselines/      Baseline scaffolds/wrappers
results/        Result summaries and generated reports
tests/          Minimal unit tests
```

## Notes for reproducibility

- Do not commit model checkpoints, HuggingFace caches, or DKL generation caches.
- If you use topic-level subdatasets, pass directories to `--data`; `expand_data_inputs` will track parent-to-subdataset mappings.

## Citation

If you use CURATE, please cite the corresponding paper.
