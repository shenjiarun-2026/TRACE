import argparse

from .io import load_rankings, write_json
from .pairwise import infer_models_from_rankings


def main():
    parser = argparse.ArgumentParser(description="Skeleton helper for TRACE leave-one-model-out evaluation.")
    parser.add_argument("--rankings", required=True)
    parser.add_argument("--output", default="lomo_plan.json")
    args = parser.parse_args()
    rankings = load_rankings(args.rankings)
    models = infer_models_from_rankings(rankings)
    report = {
        "note": "Run trace.train once per holdout model by filtering rankings externally or extend this skeleton.",
        "models_in_rankings": models,
        "folds": [{"holdout_model": m, "train_models": [x for x in models if x != m]} for m in models],
    }
    write_json(args.output, report)
    print(f"[TRACE] Wrote LOMO plan to {args.output}")


if __name__ == "__main__":
    main()
