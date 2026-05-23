import argparse

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from .descriptors import compute_dataset_descriptors, descriptor_features
from .io import load_dataset, load_weights, write_dataset, write_json


def score_rows_as_dataset(model, tokenizer, rows, args, weights):
    # TRACE weights are dataset-level. For filtering, we score each row as a tiny dataset.
    scored = []
    for idx, row in enumerate(rows):
        result = compute_dataset_descriptors(model, tokenizer, [row], args)
        feats = descriptor_features(result["metrics"], use_compliance=args.compute_compliance)
        score = float(sum(weights.get(k, 0.0) * feats.get(k, 0.0) for k in weights))
        scored.append({"index": idx, "score": score, "features": feats, "row": row})
    return scored


def parse_args():
    parser = argparse.ArgumentParser(description="Filter data with a learned TRACE weight vector.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scores_output", default=None)
    parser.add_argument("--top_n", type=int, default=1000)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--response_key", default="output")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_samples_for_style", type=int, default=1)
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--skip_toxicity", action="store_true")
    parser.add_argument("--detoxify_checkpoint", default=None)
    parser.add_argument("--compute_compliance", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device_map_auto", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = load_weights(args.weights)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto" if args.device_map_auto else None,
        trust_remote_code=args.trust_remote_code,
    )
    if not args.device_map_auto:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    rows = load_dataset(args.dataset)
    scored = score_rows_as_dataset(model, tokenizer, rows, args, weights)
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[: min(args.top_n, len(scored))]
    write_dataset(args.output, [x["row"] for x in top])
    if args.scores_output:
        write_json(args.scores_output, [{k: v for k, v in x.items() if k != "row"} for x in scored])
    print(f"[TRACE] Wrote {len(top)} rows to {args.output}; score range=({np.min([x['score'] for x in scored]):.4f}, {np.max([x['score'] for x in scored]):.4f})")


if __name__ == "__main__":
    main()
