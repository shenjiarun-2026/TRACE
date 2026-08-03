import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .descriptors import (
    SFTPerplexityDataset,
    collate_pad,
    compute_compliance_per_sample,
    compute_info_density_per_sample,
    compute_perplexity,
    compute_self_bleu_per_sample,
    compute_toxicity_per_sample,
)
from .io import load_dataset, load_weights, write_dataset, write_json


def compute_sample_scores(ppl_scores, self_bleu_scores, info_density_scores, compliance_scores, tss95_scores, weights):
    """Weighted per-sample score, matching the feature transforms used at training time."""
    scores = []
    n = len(ppl_scores)
    for i in range(n):
        ppl_inv = 1.0 / (1.0 + ppl_scores[i]) if np.isfinite(ppl_scores[i]) else 0.0
        feats = {
            "ppl_inv": ppl_inv,
            "self_bleu_inv": 1.0 - self_bleu_scores[i],
            "info_density": info_density_scores[i],
            "compliance": compliance_scores[i],
            "tss95": tss95_scores[i],
        }
        scores.append(float(sum(weights.get(k, 0.0) * v for k, v in feats.items())))
    return scores


def parse_args():
    parser = argparse.ArgumentParser(description="Filter data with a learned CURATE weight vector.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scores_output", default=None)
    parser.add_argument("--top_n", type=int, default=1000)
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--response_key", default="output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--skip_toxicity", action="store_true")
    parser.add_argument("--detoxify_checkpoint", default=None)
    parser.add_argument("--compute_compliance", action="store_true",
                        help="Accepted for compatibility; compliance is always computed in weighted mode.")
    parser.add_argument(
        "--selection_mode",
        choices=["weighted", "ppl_only", "tss95_only", "random"],
        default="weighted",
        help="Ablation mode: weighted (full CURATE scorer), ppl_only (lowest PPL), "
             "tss95_only (highest safety), random (uniform sample).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --selection_mode=random")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device_map_auto", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = load_weights(args.weights)
    print("[CURATE] Using weights:")
    for key, value in sorted(weights.items(), key=lambda x: -abs(x[1])):
        print(f"  {key:24s} {value:+.4f}")

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
    n = len(rows)
    print(f"[CURATE] Loaded {n} rows from {args.dataset}")
    responses = [r.get(args.response_key, "") for r in rows]

    ppl_scores = [float("nan")] * n
    tss95_scores = [0.0] * n
    self_bleu_scores = [0.0] * n
    info_density_scores = [0.0] * n
    compliance_scores = [0.0] * n
    mode = args.selection_mode

    if mode == "random":
        print(f"[CURATE] [Mode=random] Sampling {args.top_n} with seed={args.seed}")
        rng = random.Random(args.seed)
        idx = list(range(n))
        rng.shuffle(idx)
        selected_idx = set(idx[: min(args.top_n, n)])
        scored = [(rows[i], 0.0, i) for i in sorted(selected_idx)]

    else:
        if mode in ("weighted", "ppl_only") and not args.skip_ppl:
            print("[CURATE] Computing per-sample perplexity...")
            ds = SFTPerplexityDataset(rows, tokenizer, args.prompt_key, args.response_key, args.max_length)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_pad(b, tokenizer.pad_token_id))
            ppl_scores = compute_perplexity(model, dl, use_amp=not args.no_amp)["per_example"]

        if mode in ("weighted", "tss95_only") and not args.skip_toxicity:
            print("[CURATE] Computing per-sample toxicity (Detoxify; returns 1 - toxicity)...")
            tss95_scores = compute_toxicity_per_sample(responses, checkpoint=args.detoxify_checkpoint)

        if mode == "weighted":
            print("[CURATE] Computing per-sample self-BLEU (leave-one-out over pool)...")
            self_bleu_scores = compute_self_bleu_per_sample(responses, tokenizer)
            print("[CURATE] Computing per-sample information density...")
            info_density_scores = compute_info_density_per_sample(responses, tokenizer)
            compliance_scores = compute_compliance_per_sample(responses)
            final_scores = compute_sample_scores(
                ppl_scores, self_bleu_scores, info_density_scores, compliance_scores, tss95_scores, weights
            )
        elif mode == "ppl_only":
            # higher is better: -ppl
            final_scores = [(-p if np.isfinite(p) else -float("inf")) for p in ppl_scores]
        else:  # tss95_only
            final_scores = [float(s) for s in tss95_scores]

        order = sorted(range(n), key=lambda i: final_scores[i], reverse=True)
        top_n = min(args.top_n, n)
        scored = [(rows[i], final_scores[i], i) for i in order[:top_n]]

    top_rows = [x[0] for x in scored]
    write_dataset(args.output, top_rows)
    print(f"[CURATE] Wrote {len(top_rows)} rows to {args.output}")

    if args.scores_output and mode != "random":
        selected_set = {x[2] for x in scored}
        dump = [{
            "index": i,
            "final_score": final_scores[i],
            "ppl": ppl_scores[i],
            "self_bleu": self_bleu_scores[i],
            "info_density": info_density_scores[i],
            "compliance": compliance_scores[i],
            "tss95": tss95_scores[i],
            "selected": i in selected_set,
        } for i in range(n)]
        write_json(args.scores_output, dump)
        print(f"[CURATE] Wrote score dump to {args.scores_output}")

    print("[CURATE] FILTERING COMPLETE")
    print(f"Selection mode: {mode}")
    print(f"Original dataset size: {n}")
    print(f"Filtered dataset size: {len(top_rows)}")


if __name__ == "__main__":
    main()
