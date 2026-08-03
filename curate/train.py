import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from .descriptors import compute_dataset_descriptors
from .io import expand_data_inputs, load_dataset, load_rankings, write_json
from .pairwise import aggregate_metric_weights, build_pooled_pairwise_rows


def summarize_weight_matrix(matrix, names, eps=1e-6):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return []
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    lo, hi = np.percentile(matrix, [2.5, 97.5], axis=0)
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "name": name,
            "mean": float(mean[i]),
            "std": float(std[i]),
            "ci95": [float(lo[i]), float(hi[i])],
            "selection_prob": float((np.abs(matrix[:, i]) > eps).mean()),
            "sign_stability": float(max((matrix[:, i] > eps).mean(), (matrix[:, i] < -eps).mean())),
        })
    return sorted(rows, key=lambda r: -abs(r["mean"]))


def stratified_bootstrap_indices(y, rng):
    y = np.asarray(y)
    idxs = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        idxs.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
    out = np.concatenate(idxs)
    rng.shuffle(out)
    return out


def train_global_logistic(
    X,
    y,
    feature_names,
    pairwise_mode="concat",
    sample_weight=None,
    groups=None,
    cv_group_mode="auto",
    c_grid=None,
    l1_ratio_grid=None,
    outer_splits=5,
    outer_repeats=10,
    bootstrap_runs=200,
    seed=1234,
):
    c_grid = c_grid or [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    l1_ratio_grid = l1_ratio_grid or [0.0, 0.5, 0.9, 1.0]
    base = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=10000,
        fit_intercept=False,
        random_state=seed,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", base)])
    grid = {"clf__C": c_grid, "clf__l1_ratio": l1_ratio_grid}

    use_group = groups is not None and len(groups) == len(y) and cv_group_mode in {"auto", "pair"} and len(np.unique(groups)) < len(groups)
    accs, f1s, fold_params = [], [], []
    if use_group and len(np.unique(groups)) >= 2:
        splitter = GroupKFold(n_splits=min(outer_splits, len(np.unique(groups)))).split(X, y, groups)
    else:
        min_class = np.unique(y, return_counts=True)[1].min()
        splitter = RepeatedStratifiedKFold(
            n_splits=min(outer_splits, min_class), n_repeats=outer_repeats, random_state=seed
        ).split(X, y)
        use_group = False

    for train_idx, test_idx in splitter:
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        wtr = sample_weight[train_idx] if sample_weight is not None else None
        wte = sample_weight[test_idx] if sample_weight is not None else None
        inner_k = min(3, np.unique(ytr, return_counts=True)[1].min())
        if use_group:
            gtr = groups[train_idx]
            inner_k = min(inner_k, len(np.unique(gtr)))
            if inner_k >= 2:
                gs = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=inner_k), scoring="accuracy", n_jobs=-1)
                gs.fit(Xtr, ytr, groups=gtr, clf__sample_weight=np.maximum(wtr, 1e-3) if wtr is not None else None)
                best = gs.best_estimator_
                fold_params.append(gs.best_params_)
            else:
                best = pipe.fit(Xtr, ytr, clf__sample_weight=np.maximum(wtr, 1e-3) if wtr is not None else None)
                fold_params.append({})
        else:
            if inner_k >= 2:
                gs = GridSearchCV(pipe, grid, cv=inner_k, scoring="accuracy", n_jobs=-1)
                gs.fit(Xtr, ytr, clf__sample_weight=np.maximum(wtr, 1e-3) if wtr is not None else None)
                best = gs.best_estimator_
                fold_params.append(gs.best_params_)
            else:
                best = pipe.fit(Xtr, ytr, clf__sample_weight=np.maximum(wtr, 1e-3) if wtr is not None else None)
                fold_params.append({})
        pred = best.predict(Xte)
        wf = np.maximum(wte, 1e-3) if wte is not None else None
        accs.append(accuracy_score(yte, pred, sample_weight=wf))
        f1s.append(f1_score(yte, pred, sample_weight=wf, zero_division=0))

    full_k = min(5, np.unique(y, return_counts=True)[1].min())
    if use_group:
        full_k = min(full_k, len(np.unique(groups)))
    if full_k >= 2:
        if use_group:
            gs_full = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=full_k), scoring="accuracy", n_jobs=-1)
            gs_full.fit(X, y, groups=groups, clf__sample_weight=np.maximum(sample_weight, 1e-3) if sample_weight is not None else None)
        else:
            gs_full = GridSearchCV(pipe, grid, cv=full_k, scoring="accuracy", n_jobs=-1)
            gs_full.fit(X, y, clf__sample_weight=np.maximum(sample_weight, 1e-3) if sample_weight is not None else None)
        final_model = gs_full.best_estimator_
        final_params = gs_full.best_params_
    else:
        final_model = pipe.fit(X, y, clf__sample_weight=np.maximum(sample_weight, 1e-3) if sample_weight is not None else None)
        final_params = {}

    coef = final_model.named_steps["clf"].coef_[0]
    weights = aggregate_metric_weights(feature_names, coef, pairwise_mode)

    rng = np.random.default_rng(seed)
    boot_metric_weights = []
    metric_keys = sorted(weights.keys())
    for _ in range(int(bootstrap_runs)):
        idx = stratified_bootstrap_indices(y, rng)
        Xb, yb = X[idx], y[idx]
        wb = sample_weight[idx] if sample_weight is not None else None
        bm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga", max_iter=10000, fit_intercept=False,
                random_state=seed, C=final_params.get("clf__C", 1.0), l1_ratio=final_params.get("clf__l1_ratio", 0.5)
            )),
        ])
        bm.fit(Xb, yb, clf__sample_weight=np.maximum(wb, 1e-3) if wb is not None else None)
        boot_metric_weights.append(aggregate_metric_weights(feature_names, bm.named_steps["clf"].coef_[0], pairwise_mode))

    perf = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)), "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
    stability = {
        "final_params": final_params,
        "cv_perf": perf,
        "fold_params": fold_params,
        "metric_weight_stability": summarize_weight_matrix(
            np.asarray([[mw.get(k, 0.0) for k in metric_keys] for mw in boot_metric_weights]), metric_keys
        ),
    }
    return final_model, weights, perf, stability


def model_name_from_path(path: str) -> str:
    return os.path.basename(os.path.normpath(path))


def compute_all_model_descriptors(args):
    model_paths = [p.strip() for p in args.model.split(",") if p.strip()]
    tokenizer_paths = [p.strip() for p in args.tokenizer.split(",") if p.strip()] if args.tokenizer else []
    inputs = [p.strip() for p in args.data.split(",") if p.strip()]
    dataset_files, parent_to_subs, _ = expand_data_inputs(inputs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for i, model_path in enumerate(model_paths):
        tok_path = tokenizer_paths[0] if len(tokenizer_paths) == 1 else (tokenizer_paths[i] if tokenizer_paths else model_path)
        model_name = model_name_from_path(model_path)
        print(f"[CURATE] Loading model {model_name}: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            device_map="auto" if args.device_map_auto else None,
            trust_remote_code=args.trust_remote_code,
        )
        if not args.device_map_auto:
            model.to(device)
        model.eval()
        model_results = {}
        for data_path in dataset_files:
            print(f"[CURATE] Descriptor computation: model={model_name} data={data_path}")
            model_results[data_path] = compute_dataset_descriptors(model, tokenizer, load_dataset(data_path), args)
        out[model_name] = model_results
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out, parent_to_subs


def parse_args():
    parser = argparse.ArgumentParser(description="Train CURATE global metric weights.")
    parser.add_argument("--model", required=True, help="Comma-separated model paths or HF IDs.")
    parser.add_argument("--tokenizer", default=None, help="Optional comma-separated tokenizer paths.")
    parser.add_argument("--data", required=True, help="Comma-separated dataset files/directories.")
    parser.add_argument("--rankings", required=True, help="Benchmark-induced ranking JSON.")
    parser.add_argument("--output", default="metrics_by_model.json")
    parser.add_argument("--weights_output", default="curate_global_weights.json")
    parser.add_argument("--stability_output", default="curate_stability_report.json")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--response_key", default="output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_samples_for_style", type=int, default=512)
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--skip_toxicity", action="store_true")
    parser.add_argument("--detoxify_checkpoint", default=None)
    parser.add_argument("--compute_compliance", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device_map_auto", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--pairwise_target", choices=["safety", "general", "combined"], default="combined")
    parser.add_argument("--safety_lambda", type=float, default=0.7)
    parser.add_argument("--pairwise_features", choices=["concat", "delta", "augmented"], default="concat")
    parser.add_argument("--cv_group_mode", choices=["auto", "none", "pair"], default="auto")
    parser.add_argument("--outer_repeats", type=int, default=10)
    parser.add_argument("--bootstrap_runs", type=int, default=200)
    parser.add_argument("--C_grid", default="0.01,0.03,0.1,0.3,1.0,3.0")
    parser.add_argument("--l1_ratio_grid", default="0.0,0.5,0.9,1.0")
    return parser.parse_args()


def main():
    args = parse_args()
    all_results_by_model, parent_to_subs = compute_all_model_descriptors(args)
    write_json(args.output, all_results_by_model)
    rankings = load_rankings(args.rankings)
    X, y, names, meta, pw, groups = build_pooled_pairwise_rows(
        all_results_by_model,
        rankings,
        parent_to_subs=parent_to_subs,
        pairwise_target=args.pairwise_target,
        safety_lambda=args.safety_lambda,
        pairwise_features=args.pairwise_features,
        feature_kwargs={"use_compliance": args.compute_compliance},
    )
    if len(X) == 0:
        raise RuntimeError("No pairwise rows were built. Check dataset keys, model names, and rankings.")
    print(f"[CURATE] Pooled training rows: {len(y)}; features: {len(names)}")
    c_grid = [float(x) for x in args.C_grid.split(",") if x.strip()]
    l1_grid = [float(x) for x in args.l1_ratio_grid.split(",") if x.strip()]
    _, weights, perf, stability = train_global_logistic(
        X, y, names, pairwise_mode=args.pairwise_features, sample_weight=pw, groups=groups,
        cv_group_mode=args.cv_group_mode, c_grid=c_grid, l1_ratio_grid=l1_grid,
        outer_repeats=args.outer_repeats, bootstrap_runs=args.bootstrap_runs,
    )
    write_json(args.stability_output, stability)
    write_json(args.weights_output, {
        "method": "CURATE",
        "weights": weights,
        "acc": perf["acc_mean"],
        "f1": perf["f1_mean"],
        "n_rows": int(len(y)),
        "n_models": int(len(all_results_by_model)),
        "pairwise_target": args.pairwise_target,
        "safety_lambda": args.safety_lambda,
        "pairwise_features": args.pairwise_features,
    })
    print("[CURATE] Learned global weights:")
    for key, value in sorted(weights.items(), key=lambda x: -abs(x[1])):
        print(f"  {key:24s} {value:+.4f}")
    print(f"[CURATE] CV acc={perf['acc_mean']:.3f}±{perf['acc_std']:.3f}; f1={perf['f1_mean']:.3f}±{perf['f1_std']:.3f}")


if __name__ == "__main__":
    main()
