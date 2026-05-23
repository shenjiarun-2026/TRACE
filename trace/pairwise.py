import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .descriptors import descriptor_features


def norm_key(value: str) -> str:
    return os.path.splitext(os.path.basename(value))[0].strip().lower()


def rank_to_score(rank, n):
    return 1.0 - (float(rank) - 1.0) / max(1, int(n) - 1)


def combined_rank_score(general_rank, safety_rank, n, safety_lambda):
    return safety_lambda * rank_to_score(safety_rank, n) + (1.0 - safety_lambda) * rank_to_score(general_rank, n)


def infer_models_from_rankings(rankings: dict) -> List[str]:
    names = set()
    for value in rankings.values():
        for model_info in value.get("models", []):
            if "name" in model_info:
                names.add(model_info["name"])
    return sorted(names)


def make_pair_features(f1: dict, f2: dict, feature_names: List[str], mode="concat") -> Tuple[List[float], List[str]]:
    vals, names = [], []
    if mode == "concat":
        for key in feature_names:
            vals.extend([f1[key], f2[key]])
            names.extend([f"d1__{key}", f"d2__{key}"])
    elif mode == "delta":
        for key in feature_names:
            vals.append(f1[key] - f2[key])
            names.append(f"delta__{key}")
    elif mode == "augmented":
        for key in feature_names:
            d1, d2 = f1[key], f2[key]
            vals.extend([d1, d2, d1 - d2, abs(d1 - d2)])
            names.extend([f"d1__{key}", f"d2__{key}", f"diff__{key}", f"adiff__{key}"])
    else:
        raise ValueError(f"Unsupported pair feature mode: {mode}")
    return vals, names


def aggregate_metric_weights(feature_names: List[str], coefs, mode="concat") -> Dict[str, float]:
    contribs = defaultdict(list)
    for idx, name in enumerate(feature_names):
        coef = float(coefs[idx])
        if mode == "concat":
            if name.startswith("d1__"):
                contribs[name.split("__", 1)[1]].append(coef)
            elif name.startswith("d2__"):
                contribs[name.split("__", 1)[1]].append(-coef)
        elif mode == "delta" and name.startswith("delta__"):
            contribs[name.split("__", 1)[1]].append(coef)
        elif mode == "augmented":
            if name.startswith("d1__"):
                contribs[name.split("__", 1)[1]].append(coef)
            elif name.startswith("d2__"):
                contribs[name.split("__", 1)[1]].append(-coef)
            elif name.startswith("diff__"):
                contribs[name.split("__", 1)[1]].append(coef)
    raw = {k: float(np.mean(v)) for k, v in contribs.items() if v}
    denom = sum(abs(v) for v in raw.values()) or 1.0
    return {k: v / denom for k, v in raw.items()}


def build_feature_dict(model_results: dict, feature_names: Optional[List[str]] = None, **feature_kwargs):
    fds = {path: descriptor_features(result["metrics"], **feature_kwargs) for path, result in model_results.items()}
    if feature_names is None:
        feature_names = sorted(set().union(*[set(v.keys()) for v in fds.values()])) if fds else []
    return fds, feature_names


def _model_rankings_by_parent(rankings: dict, model_name: str, parents: List[str]) -> dict:
    parent_norm = {norm_key(p): p for p in parents}
    out = {}
    for dataset_key, value in rankings.items():
        key = norm_key(dataset_key)
        if key not in parent_norm:
            continue
        parent = parent_norm[key]
        matches = [m for m in value.get("models", []) if m.get("name") == model_name]
        if matches:
            out[parent] = matches[0]
    return out


def build_pairwise_rows_for_model(
    model_name: str,
    model_results: dict,
    rankings: dict,
    parent_to_subs: Optional[Dict[str, List[str]]] = None,
    feature_names: Optional[List[str]] = None,
    pairwise_target="combined",
    safety_lambda=0.7,
    pairwise_features="concat",
    feature_kwargs=None,
):
    feature_kwargs = feature_kwargs or {}
    fds, feature_names = build_feature_dict(model_results, feature_names=feature_names, **feature_kwargs)
    if not fds or not feature_names:
        return np.array([]), np.array([]), [], [], np.array([]), np.array([])

    if parent_to_subs:
        parent_names = list(parent_to_subs.keys())
    else:
        parent_names = list(model_results.keys())
        parent_to_subs = {p: [p] for p in parent_names}

    rank_by_parent = _model_rankings_by_parent(rankings, model_name, parent_names)
    if len(rank_by_parent) < 2:
        return np.array([]), np.array([]), [], [], np.array([]), np.array([])

    X, y, meta, weights, groups = [], [], [], [], []
    out_names = None
    parents = sorted(rank_by_parent.keys())
    for i in range(len(parents)):
        for j in range(i + 1, len(parents)):
            p1, p2 = parents[i], parents[j]
            r1, r2 = rank_by_parent[p1], rank_by_parent[p2]
            n = max(int(r1.get("num_datasets", len(parents))), int(r2.get("num_datasets", len(parents))))
            if pairwise_target == "safety":
                s1, s2 = rank_to_score(r1["safety_rank"], n), rank_to_score(r2["safety_rank"], n)
            elif pairwise_target == "general":
                s1, s2 = rank_to_score(r1["general_rank"], n), rank_to_score(r2["general_rank"], n)
            else:
                s1 = combined_rank_score(r1["general_rank"], r1["safety_rank"], n, safety_lambda)
                s2 = combined_rank_score(r2["general_rank"], r2["safety_rank"], n, safety_lambda)
            label = 1 if s1 > s2 else 0
            margin = abs(s1 - s2)
            group = f"{model_name}|{norm_key(p1)}|{norm_key(p2)}"

            for d1 in parent_to_subs.get(p1, []):
                for d2 in parent_to_subs.get(p2, []):
                    if d1 not in fds or d2 not in fds:
                        continue
                    f1, f2 = fds[d1], fds[d2]
                    if not all(np.isfinite(f1.get(k, np.nan)) and np.isfinite(f2.get(k, np.nan)) for k in feature_names):
                        continue
                    vals, names = make_pair_features(f1, f2, feature_names, pairwise_features)
                    out_names = out_names or names
                    X.append(vals)
                    y.append(label)
                    weights.append(margin)
                    groups.append(group)
                    meta.append({"model": model_name, "d1": d1, "d2": d2, "label": label})
                    vals_sym, _ = make_pair_features(f2, f1, feature_names, pairwise_features)
                    X.append(vals_sym)
                    y.append(1 - label)
                    weights.append(margin)
                    groups.append(group)
                    meta.append({"model": model_name, "d1": d2, "d2": d1, "label": 1 - label})
    return np.asarray(X), np.asarray(y), out_names or [], meta, np.asarray(weights), np.asarray(groups)


def build_pooled_pairwise_rows(all_results_by_model: dict, rankings: dict, parent_to_subs=None, **kwargs):
    Xs, ys, ws, gs = [], [], [], []
    names_ref = None
    meta_all = []
    for model_name, model_results in all_results_by_model.items():
        X, y, names, meta, w, g = build_pairwise_rows_for_model(
            model_name, model_results, rankings, parent_to_subs=parent_to_subs, feature_names=None, **kwargs
        )
        if len(X) == 0:
            print(f"[TRACE] Warning: no pairwise rows for {model_name}; skipped.")
            continue
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            raise ValueError(f"Feature mismatch for {model_name}: expected {names_ref}, got {names}")
        Xs.append(X)
        ys.append(y)
        ws.append(w)
        gs.append(g)
        meta_all.extend(meta)
        print(f"[TRACE] {model_name}: {len(y)} pooled pairwise rows")
    if not Xs:
        return np.array([]), np.array([]), [], [], np.array([]), np.array([])
    return np.vstack(Xs), np.concatenate(ys), names_ref or [], meta_all, np.concatenate(ws), np.concatenate(gs)
