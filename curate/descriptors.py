import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


QUALITY_GROUP = ["compliance", "info_density", "ppl_inv"]
SAFETY_DIV_GROUP = ["tss95", "self_bleu_inv"]
DEFAULT_FEATURES = ["ppl_inv", "tss95", "self_bleu_inv", "info_density", "compliance"]


class SFTPerplexityDataset(Dataset):
    def __init__(self, rows, tokenizer, prompt_key="prompt", response_key="output", max_len=2048):
        self.rows = rows
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.max_len = max_len
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt_ids = self.tokenizer(row.get(self.prompt_key, ""), add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(row.get(self.response_key, ""), add_special_tokens=False)["input_ids"]
        if self.eos_id is not None:
            response_ids = response_ids + [self.eos_id]
        ids = prompt_ids + response_ids
        if len(ids) > self.max_len:
            overflow = len(ids) - self.max_len
            if overflow < len(prompt_ids):
                prompt_ids = prompt_ids[overflow:]
            else:
                prompt_ids = []
                response_ids = response_ids[-self.max_len :]
            ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids[:]
        return {"input_ids": ids, "labels": labels}


def collate_pad(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    ids, labels, attention = [], [], []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        ids.append(item["input_ids"] + [pad_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention.append([1] * len(item["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention),
    }


@torch.no_grad()
def compute_perplexity(model, dataloader, use_amp=True) -> Dict[str, object]:
    model.eval()
    device = next(model.parameters()).device
    nll, count = 0.0, 0
    per_example = []
    for batch in tqdm(dataloader, desc="Computing PPL"):
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention = batch["attention_mask"].to(device)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(ids, attention_mask=attention).logits[:, :-1, :]
            target = labels[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                torch.where(target == -100, torch.zeros_like(target), target).reshape(-1),
                reduction="none",
            )
            valid = (target != -100).reshape(-1)
            nll += loss[valid].sum().item()
            count += valid.long().sum().item()
        row_loss = loss.view(target.size(0), -1)
        row_mask = valid.view(target.size(0), -1)
        for i in range(row_loss.size(0)):
            denom = row_mask[i].sum().item()
            per_example.append(math.exp(row_loss[i][row_mask[i]].sum().item() / denom) if denom else float("nan"))
    return {"overall": math.exp(nll / max(1, count)) if count else float("inf"), "per_example": per_example}


def compute_self_bleu(responses: List[str], tokenizer, n=4) -> Optional[float]:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        return None
    toks = [tokenizer.tokenize(r) for r in responses]
    if len(toks) < 2:
        return 0.0
    smoothing = SmoothingFunction().method1
    scores = []
    for i in range(len(toks)):
        refs = [toks[j] for j in range(len(toks)) if j != i]
        scores.append(sentence_bleu(refs, toks[i], weights=[1 / n] * n, smoothing_function=smoothing))
    return float(np.mean(scores)) if scores else 0.0


def compute_info_density(responses: List[str], tokenizer) -> float:
    vals = []
    for response in responses:
        toks = tokenizer.tokenize(response)
        if toks:
            vals.append(len(set(toks)) / len(toks))
    return float(np.mean(vals)) if vals else 0.0


def compute_tss95(responses: List[str], checkpoint: Optional[str] = None) -> Optional[Dict[str, object]]:
    try:
        from detoxify import Detoxify
    except ImportError:
        return None
    model = Detoxify("original", checkpoint=checkpoint) if checkpoint else Detoxify("original")
    scores = model.predict(responses)["toxicity"]
    if isinstance(scores, torch.Tensor):
        scores = scores.tolist()
    return {"tss_95th": float(1.0 - np.percentile(scores, 95)), "toxicity_scores": [float(x) for x in scores]}


def compute_compliance(responses: List[str]) -> Dict[str, float]:
    refusal_phrases = ["i cannot", "i can't", "unable to", "apologize", "sorry", "as an ai", "cannot fulfill"]
    vals = [0.0 if any(p in r.lower() for p in refusal_phrases) else 1.0 for r in responses]
    return {"compliance_rate": float(np.mean(vals)) if vals else 0.0}


def compute_self_bleu_per_sample(responses: List[str], tokenizer, n=4) -> List[float]:
    """Leave-one-out self-BLEU of each response against the rest of the pool."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        return [0.0] * len(responses)
    toks = [tokenizer.tokenize(r) for r in responses]
    smoothing = SmoothingFunction().method1
    scores = []
    for i in tqdm(range(len(toks)), desc="Computing Self-BLEU"):
        refs = [toks[j] for j in range(len(toks)) if j != i]
        if refs:
            scores.append(sentence_bleu(refs, toks[i], weights=[1 / n] * n, smoothing_function=smoothing))
        else:
            scores.append(0.0)
    return scores


def compute_info_density_per_sample(responses: List[str], tokenizer) -> List[float]:
    densities = []
    for response in tqdm(responses, desc="Computing Info Density"):
        toks = tokenizer.tokenize(response)
        densities.append(len(set(toks)) / len(toks) if toks else 0.0)
    return densities


def compute_compliance_per_sample(responses: List[str]) -> List[float]:
    """1.0 if the response does not contain a refusal phrase, else 0.0."""
    refusal_phrases = ["i cannot", "i can't", "unable to", "apologize", "sorry", "as an ai", "cannot fulfill"]
    return [0.0 if any(p in r.lower() for p in refusal_phrases) else 1.0 for r in responses]


def compute_toxicity_per_sample(responses: List[str], checkpoint: Optional[str] = None) -> List[float]:
    """Per-sample (1 - toxicity) via Detoxify, so that higher is safer/better."""
    try:
        from detoxify import Detoxify
    except ImportError:
        print("[Warning] Detoxify not available, using 0.0 for toxicity scores")
        return [0.0] * len(responses)
    model = Detoxify("original", checkpoint=checkpoint) if checkpoint else Detoxify("original")
    scores = model.predict(responses)["toxicity"]
    if isinstance(scores, torch.Tensor):
        scores = scores.tolist()
    return [1.0 - float(s) for s in scores]


def compute_difficulty_balance(per_example_ppl: List[float], n_bins=10) -> Dict[str, Optional[float]]:
    vals = [x for x in per_example_ppl if np.isfinite(x)]
    if not vals:
        return {"db_js_divergence": None}
    hist, _ = np.histogram(vals, bins=n_bins, density=True)
    hist = hist / max(hist.sum(), 1e-12)
    target = np.ones(n_bins) / n_bins
    mid = 0.5 * (hist + target)
    js = 0.5 * (
        np.sum(hist * np.log((hist + 1e-10) / (mid + 1e-10)))
        + np.sum(target * np.log((target + 1e-10) / (mid + 1e-10)))
    )
    return {"db_js_divergence": float(js)}


def descriptor_features(metrics: dict, use_dkl=False, use_embedding=False, use_reward=False, use_compliance=True) -> Dict[str, float]:
    features = {}
    ppl = metrics.get("perplexity", {}).get("overall")
    if ppl is not None and np.isfinite(ppl):
        features["ppl_inv"] = 1.0 / (1.0 + float(ppl))
    if use_dkl:
        dkl = metrics.get("dkl", {}).get("mean")
        if dkl is not None and np.isfinite(dkl):
            features["dkl_inv"] = math.exp(-0.5 * float(dkl))
    tox = metrics.get("toxicity", {}).get("tss_95th")
    if tox is not None:
        features["tss95"] = float(tox)
    info = metrics.get("helpfulness", {}).get("avg_information_density")
    if info is not None:
        features["info_density"] = float(info)
    sb = metrics.get("diversity", {}).get("self_bleu")
    if sb is not None and np.isfinite(sb):
        features["self_bleu_inv"] = 1.0 - float(sb)
    if use_embedding:
        emb = metrics.get("diversity", {}).get("embedding_dispersion")
        if emb is not None and np.isfinite(emb):
            features["embedding_dispersion"] = float(emb)
    if use_reward:
        reward = metrics.get("reward_score", {}).get("mean")
        if reward is not None:
            features["reward_score"] = float(reward)
    if use_compliance:
        comp = metrics.get("compliance", {}).get("compliance_rate")
        if comp is not None:
            features["compliance"] = float(comp)
    return features


def compute_dataset_descriptors(model, tokenizer, rows, args) -> dict:
    responses = [r.get(args.response_key, "") for r in rows[: args.max_samples_for_style]]
    metrics = {}
    if not args.skip_ppl:
        ds = SFTPerplexityDataset(rows, tokenizer, args.prompt_key, args.response_key, args.max_length)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_pad(b, tokenizer.pad_token_id))
        metrics["perplexity"] = compute_perplexity(model, dl, use_amp=not args.no_amp)
    if not args.skip_toxicity:
        toxicity = compute_tss95(responses, checkpoint=args.detoxify_checkpoint)
        if toxicity is not None:
            metrics["toxicity"] = toxicity
    metrics["helpfulness"] = {"avg_information_density": compute_info_density(responses, tokenizer)}
    metrics["diversity"] = {"self_bleu": compute_self_bleu(responses, tokenizer)}
    if "perplexity" in metrics:
        metrics["difficulty_balance"] = compute_difficulty_balance(metrics["perplexity"]["per_example"])
    if args.compute_compliance:
        metrics["compliance"] = compute_compliance(responses)
    return {"dataset_size": len(rows), "metrics": metrics}
