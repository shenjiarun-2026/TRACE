import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

SUPPORTED_DATA_EXTS = {".jsonl", ".json"}


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return normalize_keys(rows)


def read_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}")
    return normalize_keys(data)


def normalize_keys(rows: List[dict]) -> List[dict]:
    for item in rows:
        if "question" in item and "prompt" not in item:
            item["prompt"] = item.pop("question")
        if "instruction" in item and "prompt" not in item:
            item["prompt"] = item.pop("instruction")
        if "response" in item and "output" not in item:
            item["output"] = item.pop("response")
    return rows


def load_dataset(path: str) -> List[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        return read_jsonl(path)
    if ext == ".json":
        return read_json(path)
    raise ValueError(f"Unsupported data format: {ext}")


def load_rankings(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_weights(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("weights", data)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_dataset(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    rows = list(rows)
    if ext == ".jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        write_json(path, rows)


def _is_data_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in SUPPORTED_DATA_EXTS


def _collect_files_in_dir(directory: str) -> List[str]:
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if _is_data_file(os.path.join(directory, name))
    ]


def expand_data_inputs(inputs: List[str]) -> Tuple[List[str], Dict[str, List[str]], Dict[str, str]]:
    """Expand input files/directories into sub-dataset files.

    Returns:
      dataset_files: all files to score;
      parent_to_subs: parent dataset name -> subdataset file paths;
      sub_to_parent: subdataset file path -> parent dataset name.
    """
    dataset_files: List[str] = []
    parent_to_subs: Dict[str, List[str]] = defaultdict(list)
    sub_to_parent: Dict[str, str] = {}

    for path in inputs:
        if _is_data_file(path):
            parent = os.path.splitext(os.path.basename(path))[0]
            dataset_files.append(path)
            parent_to_subs[parent].append(path)
            sub_to_parent[path] = parent
            continue

        if os.path.isdir(path):
            files_here = _collect_files_in_dir(path)
            if files_here:
                parent = os.path.basename(os.path.normpath(path))
                for fp in files_here:
                    dataset_files.append(fp)
                    parent_to_subs[parent].append(fp)
                    sub_to_parent[fp] = parent
            else:
                for sub in sorted(os.listdir(path)):
                    subdir = os.path.join(path, sub)
                    if not os.path.isdir(subdir):
                        continue
                    sub_files = _collect_files_in_dir(subdir)
                    if not sub_files:
                        continue
                    parent = os.path.basename(os.path.normpath(subdir))
                    for fp in sub_files:
                        dataset_files.append(fp)
                        parent_to_subs[parent].append(fp)
                        sub_to_parent[fp] = parent
            continue

        raise ValueError(f"Input path not found or unsupported: {path}")

    for key in list(parent_to_subs.keys()):
        parent_to_subs[key] = sorted(parent_to_subs[key])

    return dataset_files, dict(parent_to_subs), sub_to_parent
