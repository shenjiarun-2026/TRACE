#!/usr/bin/env python3
"""Build a 200-example validation set for safety alignment and capability preservation.

The output is JSONL. Each line has:
{"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


SAMPLES_PER_BENCHMARK = 25
DEFAULT_SEED = 42
DEFAULT_OUTPUT = "data/safety_capability_validation_200.jsonl"

REFUSAL_RESPONSE = (
    "I'm sorry, but I cannot assist with that request. This involves potentially harmful content "
    "which could cause serious harm. I'd be happy to help you with something constructive instead."
)

ADV_BENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
)
DNA_URL = "https://raw.githubusercontent.com/Libr-AI/do-not-answer/main/datasets/data_en.csv"
XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"

MMLU_STEM_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
    "medical_genetics",
    "nutrition",
    "virology",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample 8 benchmarks x 25 examples into a safety/capability validation JSONL."
    )
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT), help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--samples-per-benchmark",
        type=int,
        default=SAMPLES_PER_BENCHMARK,
        help="Number of examples to sample from each benchmark.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Also write source benchmark metadata beside messages.",
    )
    return parser.parse_args()


def require_datasets() -> None:
    if load_dataset is None:
        raise SystemExit(
            "This script requires the HuggingFace datasets package for HF benchmarks.\n"
            "Install it with: pip install datasets"
        )


def load_hf_dataset(*args: Any, **kwargs: Any) -> Any:
    require_datasets()
    return load_dataset(*args, **kwargs)


def as_records(dataset: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(dataset, dict):
        raise TypeError("Expected a split Dataset, got DatasetDict-like object.")
    return [dict(row) for row in dataset]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(normalize_text(item) for item in value if normalize_text(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def require_count(name: str, rows: list[Any], count: int) -> None:
    if len(rows) < count:
        raise RuntimeError(f"{name} only has {len(rows)} usable examples; need {count}.")


def sample_rows(name: str, rows: list[Any], count: int, rng: random.Random) -> list[Any]:
    require_count(name, rows, count)
    return rng.sample(rows, count)


def make_message(user: str, assistant: str, source: str | None = None, include_metadata: bool = False) -> dict[str, Any]:
    item: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }
    if include_metadata and source is not None:
        item["source"] = source
    return item


def first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def load_advbench(count: int, rng: random.Random, include_metadata: bool) -> list[dict[str, Any]]:
    df = pd.read_csv(ADV_BENCH_URL)
    column = "goal" if "goal" in df.columns else df.columns[0]
    prompts = df[column].dropna().astype(str).str.strip()
    prompts = [prompt for prompt in prompts if prompt]
    return [
        make_message(prompt, REFUSAL_RESPONSE, "AdvBench", include_metadata)
        for prompt in sample_rows("AdvBench", prompts, count, rng)
    ]


def load_do_not_answer(count: int, rng: random.Random, include_metadata: bool) -> list[dict[str, Any]]:
    try:
        records = load_first_available_dataset(
            "Do-Not-Answer",
            [("LibrAI/do-not-answer", "train"), ("LibrAI/do-not-answer", "test")],
        )
        prompts = [
            normalize_text(first_present(row, ["question", "prompt", "instruction", "text"]))
            for row in records
        ]
        prompts = [prompt for prompt in prompts if prompt]
    except Exception:
        df = pd.read_csv(DNA_URL)
        column = "question" if "question" in df.columns else df.columns[0]
        prompts = df[column].dropna().astype(str).str.strip()
        prompts = [prompt for prompt in prompts if prompt]

    return [
        make_message(prompt, REFUSAL_RESPONSE, "Do-Not-Answer", include_metadata)
        for prompt in sample_rows("Do-Not-Answer", prompts, count, rng)
    ]


def load_xstest(count: int, rng: random.Random, include_metadata: bool) -> list[dict[str, Any]]:
    try:
        records = as_records(load_hf_dataset("Bertphil/XSTest", split="train"))
        prompts = extract_xstest_prompts(records)
    except Exception:
        df = pd.read_csv(XSTEST_URL)
        prompts = extract_xstest_prompts(df.to_dict("records"))

    return [
        make_message(prompt, REFUSAL_RESPONSE, "XSTest", include_metadata)
        for prompt in sample_rows("XSTest", prompts, count, rng)
    ]


def extract_xstest_prompts(records: list[dict[str, Any]]) -> list[str]:
    unsafe_records = []
    for row in records:
        label = normalize_text(first_present(row, ["label", "safety_label", "target"])).lower()
        row_type = normalize_text(first_present(row, ["type", "category"])).lower()
        if "unsafe" in label or "contrast" in row_type or "harm" in label:
            unsafe_records.append(row)

    selected = unsafe_records or records
    prompts = []
    for row in selected:
        prompt = normalize_text(first_present(row, ["prompt", "question", "instruction", "text"]))
        if prompt:
            prompts.append(prompt)
    return prompts


def load_safetybench(count: int, rng: random.Random, include_metadata: bool) -> list[dict[str, Any]]:
    records = []
    try:
        ds = load_hf_dataset("thu-coai/SafetyBench", "dev", split="en")
    except Exception:
        ds = load_hf_dataset("thu-coai/SafetyBench", split="en")

    for row in as_records(ds):
        list_values = [value for value in row.values() if isinstance(value, list)]
        if list_values:
            for value in list_values:
                records.extend(item for item in value if isinstance(item, dict))
        else:
            records.append(row)

    prompts = []
    labels = ["A", "B", "C", "D", "E"]
    for row in records:
        question = normalize_text(first_present(row, ["question", "prompt", "instruction"]))
        options = first_present(row, ["options", "choices"])
        if isinstance(options, str):
            options = [options]
        if question and isinstance(options, list) and options:
            option_text = "\n".join(f"{labels[i]}. {normalize_text(option)}" for i, option in enumerate(options[: len(labels)]))
            prompts.append(f"Question: {question}\n{option_text}\nPlease output the safest answer choice:")
        elif question:
            prompts.append(question)

    return [
        make_message(prompt, REFUSAL_RESPONSE, "SafetyBench", include_metadata)
        for prompt in sample_rows("SafetyBench", prompts, count, rng)
    ]


def load_aime2024(count: int, rng: random.Random, include_metadata: bool) -> list[dict[str, Any]]:
    candidates = [
        ("RyanYr/AIME2024", "test"),
        ("Maxwell-Jia/AIME_2024", "train"),
        ("AI-MO/aimo-validation-aime", "train"),
    ]
    records = load_first_available_dataset("AIME2024", candidates)
    examples = []
    for row in records:
        problem = normalize_text(first_present(row, ["problem", "Problem", "question", "prompt"]))
        solution = normalize_text(first_present(row, ["solution", "Solution", "response", "answer"]))
        answer = normalize_text(first_present(row, ["answer", "Answer", "final_answer", "ground_truth"]))
        if not problem or not solution:
            continue
        if answer and answer not in solution:
            solution = f"{solution}\n\nFinal answer: {answer}"
        examples.append((problem, solution))

    return [
        make_message(problem, solution, "AIME2024", include_metadata)
        for problem, solution in sample_rows("AIME2024", examples, count, rng)
    ]


def load_first_available_dataset(name: str, candidates: list[tuple[str, str]]) -> list[dict[str, Any]]:
    errors = []
    for dataset_name, split in candidates:
        try:
            return as_records(load_hf_dataset(dataset_name, split=split))
        except Exception as exc:
            errors.append(f"{dataset_name}/{split}: {exc}")
    raise RuntimeError(f"Could not load {name}. Tried:\n" + "\n".join(errors))


def load_mmlu_stem(count: int, rng: random.Random, include_metadata: bool) -> list[dict[str, Any]]:
    records = []
    for subject in MMLU_STEM_SUBJECTS:
        try:
            records.extend(as_records(load_hf_dataset("cais/mmlu", subject, split="validation")))
        except Exception as exc:
            print(f"[Warn] Skipping MMLU subject {subject}: {exc}")

    examples = []
    for row in records:
        question = normalize_text(first_present(row, ["question", "prompt"]))
        choices = first_present(row, ["choices", "options"])
        if not isinstance(choices, list):
            choices = [first_present(row, [key]) for key in ["A", "B", "C", "D"]]
        choices = [normalize_text(choice) for choice in choices if normalize_text(choice)]
        answer = first_present(row, ["answer", "target", "label"])
        if isinstance(answer, int):
            correct_option = "ABCD"[answer] if 0 <= answer < 4 else str(answer)
        else:
            answer_text = normalize_text(answer).strip()
            correct_option = answer_text if answer_text in {"A", "B", "C", "D"} else answer_text[:1].upper()
        if question and len(choices) >= 4 and correct_option in {"A", "B", "C", "D"}:
            user = f"Question: {question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
            assistant = f"The answer is {correct_option}."
            examples.append((user, assistant))

    return [
        make_message(user, assistant, "MMLU-STEM", include_metadata)
        for user, assistant in sample_rows("MMLU-STEM", examples, count, rng)
    ]


def load_humaneval(count: int, rng: random.Random, include_metadata: bool) -> list[dict[str, Any]]:
    records = load_first_available_dataset("HumanEval", [("openai/openai_humaneval", "test"), ("openai_humaneval", "test")])
    examples = []
    for row in records:
        prompt = normalize_text(first_present(row, ["prompt", "docstring", "question"]))
        solution = normalize_text(first_present(row, ["canonical_solution", "solution", "code"]))
        if prompt and solution:
            examples.append((prompt, solution))

    return [
        make_message(prompt, solution, "HumanEval", include_metadata)
        for prompt, solution in sample_rows("HumanEval", examples, count, rng)
    ]


def load_mbpp(count: int, rng: random.Random, include_metadata: bool) -> list[dict[str, Any]]:
    candidates = [
        ("google-research-datasets/mbpp", "test"),
        ("google-research-datasets/mbpp", "validation"),
        ("mbpp", "test"),
        ("mbpp", "train"),
    ]
    records = load_first_available_dataset("MBPP", candidates)
    examples = []
    for row in records:
        description = normalize_text(first_present(row, ["text", "description", "prompt", "question"]))
        solution = normalize_text(first_present(row, ["code", "canonical_solution", "solution"]))
        if description and solution:
            examples.append((description, solution))

    return [
        make_message(description, solution, "MBPP", include_metadata)
        for description, solution in sample_rows("MBPP", examples, count, rng)
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_output(rows: list[dict[str, Any]], expected_count: int) -> None:
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} rows, got {len(rows)}.")
    for idx, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise RuntimeError(f"Row {idx} has invalid messages: {messages!r}")
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            raise RuntimeError(f"Row {idx} has invalid roles: {messages!r}")
        if not messages[0].get("content") or not messages[1].get("content"):
            raise RuntimeError(f"Row {idx} has empty content.")


def main() -> None:
    args = parse_args()
    require_datasets()
    rng = random.Random(args.seed)
    count = args.samples_per_benchmark

    loaders = [
        ("AdvBench", load_advbench),
        ("Do-Not-Answer", load_do_not_answer),
        ("XSTest", load_xstest),
        ("SafetyBench", load_safetybench),
        ("AIME2024", load_aime2024),
        ("MMLU-STEM", load_mmlu_stem),
        ("HumanEval", load_humaneval),
        ("MBPP", load_mbpp),
    ]

    all_rows = []
    for name, loader in loaders:
        print(f"[Load] {name}...")
        rows = loader(count, rng, args.include_metadata)
        all_rows.extend(rows)
        print(f"[OK] {name}: {len(rows)}")

    expected_count = count * len(loaders)
    validate_output(all_rows, expected_count)
    write_jsonl(args.output, all_rows)
    print(f"[Done] Wrote {len(all_rows)} examples to {args.output}")


if __name__ == "__main__":
    main()
