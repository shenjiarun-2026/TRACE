import argparse
import json
import math


def simple_ifd_score(row):
    # Placeholder for an IFD-style baseline. Replace with the exact definition used in experiments.
    prompt = row.get("prompt", "")
    output = row.get("output", "")
    return len(output.split()) / max(1, len(prompt.split()))


def main():
    parser = argparse.ArgumentParser(description="Minimal Superfiltering/IFD-style baseline scaffold.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top_n", type=int, default=1000)
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.dataset, encoding="utf-8") if line.strip()]
    scored = sorted([(simple_ifd_score(r), r) for r in rows], key=lambda x: x[0], reverse=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in scored[: min(args.top_n, len(scored))]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
