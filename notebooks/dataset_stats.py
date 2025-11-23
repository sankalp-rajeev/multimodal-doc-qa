import json
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    instances: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instances.append(json.loads(line))
    return instances


def compute_stats(instances: List[Dict[str, Any]], name: str):
    num = len(instances)
    if num == 0:
        print(f"{name}: no instances")
        return

    ctx_lengths = []
    ans_lengths = []

    for ex in instances:
        ctx_tokens = ex["context"].split()
        ans_tokens = ex["answer_text"].split()
        ctx_lengths.append(len(ctx_tokens))
        ans_lengths.append(len(ans_tokens))

    avg_ctx = sum(ctx_lengths) / num
    max_ctx = max(ctx_lengths)
    avg_ans = sum(ans_lengths) / num
    max_ans = max(ans_lengths)

    print(f"=== {name} ===")
    print(f"Instances: {num}")
    print(f"Context length: avg={avg_ctx:.1f}, max={max_ctx}")
    print(f"Answer length:  avg={avg_ans:.1f}, max={max_ans}")
    print()


def main():
    base = Path("data/processed")
    train_path = base / "funsd_qa_train.jsonl"
    val_path = base / "funsd_qa_val.jsonl"
    test_path = base / "funsd_qa_test.jsonl"

    train = load_jsonl(train_path)
    val = load_jsonl(val_path)
    test = load_jsonl(test_path)

    compute_stats(train, "TRAIN")
    compute_stats(val, "VAL")
    compute_stats(test, "TEST")


if __name__ == "__main__":
    main()
