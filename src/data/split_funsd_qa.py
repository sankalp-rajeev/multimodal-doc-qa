import json
import random
from pathlib import Path
from typing import Dict, List, Any


RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1  # test will be 0.1


def load_qa_instances(path: Path) -> List[Dict[str, Any]]:
    instances: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            instances.append(json.loads(line))
    return instances


def group_by_doc(instances: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for ex in instances:
        doc_id = ex["doc_id"]
        by_doc.setdefault(doc_id, []).append(ex)
    return by_doc


def write_jsonl(instances: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in instances:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    data_path = Path("data/processed/funsd_qa.jsonl")
    out_train = Path("data/processed/funsd_qa_train.jsonl")
    out_val = Path("data/processed/funsd_qa_val.jsonl")
    out_test = Path("data/processed/funsd_qa_test.jsonl")

    print(f"Loading QA instances from: {data_path}")
    all_instances = load_qa_instances(data_path)
    print(f"Total QA instances: {len(all_instances)}")

    by_doc = group_by_doc(all_instances)
    doc_ids = list(by_doc.keys())
    print(f"Unique documents: {len(doc_ids)}")

    random.seed(RANDOM_SEED)
    random.shuffle(doc_ids)

    n_docs = len(doc_ids)
    n_train = int(n_docs * TRAIN_RATIO)
    n_val = int(n_docs * VAL_RATIO)
    # Remaining goes to test
    n_test = n_docs - n_train - n_val

    train_docs = doc_ids[:n_train]
    val_docs = doc_ids[n_train : n_train + n_val]
    test_docs = doc_ids[n_train + n_val :]

    print(f"Train docs: {len(train_docs)}, Val docs: {len(val_docs)}, Test docs: {len(test_docs)}")

    def collect(doc_list: List[str]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        doc_set = set(doc_list)
        for doc_id, insts in by_doc.items():
            if doc_id in doc_set:
                result.extend(insts)
        return result

    train_instances = collect(train_docs)
    val_instances = collect(val_docs)
    test_instances = collect(test_docs)

    print(f"Train QA instances: {len(train_instances)}")
    print(f"Val QA instances: {len(val_instances)}")
    print(f"Test QA instances: {len(test_instances)}")

    write_jsonl(train_instances, out_train)
    write_jsonl(val_instances, out_val)
    write_jsonl(test_instances, out_test)

    print(f"Saved train/val/test splits to:")
    print(f"  {out_train.resolve()}")
    print(f"  {out_val.resolve()}")
    print(f"  {out_test.resolve()}")


if __name__ == "__main__":
    main()
