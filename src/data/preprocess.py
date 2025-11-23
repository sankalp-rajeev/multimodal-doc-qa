import json
from pathlib import Path
from typing import List, Dict, Any, Tuple


def read_funsd_split(split_dir: Path) -> List[Dict[str, Any]]:
    """
    Parse FUNSD annotations in one split directory
    (e.g., data/raw/funsd/training_data or testing_data).

    Returns a list of dicts:
    {
        "doc_id": str,
        "context": str,           # flattened document text
        "qa_pairs": [
            {
                "question_text": str,
                "answer_text": str,
            },
            ...
        ]
    }
    """
    ann_dir = split_dir / "annotations"

    docs: List[Dict[str, Any]] = []

    for ann_file in sorted(ann_dir.glob("*.json")):
        with ann_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        form_fields = data.get("form", [])

        # 1) Build flattened document text from all words (simulates OCR text)
        all_tokens: List[str] = []
        for field in form_fields:
            for w in field.get("words", []):
                t = (w.get("text") or "").strip()
                if t:
                    all_tokens.append(t)
        context = " ".join(all_tokens)

        # 2) Build map id -> field, and prepare question->answers linking
        id_to_field: Dict[Any, Dict[str, Any]] = {}
        for field in form_fields:
            field_id = field.get("id", None)
            if field_id is None:
                field_id = len(id_to_field)
            id_to_field[field_id] = field

        question_to_answers: Dict[Any, List[Any]] = {}
        for field in form_fields:
            field_id = field.get("id", None)
            if field_id is None:
                continue
            for link in field.get("linking", []):
                if len(link) != 2:
                    continue
                a, b = link
                # We don't assume ordering; we'll handle later
                if field_id == a:
                    question_to_answers.setdefault(a, []).append(b)
                elif field_id == b:
                    question_to_answers.setdefault(b, []).append(a)

        # 3) Extract QA pairs based on label="question" and linked fields
        qa_pairs: List[Dict[str, str]] = []
        for field in form_fields:
            label = (field.get("label") or "").lower()
            if label != "question":
                continue

            q_id = field.get("id", None)
            if q_id is None:
                continue

            question_text = (field.get("text") or "").strip()
            if not question_text:
                continue

            candidate_answer_ids = question_to_answers.get(q_id, [])
            if not candidate_answer_ids:
                continue

            for a_id in candidate_answer_ids:
                answer_field = id_to_field.get(a_id)
                if not answer_field:
                    continue
                answer_text = (answer_field.get("text") or "").strip()
                if not answer_text:
                    continue

                qa_pairs.append(
                    {
                        "question_text": question_text,
                        "answer_text": answer_text,
                    }
                )

        doc_id = ann_file.stem
        docs.append(
            {
                "doc_id": doc_id,
                "context": context,
                "qa_pairs": qa_pairs,
            }
        )

    return docs

def normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """
    Lowercase text and collapse all whitespace to single spaces,
    while keeping a mapping from normalized char index -> original char index.
    """
    norm_chars: List[str] = []
    index_map: List[int] = []

    prev_was_space = False
    for i, ch in enumerate(text):
        c = ch.lower()
        if c.isspace():
            if prev_was_space:
                # skip extra spaces
                continue
            norm_chars.append(" ")
            index_map.append(i)
            prev_was_space = True
        else:
            norm_chars.append(c)
            index_map.append(i)
            prev_was_space = False

    norm_text = "".join(norm_chars)
    return norm_text, index_map


def build_squad_style_instances(
    docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    From doc-level entries, build flat QA instances with
    answer_start computed via robust normalized substring search.

    Output format per instance:
    {
        "id": "docid_idx",
        "doc_id": str,
        "context": str,
        "question": str,
        "answer_text": str,
        "answer_start": int
    }
    """
    instances: List[Dict[str, Any]] = []

    total_qas = 0
    aligned_qas = 0
    skipped_qas = 0

    for doc in docs:
        doc_id = doc["doc_id"]
        context = doc["context"]

        # Build normalized context + char index map
        norm_context, ctx_index_map = normalize_with_map(context)

        for idx, qa in enumerate(doc["qa_pairs"]):
            total_qas += 1

            question = qa["question_text"]
            answer_text = qa["answer_text"]

            if not answer_text:
                skipped_qas += 1
                continue

            norm_answer, _ = normalize_with_map(answer_text)

            if not norm_answer.strip():
                skipped_qas += 1
                continue

            # Robust substring search on normalized text
            start_norm = norm_context.find(norm_answer)
            if start_norm == -1:
                # Could not align answer; skip this QA pair
                skipped_qas += 1
                continue

            # Map normalized char index -> original char index
            if start_norm >= len(ctx_index_map):
                # Safety fallback; should be rare
                skipped_qas += 1
                continue

            answer_start = ctx_index_map[start_norm]

            instance_id = f"{doc_id}_{idx}"
            instances.append(
                {
                    "id": instance_id,
                    "doc_id": doc_id,
                    "context": context,
                    "question": question,
                    "answer_text": answer_text,
                    "answer_start": answer_start,
                }
            )
            aligned_qas += 1

    print(
        f"build_squad_style_instances: total_qas={total_qas}, "
        f"aligned={aligned_qas}, skipped={skipped_qas}, "
        f"aligned%={aligned_qas / total_qas * 100:.2f}%"
    )

    return instances

def save_jsonl(instances: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in instances:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    funsd_root = Path("data/raw/funsd")
    out_path = Path("data/processed/funsd_qa.jsonl")

    train_dir = funsd_root / "training_data"
    test_dir = funsd_root / "testing_data"

    print(f"Reading FUNSD training split from: {train_dir}")
    train_docs = read_funsd_split(train_dir)
    print(f"Training documents: {len(train_docs)}")

    print(f"Reading FUNSD testing split from: {test_dir}")
    test_docs = read_funsd_split(test_dir)
    print(f"Testing documents: {len(test_docs)}")

    all_docs = train_docs + test_docs
    print(f"Total documents: {len(all_docs)}")

    instances = build_squad_style_instances(all_docs)
    print(f"Total QA instances built: {len(instances)}")

    save_jsonl(instances, out_path)
    print(f"Saved QA data to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
