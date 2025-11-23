import json
from pathlib import Path
from typing import Dict, List, Any


def load_funsd_split(funsd_root: Path, split: str) -> List[Dict[str, Any]]:
    """
    Load FUNSD split ('training_data' or 'testing_data') and return
    a list of documents with image path and QA pairs.

    Returns a list of dicts:
    {
        "doc_id": str,
        "image_path": Path,
        "qa_pairs": [
            {
                "question_text": str,
                "answer_text": str,
                "question_box": [x0, y0, x1, y1],
                "answer_box": [x0, y0, x1, y1],
            },
            ...
        ]
    }
    """
    split_dir = funsd_root / split
    ann_dir = split_dir / "annotations"
    img_dir = split_dir / "images"

    docs: List[Dict[str, Any]] = []

    for ann_file in sorted(ann_dir.glob("*.json")):
        with ann_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        form_fields = data.get("form", [])
        # Map field id -> field dict
        id_to_field: Dict[Any, Dict[str, Any]] = {}
        for field in form_fields:
            field_id = field.get("id", None)
            if field_id is None:
                # Fallback to index if 'id' missing
                field_id = len(id_to_field)
            id_to_field[field_id] = field

        qa_pairs: List[Dict[str, Any]] = []

        # Build a lookup from (q_id -> list of a_ids) using 'linking'
        question_to_answers: Dict[Any, List[Any]] = {}
        for field in form_fields:
            field_id = field.get("id", None)
            if field_id is None:
                continue
            for link in field.get("linking", []):
                # link is [id1, id2]; in FUNSD it's usually [question_id, answer_id]
                if len(link) != 2:
                    continue
                q_id, a_id = link
                # Be robust in case order is reversed
                if q_id == field_id:
                    question_to_answers.setdefault(q_id, []).append(a_id)
                elif a_id == field_id:
                    question_to_answers.setdefault(a_id, []).append(q_id)

        # Now go through all fields that are labeled as question
        for field in form_fields:
            label = field.get("label", "").lower()
            if label != "question":
                continue

            q_id = field.get("id", None)
            if q_id is None:
                continue

            question_text = (field.get("text") or "").strip()
            if not question_text:
                continue

            q_box = field.get("box", None)
            # box is [x0, y0, x1, y1]
            candidate_answer_ids = question_to_answers.get(q_id, [])

            # If no linked answers, skip
            if not candidate_answer_ids:
                continue

            for a_id in candidate_answer_ids:
                answer_field = id_to_field.get(a_id)
                if not answer_field:
                    continue
                answer_text = (answer_field.get("text") or "").strip()
                if not answer_text:
                    continue
                a_box = answer_field.get("box", None)

                qa_pairs.append(
                    {
                        "question_text": question_text,
                        "answer_text": answer_text,
                        "question_box": q_box,
                        "answer_box": a_box,
                    }
                )

        doc_id = ann_file.stem
        image_path = img_dir / f"{doc_id}.png"

        docs.append(
            {
                "doc_id": doc_id,
                "image_path": image_path,
                "qa_pairs": qa_pairs,
            }
        )

    return docs


def load_funsd_all(funsd_root: Path) -> List[Dict[str, Any]]:
    """
    Convenience helper: load both training and testing splits.
    """
    all_docs: List[Dict[str, Any]] = []
    for split in ["training_data", "testing_data"]:
        split_docs = load_funsd_split(funsd_root, split)
        all_docs.extend(split_docs)
    return all_docs
