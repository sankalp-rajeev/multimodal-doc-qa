import json
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

def read_funsd_layout(split_dir: Path) -> List[Dict[str, Any]]:
    """
    Reads FUNSD annotations in one split directory and returns
    document-level tokens and bounding boxes (word-level).
    
    Output per doc:
    {
        "doc_id": str,
        "image_path": str,
        "tokens": [str, ...],
        "bboxes": [[x0, y0, x1, y1], ...]
    }
    """
    ann_dir = split_dir / "annotations"
    img_dir = split_dir / "images"

    docs: List[Dict[str, Any]] = []

    for ann_file in sorted(ann_dir.glob("*.json")):
        with ann_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        form_fields = data.get("form", [])

        tokens: List[str] = []
        bboxes: List[List[int]] = []

        for field in form_fields:
            for w in field.get("words", []):
                text = (w.get("text") or "").strip()
                box = w.get("box", None)
                if text and box and len(box) == 4:
                    tokens.append(text)
                    # Ensure ints
                    bboxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])

        doc_id = ann_file.stem
        image_path = img_dir / f"{doc_id}.png"
        # Get image dimensions
        if image_path.exists():
            img = Image.open(image_path)
            img_width, img_height = img.size
        else:
            # Fallback if image missing
            img_width, img_height = 1000, 1000

        docs.append(
            {
                "doc_id": doc_id,
                "image_path": str(image_path),
                "tokens": tokens,
                "bboxes": bboxes,
                "img_width": img_width,      
                "img_height": img_height,    
            }
        )

    return docs


def save_jsonl(docs: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def main():
    funsd_root = Path("data/raw/funsd")
    out_path = Path("data/processed/funsd_layout_docs.jsonl")

    train_dir = funsd_root / "training_data"
    test_dir = funsd_root / "testing_data"

    print(f"Reading FUNSD layout from: {train_dir}")
    train_docs = read_funsd_layout(train_dir)
    print(f"Training layout docs: {len(train_docs)}")

    print(f"Reading FUNSD layout from: {test_dir}")
    test_docs = read_funsd_layout(test_dir)
    print(f"Testing layout docs: {len(test_docs)}")

    all_docs = train_docs + test_docs
    print(f"Total layout docs: {len(all_docs)}")

    save_jsonl(all_docs, out_path)
    print(f"Saved layout docs to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
