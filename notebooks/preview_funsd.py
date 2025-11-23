import sys
import os
from pathlib import Path

# --- make project root importable ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent  # one level up from notebooks/
sys.path.append(str(PROJECT_ROOT))
# ------------------------------------

import matplotlib.pyplot as plt
from PIL import Image

from src.data.funsd_utils import load_funsd_all

def main():
    funsd_root = Path("data/raw/funsd")
    docs = load_funsd_all(funsd_root)

    print(f"Loaded {len(docs)} documents from FUNSD")

    # Just take the first 10 docs that have at least 1 QA pair
    count = 0
    for doc in docs:
        if not doc["qa_pairs"]:
            continue

        print("=" * 80)
        print(f"Doc ID: {doc['doc_id']}")
        print(f"Image: {doc['image_path']}")

        # Print up to 3 QA pairs per doc for inspection
        for i, qa in enumerate(doc["qa_pairs"][:3]):
            print(f"  Q{i+1}: {qa['question_text']}")
            print(f"  A{i+1}: {qa['answer_text']}")
            print()

        # Show image
        img_path = doc["image_path"]
        if img_path.is_file():
            img = Image.open(img_path).convert("RGB")
            plt.figure(figsize=(8, 10))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Doc: {doc['doc_id']}")
            plt.show()
        else:
            print(f"Image not found: {img_path}")

        count += 1
        if count >= 10:
            break


if __name__ == "__main__":
    main()
