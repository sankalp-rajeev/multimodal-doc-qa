import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader
from src.models.dataset import FunsdQADataset



def main():
    train_path = Path("data/processed/funsd_qa_train.jsonl")

    dataset = FunsdQADataset(
        jsonl_path=str(train_path),
        tokenizer_name="bert-base-uncased",
        max_length=512,
    )

    print(f"Loaded {len(dataset)} training examples")

    # Small batch for inspection
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(dataloader))

    print("Batch keys:", batch.keys())
    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("start_positions shape:", batch["start_positions"].shape)
    print("end_positions shape:", batch["end_positions"].shape)

    # Check some sample positions
    print("start_positions:", batch["start_positions"])
    print("end_positions:", batch["end_positions"])

    # Optional: check device
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
