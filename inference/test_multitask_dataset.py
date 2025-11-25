from pathlib import Path
from src.models.multitask_dataset import MultiTaskDataset

# Test loading
dataset = MultiTaskDataset(
    qa_jsonl_path=Path("data/processed/funsd_layoutlmv3_train.jsonl"),
    bio_jsonl_path=Path("data/processed/funsd_bio_train.jsonl"),
)

# Get one sample
sample = dataset[0]

print("Sample batch:")
print(f"  input_ids shape: {sample['input_ids'].shape}")
print(f"  bbox shape: {sample['bbox'].shape}")
print(f"  pixel_values shape: {sample['pixel_values'].shape}")
print(f"  start_positions: {sample['start_positions']}")
print(f"  end_positions: {sample['end_positions']}")
print(f"  bio_labels shape: {sample['labels'].shape}")
print(f"  bio_labels (first 20): {sample['labels'][:20]}")
print("\n✓ Dataset works!")