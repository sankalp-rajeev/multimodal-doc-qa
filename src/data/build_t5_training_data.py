"""
File: src/data/build_t5_training_data.py

Purpose: Generate T5 training data from LayoutLMv3 predictions
"""

import json
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import LayoutLMv3ForQuestionAnswering, LayoutLMv3TokenizerFast
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.models.layout_dataset import LayoutLMv3FunsdDataset


def generate_t5_data(
    model_path: Path,
    qa_jsonl_path: Path,
    output_path: Path,
    split_name: str
):
    """
    Run LayoutLMv3 inference and create T5 training pairs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load trained LayoutLMv3
    print(f"Loading LayoutLMv3 from {model_path}")
    model = LayoutLMv3ForQuestionAnswering.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load dataset
    print(f"Loading {split_name} dataset...")
    dataset = LayoutLMv3FunsdDataset(
        jsonl_path=qa_jsonl_path,
        tokenizer_name="microsoft/layoutlmv3-base",
        max_seq_length=512,
    )
    
    tokenizer = dataset.tokenizer
    
    print(f"Generating T5 data for {split_name} split ({len(dataset)} examples)...")
    
    t5_pairs = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Processing {split_name}"):
            # Get original instance
            ex = dataset.instances[idx]
            
            # Get model input
            batch = dataset[idx]
            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
            bbox = batch["bbox"].unsqueeze(0).to(device)
            
            # Run LayoutLMv3
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
            )
            
            # Extract predicted span
            start_idx = outputs.start_logits.argmax(dim=-1).item()
            end_idx = outputs.end_logits.argmax(dim=-1).item()
            
            # Fix reversed spans
            if end_idx < start_idx:
                end_idx = start_idx
            
            # Limit span length
            if end_idx - start_idx > 30:
                end_idx = start_idx + 30
            
            # Decode prediction
            pred_tokens = input_ids[0][start_idx:end_idx+1]
            extracted_span = tokenizer.decode(
                pred_tokens, 
                skip_special_tokens=True
            ).strip()
            
            # Handle empty predictions
            if not extracted_span:
                extracted_span = "[NO ANSWER]"
            
            # Create T5 training pair
            question = ex["question"]
            gold_answer = ex["answer_text"]
            
            # T5 input format
            input_text = f"refine answer: question: {question} extracted: {extracted_span}"
            target_text = gold_answer
            
            t5_pairs.append({
                "id": ex["id"],
                "doc_id": ex["doc_id"],
                "input_text": input_text,
                "target_text": target_text,
                "extracted_span": extracted_span,
                "question": question,
            })
    
    # Save T5 training data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pair in t5_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    print(f"\nSaved {len(t5_pairs)} T5 training pairs to {output_path}")
    
    # Print statistics
    empty_predictions = sum(1 for p in t5_pairs if p["extracted_span"] == "[NO ANSWER]")
    exact_matches = sum(1 for p in t5_pairs if p["extracted_span"] == p["target_text"])
    
    print(f"\nStatistics for {split_name}:")
    print(f"  Total pairs: {len(t5_pairs)}")
    print(f"  Empty predictions: {empty_predictions} ({empty_predictions/len(t5_pairs)*100:.2f}%)")
    print(f"  Exact matches: {exact_matches} ({exact_matches/len(t5_pairs)*100:.2f}%)")
    
    # Print samples
    print(f"\nSample T5 pairs from {split_name}:")
    for i in range(min(5, len(t5_pairs))):
        print(f"\n  Example {i+1}:")
        print(f"    Input:  {t5_pairs[i]['input_text'][:80]}...")
        print(f"    Target: {t5_pairs[i]['target_text']}")


def main():
    model_path = Path("models/layoutlmv3/best_model")
    
    print("="*60)
    print("T5 Training Data Generation")
    print("="*60)
    
    # Generate for train split
    print("\n[1/2] Processing training set...")
    generate_t5_data(
        model_path=model_path,
        qa_jsonl_path=Path("data/processed/funsd_layoutlmv3_train.jsonl"),
        output_path=Path("data/processed/t5_train.jsonl"),
        split_name="train"
    )
    
    # Generate for val split
    print("\n[2/2] Processing validation set...")
    generate_t5_data(
        model_path=model_path,
        qa_jsonl_path=Path("data/processed/funsd_layoutlmv3_val.jsonl"),
        output_path=Path("data/processed/t5_val.jsonl"),
        split_name="val"
    )
    
    print("\n" + "="*60)
    print("✓ T5 training data generation complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - data/processed/t5_train.jsonl")
    print("  - data/processed/t5_val.jsonl")
    print("\nNext step: Train T5 model using these files")


if __name__ == "__main__":
    main()