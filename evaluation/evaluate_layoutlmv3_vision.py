"""
Evaluate LayoutLMv3 + Vision (Model 3) on validation set
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import torch
from transformers import (
    LayoutLMv3ForQuestionAnswering, 
    LayoutLMv3TokenizerFast,
    LayoutLMv3ImageProcessor  # ADD THIS
)
from src.models.layout_dataset_vision import LayoutLMv3VisionDataset
from src.evaluation.metrics import compute_em_f1
from tqdm import tqdm


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Model 3 (LayoutLMv3 + Vision)...")
    model_path = Path("models/layoutlmv3_vision/best_model")
    model = LayoutLMv3ForQuestionAnswering.from_pretrained(model_path)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")  # CHANGED
    model.to(device)
    model.eval()
    
    print("Loading validation dataset...")
    val_dataset = LayoutLMv3VisionDataset(
        jsonl_path=Path("data/processed/funsd_layoutlmv3_val.jsonl"),
        tokenizer_name="microsoft/layoutlmv3-base",  # CHANGED
        max_seq_length=512,
    )
    
    print(f"\nEvaluating on {len(val_dataset)} examples...")
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="Evaluating"):
            # Get batch
            batch = val_dataset[idx]
            
            # Move to device
            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
            bbox = batch["bbox"].unsqueeze(0).to(device)
            pixel_values = batch["pixel_values"].unsqueeze(0).to(device)
            
            # Get original instance for gold answer
            instance = val_dataset.instances[idx]
            gold_answer = instance["answer_text"]
            
            # Predict
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values,
            )
            
            # Extract span
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
            predicted_answer = tokenizer.decode(
                pred_tokens,
                skip_special_tokens=True
            ).strip()
            
            if not predicted_answer:
                predicted_answer = "[NO ANSWER]"
            
            predictions.append(predicted_answer)
            references.append(gold_answer)
    
    # Compute metrics
    metrics = compute_em_f1(predictions, references)
    
    print("\n" + "="*60)
    print("Model 3 (LayoutLMv3 + Vision) - Final Results")
    print("="*60)
    print(f"Exact Match (EM): {metrics['em'] * 100:.2f}%")
    print(f"F1 Score:         {metrics['f1'] * 100:.2f}%")
    print(f"Total examples:   {len(predictions)}")
    print("="*60)
    
    # Show sample predictions
    print("\nSample Predictions:")
    for i in range(min(10, len(predictions))):
        print(f"\nExample {i+1}:")
        print(f"  Gold:      {references[i]}")
        print(f"  Predicted: {predictions[i]}")
        if predictions[i] != references[i]:
            print(f"  → MISMATCH")
    
    return metrics


if __name__ == "__main__":
    evaluate_model()