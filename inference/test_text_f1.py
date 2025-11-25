import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

import torch
from transformers import LayoutLMv3ForQuestionAnswering, LayoutLMv3TokenizerFast
from src.models.layout_dataset import LayoutLMv3FunsdDataset
from src.evaluation.metrics import compute_em_f1
from tqdm import tqdm


def main():
    # Load model
    model_path = Path("models/layoutlmv3/best_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {model_path}")
    model = LayoutLMv3ForQuestionAnswering.from_pretrained(model_path)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(model_path)
    # tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    model.to(device)
    model.eval()
    
    # Load validation dataset
    val_path = Path("data/processed/funsd_layoutlmv3_val.jsonl")
    val_dataset = LayoutLMv3FunsdDataset(
        jsonl_path=val_path,
        tokenizer_name=str(model_path),
        max_seq_length=512,
    )
    
    print(f"Running inference on {len(val_dataset)} validation examples...")
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset))):
            ex = val_dataset.instances[idx]
            batch = val_dataset[idx]
            
            # Move to device and add batch dimension
            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
            bbox = batch["bbox"].unsqueeze(0).to(device)
            
            # Get predictions
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
            )
            
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
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            
            predictions.append(pred_text)
            references.append(ex["answer_text"])
    
    # Compute metrics
    metrics = compute_em_f1(predictions, references)
    
    print("\n" + "="*50)
    print("Text-based QA Metrics (Validation Set)")
    print("="*50)
    print(f"Exact Match (EM): {metrics['em'] * 100:.2f}%")
    print(f"F1 Score:         {metrics['f1'] * 100:.2f}%")
    print("="*50)
    
    # Show some examples
    print("\nSample predictions:")
    for i in range(min(5, len(predictions))):
        print(f"\nExample {i+1}:")
        print(f"  Question: {val_dataset.instances[i]['question'][:60]}...")
        print(f"  Gold:     {references[i]}")
        print(f"  Pred:     {predictions[i]}")


if __name__ == "__main__":
    main()