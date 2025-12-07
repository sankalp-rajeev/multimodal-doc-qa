"""
Extract REAL answer length statistics from trained models
Fixed version - counts actual answer spans, not padded sequences
"""

import sys
from pathlib import Path

# Get project root (parent of evaluation directory)
current_file = Path(__file__).resolve()
if current_file.parent.name == "evaluation":
    project_root = current_file.parent.parent
else:
    project_root = current_file.parent

sys.path.insert(0, str(project_root))
print(f"Project root: {project_root}")

import json
import torch
from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3TokenizerFast,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Config
)
from PIL import Image
from tqdm import tqdm
import numpy as np


def evaluate_bert():
    """Model 1: BERT Baseline"""
    print("\n=== Model 1: BERT Baseline ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("models/bert_baseline/best_model")
    val_path = Path("data/processed/funsd_qa_val.jsonl")
    
    # Load model
    model = BertForQuestionAnswering.from_pretrained(str(model_path))
    tokenizer = BertTokenizerFast.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    
    # Load validation data
    instances = []
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    
    print(f"Loaded {len(instances)} validation instances")
    
    answer_lengths = []
    
    with torch.no_grad():
        for ex in tqdm(instances, desc="BERT"):
            question = ex['question']
            context = ex['context']
            
            # Tokenize
            inputs = tokenizer(
                question,
                context,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            outputs = model(**inputs)
            start_idx = outputs.start_logits.argmax().item()
            end_idx = outputs.end_logits.argmax().item()
            
            # Fix reversed spans
            if end_idx < start_idx:
                end_idx = start_idx
            
            # Decode the actual answer
            answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
            predicted_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            
            # Count words in the predicted answer (more meaningful than subword tokens)
            if predicted_answer:
                word_count = len(predicted_answer.split())
            else:
                word_count = 0
            
            answer_lengths.append(word_count)
    
    return answer_lengths


def evaluate_layoutlmv3():
    """Model 2: LayoutLMv3 (text + layout)"""
    print("\n=== Model 2: LayoutLMv3 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("models/layoutlmv3/best_model")
    val_path = Path("data/processed/funsd_layoutlmv3_val.jsonl")
    
    # Load model
    config = LayoutLMv3Config.from_json_file(str(model_path / "config.json"))
    model = LayoutLMv3ForQuestionAnswering.from_pretrained(
        str(model_path),
        config=config,
        local_files_only=True
    )
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    model.to(device)
    model.eval()
    
    # Load validation data
    instances = []
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    
    print(f"Loaded {len(instances)} validation instances")
    
    answer_lengths = []
    
    with torch.no_grad():
        for ex in tqdm(instances, desc="LayoutLMv3"):
            question = ex['question']
            words = ex['words']
            boxes = ex['bboxes']
            
            # Prepare input
            q_words = question.strip().split()
            q_boxes = [[0, 0, 0, 0]] * len(q_words)
            
            all_words = q_words + words
            all_boxes = q_boxes + boxes
            
            # Tokenize
            encoded = tokenizer(
                all_words,
                boxes=all_boxes,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Predict
            outputs = model(**encoded)
            start_idx = outputs.start_logits.argmax().item()
            end_idx = outputs.end_logits.argmax().item()
            
            # Fix reversed spans
            if end_idx < start_idx:
                end_idx = start_idx
            
            # Decode answer
            answer_tokens = encoded['input_ids'][0][start_idx:end_idx+1]
            predicted_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            
            # Count words
            if predicted_answer:
                word_count = len(predicted_answer.split())
            else:
                word_count = 0
            
            answer_lengths.append(word_count)
    
    return answer_lengths


def evaluate_layoutlmv3_vision():
    """Model 3: LayoutLMv3 + Vision"""
    print("\n=== Model 3: LayoutLMv3 + Vision ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("models/layoutlmv3_vision/best_model")
    val_path = Path("data/processed/funsd_layoutlmv3_val.jsonl")
    
    # Load model
    config = LayoutLMv3Config.from_json_file(str(model_path / "config.json"))
    model = LayoutLMv3ForQuestionAnswering.from_pretrained(
        str(model_path),
        config=config,
        local_files_only=True
    )
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    image_processor = LayoutLMv3ImageProcessor.from_pretrained("microsoft/layoutlmv3-base")
    model.to(device)
    model.eval()
    
    # Load validation data
    instances = []
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    
    print(f"Loaded {len(instances)} validation instances")
    
    answer_lengths = []
    
    with torch.no_grad():
        for ex in tqdm(instances, desc="+Vision"):
            question = ex['question']
            words = ex['words']
            boxes = ex['bboxes']
            image_path = ex['image_path']
            
            # Prepare input
            q_words = question.strip().split()
            q_boxes = [[0, 0, 0, 0]] * len(q_words)
            
            all_words = q_words + words
            all_boxes = q_boxes + boxes
            
            # Tokenize
            encoded = tokenizer(
                all_words,
                boxes=all_boxes,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Load image
            try:
                image = Image.open(image_path).convert("RGB")
                pixel_values = image_processor(image, return_tensors="pt", apply_ocr=False).pixel_values
            except:
                image = Image.new('RGB', (224, 224), color='white')
                pixel_values = image_processor(image, return_tensors="pt", apply_ocr=False).pixel_values
            
            encoded['pixel_values'] = pixel_values
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Predict
            outputs = model(**encoded)
            start_idx = outputs.start_logits.argmax().item()
            end_idx = outputs.end_logits.argmax().item()
            
            # Fix reversed spans
            if end_idx < start_idx:
                end_idx = start_idx
            
            # Decode answer
            answer_tokens = encoded['input_ids'][0][start_idx:end_idx+1]
            predicted_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            
            # Count words
            if predicted_answer:
                word_count = len(predicted_answer.split())
            else:
                word_count = 0
            
            answer_lengths.append(word_count)
    
    return answer_lengths


def evaluate_multitask():
    """Model 4: Multi-Task"""
    print("\n=== Model 4: Multi-Task ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("models/multitask/best_model")
    val_path = Path("data/processed/funsd_layoutlmv3_val.jsonl")
    
    # Import custom model class
    from src.Models_src.multitask_model import LayoutLMv3ForMultiTask
    
    # Load model
    config = LayoutLMv3Config.from_json_file(str(model_path / "config.json"))
    config.num_labels = 7  # BIO labels
    model = LayoutLMv3ForMultiTask.from_pretrained(
        str(model_path),
        config=config,
        local_files_only=True
    )
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    image_processor = LayoutLMv3ImageProcessor.from_pretrained("microsoft/layoutlmv3-base")
    model.to(device)
    model.eval()
    
    # Load validation data
    instances = []
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    
    print(f"Loaded {len(instances)} validation instances")
    
    answer_lengths = []
    
    with torch.no_grad():
        for ex in tqdm(instances, desc="Multi-Task"):
            question = ex['question']
            words = ex['words']
            boxes = ex['bboxes']
            image_path = ex['image_path']
            
            # Prepare input
            q_words = question.strip().split()
            q_boxes = [[0, 0, 0, 0]] * len(q_words)
            
            all_words = q_words + words
            all_boxes = q_boxes + boxes
            
            # Tokenize
            encoded = tokenizer(
                all_words,
                boxes=all_boxes,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Load image
            try:
                image = Image.open(image_path).convert("RGB")
                pixel_values = image_processor(image, return_tensors="pt", apply_ocr=False).pixel_values
            except:
                image = Image.new('RGB', (224, 224), color='white')
                pixel_values = image_processor(image, return_tensors="pt", apply_ocr=False).pixel_values
            
            encoded['pixel_values'] = pixel_values
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Predict
            outputs = model(**encoded)
            start_idx = outputs.start_logits.argmax().item()
            end_idx = outputs.end_logits.argmax().item()
            
            # Fix reversed spans
            if end_idx < start_idx:
                end_idx = start_idx
            
            # Decode answer
            answer_tokens = encoded['input_ids'][0][start_idx:end_idx+1]
            predicted_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            
            # Count words
            if predicted_answer:
                word_count = len(predicted_answer.split())
            else:
                word_count = 0
            
            answer_lengths.append(word_count)
    
    return answer_lengths


def main():
    print("="*70)
    print("EXTRACTING REAL ANSWER LENGTH STATISTICS")
    print("="*70)
    
    # Evaluate all models
    bert_lengths = evaluate_bert()
    layout_lengths = evaluate_layoutlmv3()
    vision_lengths = evaluate_layoutlmv3_vision()
    multitask_lengths = evaluate_multitask()
    
    # Compute statistics
    results = {
        "Model 1: BERT": {
            "mean": float(np.mean(bert_lengths)),
            "median": float(np.median(bert_lengths)),
            "std": float(np.std(bert_lengths)),
            "min": int(np.min(bert_lengths)),
            "max": int(np.max(bert_lengths)),
            "distribution": bert_lengths
        },
        "Model 2: LayoutLMv3": {
            "mean": float(np.mean(layout_lengths)),
            "median": float(np.median(layout_lengths)),
            "std": float(np.std(layout_lengths)),
            "min": int(np.min(layout_lengths)),
            "max": int(np.max(layout_lengths)),
            "distribution": layout_lengths
        },
        "Model 3: +Vision": {
            "mean": float(np.mean(vision_lengths)),
            "median": float(np.median(vision_lengths)),
            "std": float(np.std(vision_lengths)),
            "min": int(np.min(vision_lengths)),
            "max": int(np.max(vision_lengths)),
            "distribution": vision_lengths
        },
        "Model 4: Multi-Task": {
            "mean": float(np.mean(multitask_lengths)),
            "median": float(np.median(multitask_lengths)),
            "std": float(np.std(multitask_lengths)),
            "min": int(np.min(multitask_lengths)),
            "max": int(np.max(multitask_lengths)),
            "distribution": multitask_lengths
        }
    }
    
    # Print summary
    print("\n" + "="*70)
    print("ANSWER LENGTH SUMMARY (word counts)")
    print("="*70)
    for model_name, stats in results.items():
        print(f"\n{model_name}:")
        print(f"  Mean:   {stats['mean']:.2f} words")
        print(f"  Median: {stats['median']:.1f} words")
        print(f"  Std:    {stats['std']:.2f}")
        print(f"  Range:  [{stats['min']}, {stats['max']}]")
    
    # Save to file
    output_path = Path("evaluation/answer_length_statistics.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved to {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()