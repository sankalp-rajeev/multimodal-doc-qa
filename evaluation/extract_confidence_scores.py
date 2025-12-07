"""
Extract REAL confidence scores from all 4 models
Confidence = softmax probability of predicted answer span
"""

import sys
from pathlib import Path

# Get project root
current_file = Path(__file__).resolve()
if current_file.parent.name == "evaluation":
    project_root = current_file.parent.parent
else:
    project_root = current_file.parent

sys.path.insert(0, str(project_root))

import json
import torch
import torch.nn.functional as F
import numpy as np
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

from src.Models_src.multitask_model import LayoutLMv3ForMultiTask
from src.evaluation.metrics import normalize_answer


def compute_confidence(start_logits, end_logits, start_idx, end_idx):
    """
    Compute confidence as the product of softmax probabilities
    for the predicted start and end positions
    """
    start_probs = F.softmax(start_logits, dim=-1)
    end_probs = F.softmax(end_logits, dim=-1)
    
    confidence = start_probs[0, start_idx].item() * end_probs[0, end_idx].item()
    return confidence


def check_if_correct(predicted_answer, gold_answer):
    """Check if prediction matches gold answer (normalized)"""
    pred_norm = normalize_answer(predicted_answer)
    gold_norm = normalize_answer(gold_answer)
    return pred_norm == gold_norm


def evaluate_bert_confidence():
    """Model 1: BERT confidence scores"""
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
    
    correct_confidences = []
    incorrect_confidences = []
    
    with torch.no_grad():
        for ex in tqdm(instances, desc="BERT"):
            question = ex['question']
            context = ex['context']
            gold_answer = ex['answer_text']
            
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
            
            if end_idx < start_idx:
                end_idx = start_idx
            
            # Compute confidence
            confidence = compute_confidence(outputs.start_logits, outputs.end_logits, start_idx, end_idx)
            
            # Decode prediction
            answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
            predicted_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            
            # Check correctness
            is_correct = check_if_correct(predicted_answer, gold_answer)
            
            if is_correct:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)
    
    return correct_confidences, incorrect_confidences


def evaluate_layoutlmv3_confidence():
    """Model 2: LayoutLMv3 confidence scores"""
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
    
    correct_confidences = []
    incorrect_confidences = []
    
    with torch.no_grad():
        for ex in tqdm(instances, desc="LayoutLMv3"):
            question = ex['question']
            words = ex['words']
            boxes = ex['bboxes']
            gold_answer = ex['answer_text']
            
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
            
            if end_idx < start_idx:
                end_idx = start_idx
            
            # Compute confidence
            confidence = compute_confidence(outputs.start_logits, outputs.end_logits, start_idx, end_idx)
            
            # Decode prediction
            answer_tokens = encoded['input_ids'][0][start_idx:end_idx+1]
            predicted_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            
            # Check correctness
            is_correct = check_if_correct(predicted_answer, gold_answer)
            
            if is_correct:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)
    
    return correct_confidences, incorrect_confidences


def evaluate_vision_confidence():
    """Model 3: LayoutLMv3 + Vision confidence scores"""
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
    
    correct_confidences = []
    incorrect_confidences = []
    
    with torch.no_grad():
        for ex in tqdm(instances, desc="+Vision"):
            question = ex['question']
            words = ex['words']
            boxes = ex['bboxes']
            image_path = ex['image_path']
            gold_answer = ex['answer_text']
            
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
            
            if end_idx < start_idx:
                end_idx = start_idx
            
            # Compute confidence
            confidence = compute_confidence(outputs.start_logits, outputs.end_logits, start_idx, end_idx)
            
            # Decode prediction
            answer_tokens = encoded['input_ids'][0][start_idx:end_idx+1]
            predicted_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            
            # Check correctness
            is_correct = check_if_correct(predicted_answer, gold_answer)
            
            if is_correct:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)
    
    return correct_confidences, incorrect_confidences


def evaluate_multitask_confidence():
    """Model 4: Multi-Task confidence scores"""
    print("\n=== Model 4: Multi-Task ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("models/multitask/best_model")
    val_path = Path("data/processed/funsd_layoutlmv3_val.jsonl")
    
    # Load model
    config = LayoutLMv3Config.from_json_file(str(model_path / "config.json"))
    config.num_labels = 7
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
    
    correct_confidences = []
    incorrect_confidences = []
    
    with torch.no_grad():
        for ex in tqdm(instances, desc="Multi-Task"):
            question = ex['question']
            words = ex['words']
            boxes = ex['bboxes']
            image_path = ex['image_path']
            gold_answer = ex['answer_text']
            
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
            
            if end_idx < start_idx:
                end_idx = start_idx
            
            # Compute confidence
            confidence = compute_confidence(outputs.start_logits, outputs.end_logits, start_idx, end_idx)
            
            # Decode prediction
            answer_tokens = encoded['input_ids'][0][start_idx:end_idx+1]
            predicted_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            
            # Check correctness
            is_correct = check_if_correct(predicted_answer, gold_answer)
            
            if is_correct:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)
    
    return correct_confidences, incorrect_confidences


def main():
    print("="*70)
    print("EXTRACTING CONFIDENCE SCORES FROM ALL MODELS")
    print("="*70)
    
    # Evaluate all models
    bert_correct, bert_incorrect = evaluate_bert_confidence()
    layout_correct, layout_incorrect = evaluate_layoutlmv3_confidence()
    vision_correct, vision_incorrect = evaluate_vision_confidence()
    multitask_correct, multitask_incorrect = evaluate_multitask_confidence()
    
    # Prepare results
    results = {
        "Model 1: BERT": {
            "correct_confidences": bert_correct,
            "incorrect_confidences": bert_incorrect,
            "correct_mean": float(np.mean(bert_correct)) if bert_correct else 0.0,
            "incorrect_mean": float(np.mean(bert_incorrect)) if bert_incorrect else 0.0,
            "correct_count": len(bert_correct),
            "incorrect_count": len(bert_incorrect)
        },
        "Model 2: LayoutLMv3": {
            "correct_confidences": layout_correct,
            "incorrect_confidences": layout_incorrect,
            "correct_mean": float(np.mean(layout_correct)) if layout_correct else 0.0,
            "incorrect_mean": float(np.mean(layout_incorrect)) if layout_incorrect else 0.0,
            "correct_count": len(layout_correct),
            "incorrect_count": len(layout_incorrect)
        },
        "Model 3: +Vision": {
            "correct_confidences": vision_correct,
            "incorrect_confidences": vision_incorrect,
            "correct_mean": float(np.mean(vision_correct)) if vision_correct else 0.0,
            "incorrect_mean": float(np.mean(vision_incorrect)) if vision_incorrect else 0.0,
            "correct_count": len(vision_correct),
            "incorrect_count": len(vision_incorrect)
        },
        "Model 4: Multi-Task": {
            "correct_confidences": multitask_correct,
            "incorrect_confidences": multitask_incorrect,
            "correct_mean": float(np.mean(multitask_correct)) if multitask_correct else 0.0,
            "incorrect_mean": float(np.mean(multitask_incorrect)) if multitask_incorrect else 0.0,
            "correct_count": len(multitask_correct),
            "incorrect_count": len(multitask_incorrect)
        }
    }
    
    # Print summary
    print("\n" + "="*70)
    print("CONFIDENCE SCORE SUMMARY")
    print("="*70)
    for model_name, stats in results.items():
        print(f"\n{model_name}:")
        print(f"  Correct predictions:   {stats['correct_count']:3d} (mean confidence: {stats['correct_mean']:.4f})")
        print(f"  Incorrect predictions: {stats['incorrect_count']:3d} (mean confidence: {stats['incorrect_mean']:.4f})")
        print(f"  Confidence gap: {stats['correct_mean'] - stats['incorrect_mean']:.4f}")
    
    # Save to JSON
    output_path = Path("evaluation/confidence_scores.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved confidence scores to {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()