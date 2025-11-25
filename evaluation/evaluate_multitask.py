"""
Evaluate Multi-Task Model on BOTH tasks:
1. Span Extraction (QA) - F1 and Exact Match
2. BIO Tagging (Token Classification) - Precision, Recall, F1
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import json
import torch
from collections import defaultdict
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ImageProcessor
from PIL import Image

from src.models.multitask_model import LayoutLMv3ForMultiTask
from transformers import LayoutLMv3Config


# BIO Label mapping
LABEL2ID = {
    "O": 0,
    "B-ANSWER": 1,
    "I-ANSWER": 2,
    "B-QUESTION": 3,
    "I-QUESTION": 4,
    "B-HEADER": 5,
    "I-HEADER": 6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def normalize_answer(s):
    """Normalize answer text for comparison"""
    import re
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, ground_truth):
    """Compute token-level F1 score"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common = set(pred_tokens) & set(truth_tokens)
    num_same = len(common)
    
    if num_same == 0:
        return 0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def compute_exact_match(prediction, ground_truth):
    """Compute exact match score"""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def load_model_and_tokenizer(model_path: Path):
    """Load trained multi-task model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    config = LayoutLMv3Config.from_json_file(str(model_path / "config.json"))
    
    # Load model
    model = LayoutLMv3ForMultiTask.from_pretrained(
        str(model_path),
        config=config,
        local_files_only=True
    )
    model.to(device)
    model.eval()
    
    # Load tokenizer and image processor
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    image_processor = LayoutLMv3ImageProcessor.from_pretrained("microsoft/layoutlmv3-base")
    
    return model, tokenizer, image_processor, device


def predict(model, tokenizer, image_processor, device, qa_instance, bio_doc):
    """
    Get predictions for BOTH tasks:
    1. QA: predicted answer span
    2. BIO: predicted labels for each token
    """
    question = qa_instance['question']
    words = qa_instance['words']
    boxes = qa_instance['bboxes']
    image_path = qa_instance['image_path']
    
    # Prepare input
    q_words = question.strip().split()
    q_boxes = [[0, 0, 0, 0]] * len(q_words)
    
    all_words = q_words + words
    all_boxes = q_boxes + boxes
    
    # Tokenize - keep the encoding object for word_ids
    encoding = tokenizer(
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
        # Fallback to dummy image
        image = Image.new('RGB', (224, 224), color='white')
        pixel_values = image_processor(image, return_tensors="pt", apply_ocr=False).pixel_values
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    bbox = encoding['bbox'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    pixel_values = pixel_values.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
    
    # TASK 1: QA Prediction (Span Extraction)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    start_idx = start_logits.argmax(dim=1).item()
    end_idx = end_logits.argmax(dim=1).item()
    
    if end_idx < start_idx:
        end_idx = start_idx
    
    # Decode answer
    answer_tokens = input_ids[0][start_idx:end_idx+1]
    predicted_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    
    if not predicted_answer:
        predicted_answer = "[NO ANSWER]"
    
    # TASK 2: BIO Prediction (Token Classification)
    # Get bio_logits from the model's classifier head
    sequence_output = model.layoutlmv3(
        input_ids=input_ids,
        bbox=bbox,
        attention_mask=attention_mask,
        pixel_values=pixel_values
    ).last_hidden_state
    
    bio_logits = model.classifier(sequence_output)
    predicted_labels = bio_logits.argmax(dim=-1)[0]  # [seq_len]
    
    # Map back to word-level labels (skip question words)
    word_ids = encoding.word_ids(batch_index=0)  # NOW THIS WORKS
    predicted_bio_labels = []
    
    shift = len(q_words)
    for i in range(len(words)):
        word_id_to_find = shift + i
        # Find first token for this word
        for j, w_id in enumerate(word_ids):
            if w_id == word_id_to_find:
                predicted_bio_labels.append(predicted_labels[j].item())
                break
        else:
            predicted_bio_labels.append(0)  # Default to "O"
    
    return predicted_answer, predicted_bio_labels


def evaluate_qa_task(predictions, ground_truths):
    """Evaluate QA performance"""
    f1_scores = []
    em_scores = []
    
    for pred, truth in zip(predictions, ground_truths):
        f1_scores.append(compute_f1(pred, truth))
        em_scores.append(compute_exact_match(pred, truth))
    
    return {
        'f1': sum(f1_scores) / len(f1_scores) * 100,
        'exact_match': sum(em_scores) / len(em_scores) * 100,
        'total': len(predictions)
    }


def evaluate_bio_task(predicted_labels, true_labels):
    """Evaluate BIO tagging performance (token-level)"""
    # Flatten all predictions and labels
    all_preds = []
    all_trues = []
    
    for pred_seq, true_seq in zip(predicted_labels, true_labels):
        all_preds.extend(pred_seq)
        all_trues.extend(true_seq)
    
    # Compute per-class metrics
    metrics = {}
    
    for label_id, label_name in ID2LABEL.items():
        tp = sum(1 for p, t in zip(all_preds, all_trues) if p == label_id and t == label_id)
        fp = sum(1 for p, t in zip(all_preds, all_trues) if p == label_id and t != label_id)
        fn = sum(1 for p, t in zip(all_preds, all_trues) if p != label_id and t == label_id)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[label_name] = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'support': sum(1 for t in all_trues if t == label_id)
        }
    
    # Compute macro average (average across classes)
    macro_f1 = sum(m['f1'] for m in metrics.values()) / len(metrics)
    
    # Compute micro average (weighted by support)
    total_tp = sum(1 for p, t in zip(all_preds, all_trues) if p == t and t != 0)  # Exclude "O"
    total_pred = sum(1 for p in all_preds if p != 0)
    total_true = sum(1 for t in all_trues if t != 0)
    
    micro_precision = total_tp / total_pred if total_pred > 0 else 0
    micro_recall = total_tp / total_true if total_true > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    return {
        'per_class': metrics,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1 * 100,
        'total_tokens': len(all_preds)
    }


def main():
    print("="*70)
    print("Multi-Task Model Evaluation")
    print("="*70)
    
    # Paths
    model_path = Path("models/multitask/best_model")
    qa_val_path = Path("data/processed/funsd_layoutlmv3_val.jsonl")
    bio_val_path = Path("data/processed/funsd_bio_val.jsonl")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer, image_processor, device = load_model_and_tokenizer(model_path)
    print(f"✓ Model loaded on {device}")
    
    # Load validation data
    print("\nLoading validation data...")
    qa_instances = []
    with open(qa_val_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                qa_instances.append(json.loads(line))
    
    bio_docs = {}
    with open(bio_val_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                bio_doc = json.loads(line)
                bio_docs[bio_doc['doc_id']] = bio_doc
    
    print(f"✓ Loaded {len(qa_instances)} QA instances")
    print(f"✓ Loaded {len(bio_docs)} BIO documents")
    
    # Evaluate
    print("\nEvaluating both tasks...")
    qa_predictions = []
    qa_ground_truths = []
    bio_predictions = []
    bio_ground_truths = []
    
    for i, qa_instance in enumerate(qa_instances):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(qa_instances)} instances...")
        
        doc_id = qa_instance['doc_id']
        bio_doc = bio_docs.get(doc_id)
        
        if bio_doc is None:
            continue
        
        # Get predictions
        pred_answer, pred_bio_labels = predict(
            model, tokenizer, image_processor, device,
            qa_instance, bio_doc
        )
        
        # Store QA results
        qa_predictions.append(pred_answer)
        qa_ground_truths.append(qa_instance['answer_text'])
        
        # Store BIO results
        bio_predictions.append(pred_bio_labels)
        bio_ground_truths.append(bio_doc['bio_labels'])
    
    print(f"✓ Evaluation complete!")
    
    # Compute metrics
    print("\n" + "="*70)
    print("TASK 1: SPAN EXTRACTION (QA)")
    print("="*70)
    
    qa_metrics = evaluate_qa_task(qa_predictions, qa_ground_truths)
    print(f"\nResults on {qa_metrics['total']} examples:")
    print(f"  Exact Match: {qa_metrics['exact_match']:.2f}%")
    print(f"  F1 Score:    {qa_metrics['f1']:.2f}%")
    
    print("\n" + "="*70)
    print("TASK 2: BIO TAGGING (Token Classification)")
    print("="*70)
    
    bio_metrics = evaluate_bio_task(bio_predictions, bio_ground_truths)
    print(f"\nResults on {bio_metrics['total_tokens']} tokens:")
    print(f"  Micro F1 (entity-level): {bio_metrics['micro_f1']:.2f}%")
    print(f"  Macro F1 (class-avg):    {bio_metrics['macro_f1']:.2f}%")
    
    print("\nPer-class performance:")
    for label_name, metrics in bio_metrics['per_class'].items():
        print(f"  {label_name:12s}  P: {metrics['precision']:5.2f}%  R: {metrics['recall']:5.2f}%  F1: {metrics['f1']:5.2f}%  (n={metrics['support']})")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Span Extraction F1:  {qa_metrics['f1']:.2f}%")
    print(f"BIO Tagging F1:      {bio_metrics['micro_f1']:.2f}%")
    print("="*70)
    
    # Save results
    results = {
        'qa_metrics': qa_metrics,
        'bio_metrics': {
            'micro_f1': bio_metrics['micro_f1'],
            'macro_f1': bio_metrics['macro_f1'],
            'per_class': bio_metrics['per_class']
        }
    }
    
    results_path = Path("evaluation/multitask_results.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")


if __name__ == "__main__":
    main()