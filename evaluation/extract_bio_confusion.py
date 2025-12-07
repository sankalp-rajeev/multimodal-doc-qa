"""
Extract REAL BIO confusion matrix from Multi-Task Model
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
import numpy as np
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ImageProcessor, LayoutLMv3Config
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from src.Models_src.multitask_model import LayoutLMv3ForMultiTask


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


def extract_confusion_matrix():
    """Extract real confusion matrix from Model 4 predictions"""
    
    print("="*70)
    print("EXTRACTING BIO CONFUSION MATRIX FROM MULTI-TASK MODEL")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("models/multitask/best_model")
    qa_val_path = Path("data/processed/funsd_layoutlmv3_val.jsonl")
    bio_val_path = Path("data/processed/funsd_bio_val.jsonl")
    
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading Multi-Task model...")
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
    print("✓ Model loaded")
    
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
    
    # Collect all predictions and ground truths
    all_predictions = []
    all_ground_truths = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for qa_inst in tqdm(qa_instances, desc="Extracting BIO predictions"):
            doc_id = qa_inst['doc_id']
            bio_doc = bio_docs.get(doc_id)
            
            if bio_doc is None:
                continue
            
            question = qa_inst['question']
            words = qa_inst['words']
            boxes = qa_inst['bboxes']
            image_path = qa_inst['image_path']
            bio_labels = bio_doc['bio_labels']
            
            # Prepare input
            q_words = question.strip().split()
            q_boxes = [[0, 0, 0, 0]] * len(q_words)
            q_bio_labels = [0] * len(q_words)
            
            all_words = q_words + words
            all_boxes = q_boxes + boxes
            all_bio_labels = q_bio_labels + bio_labels
            
            shift = len(q_words)
            
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
                image = Image.new('RGB', (224, 224), color='white')
                pixel_values = image_processor(image, return_tensors="pt", apply_ocr=False).pixel_values
            
            # Prepare inputs for model
            model_inputs = {
                'input_ids': encoding['input_ids'].to(device),
                'bbox': encoding['bbox'].to(device),
                'attention_mask': encoding['attention_mask'].to(device),
                'pixel_values': pixel_values.to(device)
            }
            
            # Predict
            outputs = model(**model_inputs)
            
            # Get BIO predictions
            sequence_output = model.layoutlmv3(
                input_ids=model_inputs['input_ids'],
                bbox=model_inputs['bbox'],
                attention_mask=model_inputs['attention_mask'],
                pixel_values=model_inputs['pixel_values']
            ).last_hidden_state
            
            bio_logits = model.classifier(sequence_output)
            predicted_labels = bio_logits.argmax(dim=-1)[0]
            
            # Map to word-level labels using the original encoding object
            word_ids = encoding.word_ids(batch_index=0)
            
            for i in range(len(words)):
                word_id_to_find = shift + i
                # Find first token for this word
                for j, w_id in enumerate(word_ids):
                    if w_id == word_id_to_find:
                        pred_label = predicted_labels[j].item()
                        true_label = bio_labels[i]
                        
                        all_predictions.append(pred_label)
                        all_ground_truths.append(true_label)
                        break
    
    print(f"\n✓ Collected {len(all_predictions)} token predictions")
    
    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    conf_matrix = confusion_matrix(
        all_ground_truths, 
        all_predictions,
        labels=list(range(7))  # 0-6 for all BIO labels
    )
    
    # Convert to list for JSON serialization
    conf_matrix_list = conf_matrix.tolist()
    
    # Calculate percentages (normalize by row)
    conf_matrix_pct = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True) * 100
    conf_matrix_pct_list = conf_matrix_pct.tolist()
    
    # Prepare output
    result = {
        "confusion_matrix": conf_matrix_list,
        "confusion_matrix_percentages": conf_matrix_pct_list,
        "labels": list(ID2LABEL.values()),
        "label_mapping": ID2LABEL,
        "total_predictions": len(all_predictions)
    }
    
    # Save to JSON
    output_path = Path("evaluation/bio_confusion_matrix.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Saved confusion matrix to {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("CONFUSION MATRIX SUMMARY")
    print("="*70)
    print("\nLabel counts in predictions:")
    for label_id, label_name in ID2LABEL.items():
        count = all_predictions.count(label_id)
        pct = count / len(all_predictions) * 100
        print(f"  {label_name:12s}: {count:6d} ({pct:5.2f}%)")
    
    print("\nDiagonal accuracy (correctly predicted):")
    for i, label_name in enumerate(ID2LABEL.values()):
        if conf_matrix[i, i] > 0:
            accuracy = conf_matrix[i, i] / conf_matrix[i, :].sum() * 100
            print(f"  {label_name:12s}: {accuracy:5.2f}%")
    
    print("\n" + "="*70)
    
    return result


if __name__ == "__main__":
    extract_confusion_matrix()