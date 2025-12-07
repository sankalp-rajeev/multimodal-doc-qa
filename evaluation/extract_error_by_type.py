"""
Extract error analysis by question type using keyword-based categorization
Computes accuracy for each model on different types of questions
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
from collections import defaultdict
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


def categorize_question(question):
    """
    Categorize question into one of 5 types using keyword matching
    
    Categories:
    - Date: date, day, month, year, when
    - Name: name, who, person, company, corporation, supplier, brand
    - Address: address, street, location, city, state, zip, where
    - Amount: amount, price, cost, total, number, quantity, $
    - Other: everything else
    """
    q_lower = question.lower().strip()
    
    # Date patterns
    date_keywords = ['date', 'day', 'month', 'year', 'when', 'time', 'period']
    if any(word in q_lower for word in date_keywords):
        return 'Date'
    
    # Name patterns (person, company, brand)
    name_keywords = ['name', 'who', 'person', 'company', 'corporation', 
                     'supplier', 'brand', 'manufacturer', 'authorized']
    if any(word in q_lower for word in name_keywords):
        return 'Name'
    
    # Address patterns
    address_keywords = ['address', 'street', 'location', 'city', 'state', 
                        'zip', 'where', 'country']
    if any(word in q_lower for word in address_keywords):
        return 'Address'
    
    # Amount patterns (numbers, money)
    amount_keywords = ['amount', 'price', 'cost', 'total', 'quantity', 
                       '$', 'number', 'count']
    if any(word in q_lower for word in amount_keywords):
        return 'Amount'
    
    # Default to Other
    return 'Other'


def check_if_correct(predicted_answer, gold_answer):
    """Check if prediction matches gold answer (normalized)"""
    pred_norm = normalize_answer(predicted_answer)
    gold_norm = normalize_answer(gold_answer)
    return pred_norm == gold_norm


def evaluate_model_by_type(model, tokenizer, val_data, device, model_name, 
                            image_processor=None, is_multitask=False):
    """
    Evaluate a model on validation set and track accuracy by question type
    
    Returns:
        dict: {category: {'correct': int, 'total': int, 'accuracy': float}}
    """
    print(f"\n=== Evaluating {model_name} ===")
    
    # Track results by category
    results_by_type = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    model.eval()
    
    with torch.no_grad():
        for ex in tqdm(val_data, desc=model_name):
            question = ex['question']
            gold_answer = ex['answer_text']
            
            # Categorize question
            category = categorize_question(question)
            
            # Prepare input based on model type
            if model_name == "Model 1: BERT":
                context = ex['context']
                inputs = tokenizer(
                    question,
                    context,
                    max_length=512,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
            else:  # LayoutLMv3-based models
                words = ex['words']
                boxes = ex['bboxes']
                
                q_words = question.strip().split()
                q_boxes = [[0, 0, 0, 0]] * len(q_words)
                
                all_words = q_words + words
                all_boxes = q_boxes + boxes
                
                encoded = tokenizer(
                    all_words,
                    boxes=all_boxes,
                    max_length=512,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                # Add image if needed
                if image_processor is not None:
                    image_path = ex['image_path']
                    try:
                        image = Image.open(image_path).convert("RGB")
                        pixel_values = image_processor(image, return_tensors="pt", apply_ocr=False).pixel_values
                    except:
                        image = Image.new('RGB', (224, 224), color='white')
                        pixel_values = image_processor(image, return_tensors="pt", apply_ocr=False).pixel_values
                    
                    encoded['pixel_values'] = pixel_values
                
                inputs = {k: v.to(device) for k, v in encoded.items()}
            
            # Get prediction
            outputs = model(**inputs)
            start_idx = outputs.start_logits.argmax().item()
            end_idx = outputs.end_logits.argmax().item()
            
            if end_idx < start_idx:
                end_idx = start_idx
            
            # Decode answer
            answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
            predicted_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            
            # Check correctness
            is_correct = check_if_correct(predicted_answer, gold_answer)
            
            # Update results
            results_by_type[category]['total'] += 1
            if is_correct:
                results_by_type[category]['correct'] += 1
    
    # Compute accuracy percentages
    for category in results_by_type:
        total = results_by_type[category]['total']
        correct = results_by_type[category]['correct']
        results_by_type[category]['accuracy'] = (correct / total * 100) if total > 0 else 0.0
    
    return dict(results_by_type)


def main():
    print("="*70)
    print("ERROR ANALYSIS BY QUESTION TYPE")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load validation data for each model type
    bert_val_path = Path("data/processed/funsd_qa_val.jsonl")
    layout_val_path = Path("data/processed/funsd_layoutlmv3_val.jsonl")
    
    print("\nLoading validation data...")
    
    # BERT validation data
    bert_val_data = []
    with open(bert_val_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                bert_val_data.append(json.loads(line))
    
    # LayoutLMv3 validation data
    layout_val_data = []
    with open(layout_val_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                layout_val_data.append(json.loads(line))
    
    print(f"Loaded {len(bert_val_data)} BERT instances")
    print(f"Loaded {len(layout_val_data)} LayoutLMv3 instances")
    
    # First, categorize all questions to show distribution
    print("\n" + "="*70)
    print("QUESTION TYPE DISTRIBUTION")
    print("="*70)
    
    category_counts = defaultdict(int)
    for ex in layout_val_data:
        category = categorize_question(ex['question'])
        category_counts[category] += 1
    
    print("\nCategories found:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category:12s}: {count:3d} questions ({count/len(layout_val_data)*100:.1f}%)")
    
    # Evaluate all models
    all_results = {}
    
    # Model 1: BERT
    print("\n" + "="*70)
    bert_path = Path("models/bert_baseline/best_model")
    bert_model = BertForQuestionAnswering.from_pretrained(str(bert_path))
    bert_tokenizer = BertTokenizerFast.from_pretrained(str(bert_path))
    bert_model.to(device)
    
    all_results['Model 1: BERT'] = evaluate_model_by_type(
        bert_model, bert_tokenizer, bert_val_data, device, "Model 1: BERT"
    )
    
    # Model 2: LayoutLMv3
    print("\n" + "="*70)
    layout_path = Path("models/layoutlmv3/best_model")
    layout_config = LayoutLMv3Config.from_json_file(str(layout_path / "config.json"))
    layout_model = LayoutLMv3ForQuestionAnswering.from_pretrained(
        str(layout_path),
        config=layout_config,
        local_files_only=True
    )
    layout_tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    layout_model.to(device)
    
    all_results['Model 2: LayoutLMv3'] = evaluate_model_by_type(
        layout_model, layout_tokenizer, layout_val_data, device, "Model 2: LayoutLMv3"
    )
    
    # Model 3: LayoutLMv3 + Vision
    print("\n" + "="*70)
    vision_path = Path("models/layoutlmv3_vision/best_model")
    vision_config = LayoutLMv3Config.from_json_file(str(vision_path / "config.json"))
    vision_model = LayoutLMv3ForQuestionAnswering.from_pretrained(
        str(vision_path),
        config=vision_config,
        local_files_only=True
    )
    image_processor = LayoutLMv3ImageProcessor.from_pretrained("microsoft/layoutlmv3-base")
    vision_model.to(device)
    
    all_results['Model 3: +Vision'] = evaluate_model_by_type(
        vision_model, layout_tokenizer, layout_val_data, device, 
        "Model 3: +Vision", image_processor=image_processor
    )
    
    # Model 4: Multi-Task
    print("\n" + "="*70)
    multitask_path = Path("models/multitask/best_model")
    multitask_config = LayoutLMv3Config.from_json_file(str(multitask_path / "config.json"))
    multitask_config.num_labels = 7
    multitask_model = LayoutLMv3ForMultiTask.from_pretrained(
        str(multitask_path),
        config=multitask_config,
        local_files_only=True
    )
    multitask_model.to(device)
    
    all_results['Model 4: Multi-Task'] = evaluate_model_by_type(
        multitask_model, layout_tokenizer, layout_val_data, device,
        "Model 4: Multi-Task", image_processor=image_processor, is_multitask=True
    )
    
    # Print summary table
    print("\n" + "="*70)
    print("ACCURACY BY QUESTION TYPE (REAL DATA)")
    print("="*70)
    print(f"\n{'Category':<12} {'Count':<8} {'M1 (BERT)':<12} {'M2 (Layout)':<14} {'M3 (+Vision)':<14} {'M4 (Multi-T)':<12}")
    print("-" * 80)
    
    categories = ['Date', 'Name', 'Address', 'Amount', 'Other']
    for category in categories:
        count = category_counts[category]
        m1_acc = all_results['Model 1: BERT'].get(category, {}).get('accuracy', 0)
        m2_acc = all_results['Model 2: LayoutLMv3'].get(category, {}).get('accuracy', 0)
        m3_acc = all_results['Model 3: +Vision'].get(category, {}).get('accuracy', 0)
        m4_acc = all_results['Model 4: Multi-Task'].get(category, {}).get('accuracy', 0)
        
        print(f"{category:<12} {count:<8} {m1_acc:>6.1f}%      {m2_acc:>6.1f}%        {m3_acc:>6.1f}%        {m4_acc:>6.1f}%")
    
    # Prepare output for dashboard
    output_data = {
        'categories': categories,
        'counts': {cat: category_counts[cat] for cat in categories},
        'accuracies': {}
    }
    
    for model_name in ['Model 1: BERT', 'Model 2: LayoutLMv3', 'Model 3: +Vision', 'Model 4: Multi-Task']:
        output_data['accuracies'][model_name] = {}
        for category in categories:
            acc = all_results[model_name].get(category, {}).get('accuracy', 0)
            output_data['accuracies'][model_name][category] = round(acc, 2)
    
    # Save results
    output_path = Path("evaluation/error_by_type.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved results to {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()