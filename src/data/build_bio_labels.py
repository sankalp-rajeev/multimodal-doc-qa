"""
Generate BIO labels aligned with QA dataset
Uses cleaned QA training data to avoid test contamination
"""

import json
from pathlib import Path
from collections import defaultdict

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


def load_funsd_annotations(funsd_dir: Path):
    """Load ALL FUNSD annotations (train + test)"""
    annotations = {}
    
    # Load training annotations
    train_dir = funsd_dir / "training_data" / "annotations"
    for ann_file in train_dir.glob("*.json"):
        with open(ann_file, 'r', encoding='utf-8') as f:
            annotations[ann_file.stem] = json.load(f)
    
    # Load test annotations
    test_dir = funsd_dir / "testing_data" / "annotations"
    for ann_file in test_dir.glob("*.json"):
        with open(ann_file, 'r', encoding='utf-8') as f:
            annotations[ann_file.stem] = json.load(f)
    
    return annotations


def align_words_to_entities(words, boxes, funsd_doc):
    """Match QA words to FUNSD entities and return BIO labels"""
    word_labels = [0] * len(words)  # Default "O"
    
    # For each entity in FUNSD
    for entity in funsd_doc['form']:
        entity_type = entity['label'].upper()
        entity_words = entity.get('words', [])
        
        # Track which QA words belong to this entity
        entity_word_indices = []
        
        for entity_word_info in entity_words:
            entity_text = entity_word_info['text'].strip()
            entity_box = entity_word_info['box']
            
            # Find matching word in QA words
            for qa_idx, (qa_word, qa_box) in enumerate(zip(words, boxes)):
                if qa_word.strip() == entity_text:
                    # Check box overlap (normalized coords [0, 1000])
                    box_match = all(abs(qa_box[i] - entity_box[i]) < 100 for i in range(4))
                    if box_match:
                        entity_word_indices.append(qa_idx)
                        break
        
        # Assign BIO labels
        if entity_word_indices:
            entity_word_indices.sort()
            
            for i, qa_idx in enumerate(entity_word_indices):
                if entity_type == "ANSWER":
                    word_labels[qa_idx] = LABEL2ID["B-ANSWER"] if i == 0 else LABEL2ID["I-ANSWER"]
                elif entity_type == "QUESTION":
                    word_labels[qa_idx] = LABEL2ID["B-QUESTION"] if i == 0 else LABEL2ID["I-QUESTION"]
                elif entity_type == "HEADER":
                    word_labels[qa_idx] = LABEL2ID["B-HEADER"] if i == 0 else LABEL2ID["I-HEADER"]
    
    return word_labels


def process_qa_file(qa_path: Path, funsd_annotations: dict, output_path: Path):
    """Process QA file and create BIO labels"""
    print(f"\nProcessing {qa_path.name}...")
    
    # Group QA instances by doc_id
    docs_by_id = {}
    with open(qa_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                qa_instance = json.loads(line)
                doc_id = qa_instance['doc_id']
                
                if doc_id not in docs_by_id:
                    docs_by_id[doc_id] = qa_instance
    
    print(f"Found {len(docs_by_id)} unique documents")
    
    # Create BIO labels for each document
    bio_docs = []
    missing_count = 0
    
    for doc_id, qa_instance in docs_by_id.items():
        if doc_id not in funsd_annotations:
            print(f"Warning: No FUNSD annotation for {doc_id}")
            missing_count += 1
            # Create all "O" labels
            bio_labels = [0] * len(qa_instance['words'])
        else:
            funsd_doc = funsd_annotations[doc_id]
            bio_labels = align_words_to_entities(
                qa_instance['words'],
                qa_instance['bboxes'],
                funsd_doc
            )
        
        bio_docs.append({
            "doc_id": doc_id,
            "words": qa_instance['words'],
            "boxes": qa_instance['bboxes'],
            "bio_labels": bio_labels,
            "image_path": qa_instance['image_path']
        })
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in bio_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(bio_docs)} documents to {output_path}")
    print(f"Missing annotations: {missing_count}")
    
    # Statistics
    label_counts = defaultdict(int)
    for doc in bio_docs:
        for label_id in doc['bio_labels']:
            label_counts[ID2LABEL[label_id]] += 1
    
    print(f"\nLabel distribution:")
    for label in ["O", "B-ANSWER", "I-ANSWER", "B-QUESTION", "I-QUESTION", "B-HEADER", "I-HEADER"]:
        count = label_counts.get(label, 0)
        print(f"  {label}: {count}")
    
    return bio_docs


def main():
    funsd_dir = Path("data/raw/funsd")
    qa_dir = Path("data/processed")
    output_dir = Path("data/processed")
    
    print("="*60)
    print("Generating BIO Labels for Multi-Task Learning")
    print("="*60)
    
    # Load ALL FUNSD annotations
    print("\nLoading FUNSD annotations...")
    funsd_annotations = load_funsd_annotations(funsd_dir)
    print(f"Loaded {len(funsd_annotations)} FUNSD annotations")
    
    # Process CLEAN training data
    train_docs = process_qa_file(
        qa_path=qa_dir / "funsd_layoutlmv3_train_clean.jsonl",  # Use CLEAN data
        funsd_annotations=funsd_annotations,
        output_path=output_dir / "funsd_bio_train.jsonl"
    )
    
    # Process validation data
    val_docs = process_qa_file(
        qa_path=qa_dir / "funsd_layoutlmv3_val.jsonl",
        funsd_annotations=funsd_annotations,
        output_path=output_dir / "funsd_bio_val.jsonl"
    )
    
    print("\n" + "="*60)
    print("✓ BIO label generation complete!")
    print("="*60)
    print(f"Train documents: {len(train_docs)}")
    print(f"Val documents: {len(val_docs)}")


if __name__ == "__main__":
    main()