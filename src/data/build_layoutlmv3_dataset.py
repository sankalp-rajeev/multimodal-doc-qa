import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def normalize_bbox(box: List[int], width: int, height: int) -> List[int]:
    """
    Convert absolute pixel coords [x0,y0,x1,y1] to 0-1000 LayoutLM format.
    """
    x0, y0, x1, y1 = box
    return [
        int(1000 * (x0 / width)),
        int(1000 * (y0 / height)),
        int(1000 * (x1 / width)),
        int(1000 * (y1 / height)),
    ]


def find_token_span(
    tokens: List[str], 
    answer_text: str, 
    answer_start_char: int
) -> Optional[tuple]:
    """
    Map character-level answer span to token-level span.
    
    Returns: (start_token_idx, end_token_idx) or None if not found
    """
    # Build context from tokens with character positions
    context = " ".join(tokens)
    answer_end_char = answer_start_char + len(answer_text)
    
    # Track character position as we iterate tokens
    char_pos = 0
    start_token_idx = None
    end_token_idx = None
    
    for i, token in enumerate(tokens):
        token_start = char_pos
        token_end = char_pos + len(token)
        
        # Check if token overlaps with answer start
        if start_token_idx is None and token_start <= answer_start_char < token_end:
            start_token_idx = i
        
        # Check if token overlaps with answer end
        if token_start < answer_end_char <= token_end:
            end_token_idx = i
            break
        
        # Move char position forward (token + space)
        char_pos = token_end + 1
    
    if start_token_idx is not None and end_token_idx is not None:
        return (start_token_idx, end_token_idx)
    
    return None


def build_layoutlmv3_instance(
    qa_instance: Dict[str, Any],
    layout_doc: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Merge QA instance with layout document to create LayoutLMv3 training instance.
    
    Args:
        qa_instance: from funsd_qa_train.jsonl (has question, answer_text, answer_start)
        layout_doc: from funsd_layout_docs.jsonl (has tokens, bboxes, img dimensions)
    
    Returns:
        Dict ready for LayoutLMv3 or None if alignment fails
    """
    tokens = layout_doc["tokens"]
    bboxes = layout_doc["bboxes"]
    img_width = layout_doc["img_width"]
    img_height = layout_doc["img_height"]
    
    question = qa_instance["question"]
    answer_text = qa_instance["answer_text"]
    answer_start_char = qa_instance["answer_start"]
    
    # Map character position to token indices
    token_span = find_token_span(tokens, answer_text, answer_start_char)
    
    if token_span is None:
        return None
    
    start_token_idx, end_token_idx = token_span
    
    # Normalize bounding boxes
    norm_bboxes = [normalize_bbox(bbox, img_width, img_height) for bbox in bboxes]
    
    # Build context string for reference
    context = " ".join(tokens)
    
    return {
        "id": qa_instance["id"],
        "doc_id": qa_instance["doc_id"],
        "question": question,
        "context": context,
        "answer_text": answer_text,
        "words": tokens,
        "bboxes": norm_bboxes,
        "answer_start": start_token_idx,
        "answer_end": end_token_idx,
        "image_path": layout_doc["image_path"],
    }


def save_jsonl(data: List[Dict[str, Any]], out_path: Path):
    """Save list of dicts to JSONL file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    # Input paths
    layout_docs_path = Path("data/processed/funsd_layout_docs.jsonl")
    qa_train_path = Path("data/processed/funsd_qa_train.jsonl")
    qa_val_path = Path("data/processed/funsd_qa_val.jsonl")
    qa_test_path = Path("data/processed/funsd_qa_test.jsonl")
    
    # Output paths
    out_train_path = Path("data/processed/funsd_layoutlmv3_train.jsonl")
    out_val_path = Path("data/processed/funsd_layoutlmv3_val.jsonl")
    out_test_path = Path("data/processed/funsd_layoutlmv3_test.jsonl")
    
    print("Loading layout documents...")
    layout_docs = load_jsonl(layout_docs_path)
    layout_by_doc = {doc["doc_id"]: doc for doc in layout_docs}
    print(f"Loaded {len(layout_docs)} layout documents")
    
    # Process each split
    for qa_path, out_path, split_name in [
        (qa_train_path, out_train_path, "train"),
        (qa_val_path, out_val_path, "val"),
        (qa_test_path, out_test_path, "test"),
    ]:
        print(f"\nProcessing {split_name} split...")
        qa_instances = load_jsonl(qa_path)
        print(f"Loaded {len(qa_instances)} QA instances")
        
        layoutlmv3_instances = []
        skipped = 0
        
        for qa_inst in qa_instances:
            doc_id = qa_inst["doc_id"]
            
            # Find matching layout document
            layout_doc = layout_by_doc.get(doc_id)
            if layout_doc is None:
                skipped += 1
                continue
            
            # Build LayoutLMv3 instance
            lmv3_inst = build_layoutlmv3_instance(qa_inst, layout_doc)
            
            if lmv3_inst is None:
                skipped += 1
                continue
            
            layoutlmv3_instances.append(lmv3_inst)
        
        print(f"Created {len(layoutlmv3_instances)} LayoutLMv3 instances")
        print(f"Skipped {skipped} instances due to alignment issues")
        
        save_jsonl(layoutlmv3_instances, out_path)
        print(f"Saved to {out_path.resolve()}")
    
    print("\n✓ All splits processed successfully!")


if __name__ == "__main__":
    main()