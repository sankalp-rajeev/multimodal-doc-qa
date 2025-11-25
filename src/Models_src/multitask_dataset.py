"""
Multi-Task Dataset: Returns both QA labels AND BIO labels
"""

import json
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ImageProcessor
from PIL import Image


class MultiTaskDataset(Dataset):
    """
    Dataset that returns:
    - input_ids, attention_mask, bbox, pixel_values (inputs)
    - start_positions, end_positions (QA labels)
    - bio_labels (BIO tagging labels)
    """
    
    def __init__(
        self,
        qa_jsonl_path: Path,
        bio_jsonl_path: Path,
        tokenizer_name: str = "microsoft/layoutlmv3-base",
        max_seq_length: int = 512,
    ):
        self.max_seq_length = max_seq_length
        
        # Load QA instances
        self.qa_instances = []
        with open(qa_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.qa_instances.append(json.loads(line))
        
        # Load BIO instances (by doc_id for lookup)
        self.bio_by_doc = {}
        with open(bio_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    bio_doc = json.loads(line)
                    self.bio_by_doc[bio_doc['doc_id']] = bio_doc
        
        print(f"Loaded {len(self.qa_instances)} QA instances")
        print(f"Loaded {len(self.bio_by_doc)} BIO documents")
        
        # Tokenizer and image processor
        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(tokenizer_name)
        self.image_processor = LayoutLMv3ImageProcessor.from_pretrained(tokenizer_name)
        
        print("✓ Multi-task dataset ready")
    
    def __len__(self):
        return len(self.qa_instances)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        qa_instance = self.qa_instances[idx]
        
        # Get corresponding BIO document
        doc_id = qa_instance['doc_id']
        base_doc_id = doc_id.split('_')[0] if '_' in doc_id else doc_id
        bio_doc = self.bio_by_doc.get(base_doc_id)
        if bio_doc is None:
            # Fallback: try exact match
            bio_doc = self.bio_by_doc.get(doc_id)
        
        if bio_doc is None:
            raise ValueError(f"No BIO labels found for doc_id: {doc_id} (base: {base_doc_id})")
        
        # Extract data
        question = qa_instance['question']
        words = qa_instance['words']
        boxes = qa_instance['bboxes']
        answer_start = qa_instance['answer_start']  # word-level index
        answer_end = qa_instance['answer_end']
        image_path = qa_instance['image_path']
        
        # BIO labels for the document words
        bio_labels = bio_doc['bio_labels']
        
        # Prepare input: question + document
        q_words = question.strip().split()
        q_boxes = [[0, 0, 0, 0]] * len(q_words)
        q_bio_labels = [0] * len(q_words)  # Question tokens are "O" for BIO
        
        all_words = q_words + words
        all_boxes = q_boxes + boxes
        all_bio_labels = q_bio_labels + bio_labels
        
        # Shift answer positions by question length
        shift = len(q_words)
        answer_start_shifted = answer_start + shift
        answer_end_shifted = answer_end + shift
        
        # Tokenize
        encoded = self.tokenizer(
            all_words,
            boxes=all_boxes,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        
        # Map word-level labels to token-level labels
        word_ids = encoded.word_ids(batch_index=0)
        
        # Find QA span positions at token level
        start_pos = None
        end_pos = None
        for i, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            if w_id == answer_start_shifted and start_pos is None:
                start_pos = i
            if w_id == answer_end_shifted:
                end_pos = i
        
        if start_pos is None or end_pos is None:
            start_pos = 0
            end_pos = 0
        
        # Create token-level BIO labels
        token_bio_labels = []
        for i, w_id in enumerate(word_ids):
            if w_id is None:
                # Special token (CLS, SEP, PAD)
                token_bio_labels.append(-100)  # Ignore in loss
            else:
                if w_id < len(all_bio_labels):
                    token_bio_labels.append(all_bio_labels[w_id])
                else:
                    token_bio_labels.append(0)  # O label for overflow
        
        # Load and process image
        image_path_obj = Path(image_path)
        if image_path_obj.exists():
            try:
                image = Image.open(image_path_obj).convert("RGB")
                pixel_values = self.image_processor(
                    image, 
                    return_tensors="pt",
                    apply_ocr=False
                ).pixel_values
            except Exception as e:
                # Fallback to dummy image
                image = Image.new('RGB', (224, 224), color='white')
                pixel_values = self.image_processor(
                    image,
                    return_tensors="pt",
                    apply_ocr=False
                ).pixel_values
        else:
            # Dummy image
            image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.image_processor(
                image,
                return_tensors="pt",
                apply_ocr=False
            ).pixel_values
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "bbox": encoded["bbox"].squeeze(0).long(),
            "pixel_values": pixel_values.squeeze(0),
            
            # QA labels
            "start_positions": torch.tensor(start_pos, dtype=torch.long),
            "end_positions": torch.tensor(end_pos, dtype=torch.long),
            
            # BIO labels
            "labels": torch.tensor(token_bio_labels, dtype=torch.long),
        }