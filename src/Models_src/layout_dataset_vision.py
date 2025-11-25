"""
File: src/models/layout_dataset_vision.py

LayoutLMv3 Dataset WITH vision features (pixel_values)
"""

import json
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ImageProcessor
from PIL import Image


class LayoutLMv3VisionDataset(Dataset):
    """
    LayoutLMv3 Dataset with text + layout + vision.
    
    Adds pixel_values from document images.
    """
    
    def __init__(
        self,
        jsonl_path: Path,
        tokenizer_name: str = "microsoft/layoutlmv3-base",
        max_seq_length: int = 512,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.max_seq_length = max_seq_length
        
        # Load data
        self.instances = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.instances.append(json.loads(line))
        
        print(f"Loaded {len(self.instances)} examples from {jsonl_path}")
        
        # Load tokenizer and image processor
        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(tokenizer_name)
        self.image_processor = LayoutLMv3ImageProcessor.from_pretrained(tokenizer_name)
        print("✓ Loaded tokenizer + image processor (vision enabled)")
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.instances[idx]
        
        question = ex["question"]
        words = ex["words"]
        bboxes = ex["bboxes"]
        start_w = ex["answer_start"]
        end_w = ex["answer_end"]
        image_path = ex["image_path"]
        
        # Prepare question + document words
        q_words = question.strip().split()
        q_boxes = [[0, 0, 0, 0]] * len(q_words)
        
        all_words = q_words + words
        all_boxes = q_boxes + bboxes
        
        # Shift answer indices
        shift = len(q_words)
        start_w_shifted = start_w + shift
        end_w_shifted = end_w + shift
        
        # Tokenize text + boxes
        encoded = self.tokenizer(
            all_words,
            boxes=all_boxes,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        
        # Map word indices to token indices
        word_ids = encoded.word_ids(batch_index=0)
        
        start_pos = None
        end_pos = None
        
        for i, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            if w_id == start_w_shifted and start_pos is None:
                start_pos = i
            if w_id == end_w_shifted:
                end_pos = i
        
        if start_pos is None or end_pos is None:
            start_pos = 0
            end_pos = 0
        
        # Load and process image
        # Load and process image (WITHOUT OCR - we already have text)
        image_path_obj = Path(image_path)
        if image_path_obj.exists():
            try:
                image = Image.open(image_path_obj).convert("RGB")
                # Disable OCR by setting apply_ocr=False
                pixel_values = self.image_processor(
                    image, 
                    return_tensors="pt",
                    apply_ocr=False  # ADD THIS - we already have text/boxes
                ).pixel_values
            except Exception as e:
                print(f"Warning: Failed to load image {image_path}: {e}")
                # Create dummy white image
                image = Image.new('RGB', (224, 224), color='white')
                pixel_values = self.image_processor(
                    image, 
                    return_tensors="pt",
                    apply_ocr=False  # ADD THIS
                ).pixel_values
        else:
            print(f"Warning: Image not found {image_path}")
            # Create dummy white image
            image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.image_processor(
                image, 
                return_tensors="pt",
                apply_ocr=False  # ADD THIS
            ).pixel_values
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "bbox": encoded["bbox"].squeeze(0).long(),
            "pixel_values": pixel_values.squeeze(0),  # NEW: Add image
            "start_positions": torch.tensor(start_pos, dtype=torch.long),
            "end_positions": torch.tensor(end_pos, dtype=torch.long),
        }