import json
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMv3TokenizerFast


class LayoutLMv3FunsdDataset(Dataset):
    """
    Dataset for LayoutLMv3 QA.

    Expects JSONL lines from build_layoutlmv3_dataset.py:

    {
        "id": "...",
        "doc_id": "...",
        "question": "...",
        "answer_text": "...",
        "words": [...],
        "bboxes": [...],               # normalized 0–1000
        "answer_start": int,           # start word idx
        "answer_end": int              # end word idx
    }
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer_name: str = "microsoft/layoutlmv3-base",
        max_seq_length: int = 512,
    ):
        self.instances = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.instances.append(json.loads(line))

        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.instances[idx]

        question = ex["question"]
        words = ex["words"]
        bboxes = ex["bboxes"]
        start_w = ex["answer_start"]
        end_w = ex["answer_end"]

        # -----------------------------------------------------------
        # 1. Convert question -> word-level tokens (simple whitespace)
        # -----------------------------------------------------------
        q_words = question.strip().split()
        q_boxes = [[0, 0, 0, 0]] * len(q_words)   # dummy boxes for question text

        # -----------------------------------------------------------
        # 2. Merge question words + document words
        # -----------------------------------------------------------
        all_words = q_words + words
        all_boxes = q_boxes + bboxes

        # Shift answer word indices because question words come first
        shift = len(q_words)
        start_w_shifted = start_w + shift
        end_w_shifted = end_w + shift

        # -----------------------------------------------------------
        # 3. Tokenize using is_split_into_words=True
        # -----------------------------------------------------------
        encoded = self.tokenizer(
            all_words,
            boxes=all_boxes,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )


        # -----------------------------------------------------------
        # 4. Map word indices -> token indices
        # -----------------------------------------------------------
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

        # Safety fallback
        if start_pos is None or end_pos is None:
            start_pos = 0
            end_pos = 0

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "bbox": encoded["bbox"].squeeze(0).long(),
            "start_positions": torch.tensor(start_pos),
            "end_positions": torch.tensor(end_pos),
        }
