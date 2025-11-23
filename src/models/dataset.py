from pathlib import Path
import json
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


class FunsdQADataset(Dataset):
    """
    PyTorch Dataset for FUNSD extractive QA.

    Each item returns:
      - input_ids: LongTensor [max_length]
      - attention_mask: LongTensor [max_length]
      - start_positions: LongTensor []
      - end_positions: LongTensor []
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
    ):
        self.path = Path(jsonl_path)
        self.examples = load_jsonl(self.path)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        question = ex["question"]
        context = ex["context"]
        answer_text = ex["answer_text"]
        answer_start_char = ex["answer_start"]

        # Compute character-level end index
        answer_end_char = answer_start_char + len(answer_text)

        # Tokenize question + context together.
        # We use padding="max_length" for simple batching.
        encoding = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # offsets: [1, max_length, 2]
        offsets = encoding["offset_mapping"][0]
        # sequence_ids: tells us which tokens belong to question (0) vs context (1)
        sequence_ids = encoding.sequence_ids(0)

        start_position = None
        end_position = None

        # Find token indices that best match the character-level span within the context
        for i, (offset, seq_id) in enumerate(zip(offsets.tolist(), sequence_ids)):
            if seq_id != 1:
                # Skip special tokens and question tokens; keep only context tokens
                continue

            start_char, end_char = offset

            # Set start_position only once (first token that covers the answer start)
            if (
                start_position is None
                and start_char <= answer_start_char < end_char
            ):
                start_position = i

            # Keep updating end_position until we cover the answer end,
            # then break (this gives us the last token that touches the answer span)
            if (
                start_char < answer_end_char <= end_char
            ):
                end_position = i
                break

        # Fallback: if we fail to map (rare), set to CLS token (index 0)
        if start_position is None or end_position is None:
            start_position = 0
            end_position = 0


        # Prepare final item (squeeze batch dim)
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),           # [max_length]
            "attention_mask": encoding["attention_mask"].squeeze(0), # [max_length]
            "start_positions": torch.tensor(start_position, dtype=torch.long),
            "end_positions": torch.tensor(end_position, dtype=torch.long),
        }
        return item
