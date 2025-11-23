import sys
from pathlib import Path

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import json
import math
import random
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)
import yaml
from tqdm.auto import tqdm

from src.models.dataset import FunsdQADataset
from src.evaluation.metrics import compute_em_f1



def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config() -> Dict[str, Any]:
    params_path = Path("configs/params.yaml")
    with params_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg["training"]


def evaluate_model(
    model: BertForQuestionAnswering,
    tokenizer: BertTokenizerFast,
    jsonl_path: Path,
    device: torch.device,
    max_length: int,
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Simple evaluation loop over the validation set.
    For each example:
      - tokenize question + context
      - run model
      - take argmax start/end logits
      - decode predicted answer span
      - compare to gold answer_text using EM/F1
    """
    model.eval()

    examples: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))

    predictions: List[str] = []
    references: List[str] = []

    # We can process in small batches for speed
    for i in range(0, len(examples), batch_size):
        batch_examples = examples[i : i + batch_size]

        questions = [ex["question"] for ex in batch_examples]
        contexts = [ex["context"] for ex in batch_examples]

        encodings = tokenizer(
            questions,
            contexts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings)
            start_logits = outputs.start_logits  # [B, L]
            end_logits = outputs.end_logits      # [B, L]

        start_positions = torch.argmax(start_logits, dim=-1)  # [B]
        end_positions = torch.argmax(end_logits, dim=-1)      # [B]

        for j, ex in enumerate(batch_examples):
            start_idx = start_positions[j].item()
            end_idx = end_positions[j].item()

            # Ensure valid span
            if end_idx < start_idx:
                end_idx = start_idx

            input_ids = encodings["input_ids"][j]
            pred_tokens = input_ids[start_idx : end_idx + 1]
            pred_text = tokenizer.decode(
                pred_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            predictions.append(pred_text)
            references.append(ex["answer_text"])

    metrics = compute_em_f1(predictions, references)
    return metrics


def main():

    cfg = load_config()
    max_length = int(cfg.get("max_seq_length", 512))
    batch_size = int(cfg.get("batch_size", 4))
    num_epochs = int(cfg.get("num_train_epochs", 8))
    lr = float(cfg.get("learning_rate", 4e-5))
    weight_decay = float(cfg.get("weight_decay", 0.001))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.1))
    grad_accum_steps = int(cfg.get("grad_accum_steps", 2))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    seed = int(cfg.get("seed", 42))
    
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_path = Path("data/processed/funsd_qa_train.jsonl")
    val_path = Path("data/processed/funsd_qa_val.jsonl")
    model_dir = Path("models/bert_baseline")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Dataset + DataLoader
    train_dataset = FunsdQADataset(
        jsonl_path=str(train_path),
        tokenizer_name="bert-base-uncased",
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Model + tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    model.to(device)

    # Optimizer + Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    num_training_steps = num_epochs * math.ceil(len(train_dataset) / batch_size)
    num_warmup_steps = int(warmup_ratio * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_val_f1 = 0.0

    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        model.train()
        epoch_loss = 0.0

        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                start_positions=batch["start_positions"],
                end_positions=batch["end_positions"],
            )

            # store original loss value for logging
            loss_value = outputs.loss.item()

            # normalize for gradient accumulation
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss_value
            avg_step_loss = epoch_loss / (step + 1)
            progress_bar.set_postfix({"loss": f"{avg_step_loss:.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} - avg train loss: {avg_train_loss:.4f}")

        # Validation
        val_metrics = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            jsonl_path=val_path,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
        )

        print(
            f"Validation - EM: {val_metrics['em'] * 100:.2f}%, "
            f"F1: {val_metrics['f1'] * 100:.2f}%"
        )

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_path = model_dir / "best_model"
            print(f"New best F1: {best_val_f1:.4f} - saving to {save_path}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    print("\nTraining complete.")
    print(f"Best validation F1: {best_val_f1 * 100:.2f}%")


if __name__ == "__main__":
    main()
