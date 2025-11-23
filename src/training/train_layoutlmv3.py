import sys
from pathlib import Path
import math
import random
from typing import Dict, Any, Tuple

# Ensure project root is in Python path (same trick as baseline)
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import yaml

from transformers import (
    LayoutLMv3ForQuestionAnswering,
)

from src.models.layout_dataset import LayoutLMv3FunsdDataset


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config() -> Dict[str, Any]:
    """
    Load layout-specific training config from configs/params.yaml

    Expects a top-level key: layout_training
    """
    params_path = Path("configs/params.yaml")
    with params_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg["layout_training"]


def compute_span_metrics(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_positions: torch.Tensor,
    end_positions: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compute simple span-level EM and token-level F1 based on indices.
    This doesn't decode text but is strongly correlated with QA quality.
    """
    with torch.no_grad():
        pred_start = start_logits.argmax(dim=-1)  # [B]
        pred_end = end_logits.argmax(dim=-1)      # [B]

        batch_size = start_positions.size(0)
        em_total = 0.0
        f1_total = 0.0

        for i in range(batch_size):
            gs = start_positions[i].item()
            ge = end_positions[i].item()
            ps = pred_start[i].item()
            pe = pred_end[i].item()

            # Fix reversed spans
            if ps > pe:
                ps, pe = pe, ps

            # Exact match
            em = 1.0 if (gs == ps and ge == pe) else 0.0
            em_total += em

            # Token-level F1 over index ranges
            gold_tokens = set(range(gs, ge + 1))
            pred_tokens = set(range(ps, pe + 1))

            if len(gold_tokens) == 0 and len(pred_tokens) == 0:
                f1 = 1.0
            else:
                overlap = len(gold_tokens & pred_tokens)
                if overlap == 0:
                    f1 = 0.0
                else:
                    precision = overlap / max(len(pred_tokens), 1)
                    recall = overlap / max(len(gold_tokens), 1)
                    f1 = (2 * precision * recall) / (precision + recall)
            f1_total += f1

        em_avg = em_total / batch_size
        f1_avg = f1_total / batch_size

    return em_avg, f1_avg


def main():
    cfg = load_config()

    model_name = cfg.get("model_name", "microsoft/layoutlmv3-base")
    train_path = Path(cfg.get("train_path", "data/processed/funsd_layoutlmv3_train.jsonl"))
    val_path = Path(cfg.get("val_path", "data/processed/funsd_layoutlmv3_val.jsonl"))

    max_length = int(cfg.get("max_seq_length", 512))
    batch_size = int(cfg.get("batch_size", 2))
    num_epochs = int(cfg.get("num_train_epochs", 20))
    lr = float(cfg.get("learning_rate", 3e-5))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.1))
    grad_accum_steps = int(cfg.get("grad_accum_steps", 8))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    seed = int(cfg.get("seed", 42))
    num_workers = int(cfg.get("num_workers", 4))
    output_dir = Path(cfg.get("output_dir", "models/layoutlmv3/best_model"))

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Datasets + Dataloaders
    train_dataset = LayoutLMv3FunsdDataset(
        jsonl_path=train_path,
        tokenizer_name=model_name,
        max_seq_length=max_length,
    )
    val_dataset = LayoutLMv3FunsdDataset(
        jsonl_path=val_path,
        tokenizer_name=model_name,
        max_seq_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Model
    model = LayoutLMv3ForQuestionAnswering.from_pretrained(model_name)
    model.to(device)

    # Optimizer setup (same pattern as baseline)
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

    # Training steps (account for grad accumulation)
    num_training_steps = num_epochs * len(train_loader) // grad_accum_steps
    num_warmup_steps = int(warmup_ratio * num_training_steps)


    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_val_f1 = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        model.train()
        epoch_loss = 0.0

        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bbox = batch["bbox"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                start_positions=start_positions,
                end_positions=end_positions,
            )

            loss_value = outputs.loss.item()
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

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch} - avg train loss: {avg_train_loss:.4f}")

        # ===== Validation =====
        model.eval()
        val_loss = 0.0
        val_em = 0.0
        val_f1 = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                bbox = batch["bbox"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )

                loss = outputs.loss
                val_loss += loss.item()

                em, f1 = compute_span_metrics(
                    outputs.start_logits,
                    outputs.end_logits,
                    start_positions,
                    end_positions,
                )
                val_em += em
                val_f1 += f1
                n_batches += 1

        avg_val_loss = val_loss / max(n_batches, 1)
        avg_val_em = val_em / max(n_batches, 1)
        avg_val_f1 = val_f1 / max(n_batches, 1)

        print(
            f"Validation - loss: {avg_val_loss:.4f}, "
            f"EM(span): {avg_val_em * 100:.2f}%, "
            f"F1(span): {avg_val_f1 * 100:.2f}%"
        )

        # Save best model by span-F1
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            print(
                f"✨ New best span-F1: {best_val_f1 * 100:.2f}% "
                f"- saving model to {output_dir}"
            )
            model.save_pretrained(output_dir)
            train_dataset.tokenizer.save_pretrained(output_dir)

    print("\nTraining complete.")
    print(f"Best validation span-F1: {best_val_f1 * 100:.2f}%")


if __name__ == "__main__":
    main()
