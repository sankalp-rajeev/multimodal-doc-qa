"""
Train Multi-Task LayoutLMv3 (QA + BIO Tagging)
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import random
import torch
import yaml
from transformers import LayoutLMv3Config

from src.models.multitask_model import LayoutLMv3ForMultiTask
from src.models.multitask_dataset import MultiTaskDataset
from src.training.multitask_trainer import MultiTaskTrainer
from transformers import TrainingArguments


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config():
    params_path = Path("configs/params.yaml")
    with params_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("multitask_training", {})


def main():
    cfg = load_config()
    
    # Hyperparameters
    model_name = cfg.get("model_name", "microsoft/layoutlmv3-base")
    batch_size = int(cfg.get("batch_size", 2))
    num_epochs = int(cfg.get("num_train_epochs", 15))
    lr = float(cfg.get("learning_rate", 0.00003))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.1))
    grad_accum_steps = int(cfg.get("grad_accum_steps", 4))
    seed = int(cfg.get("seed", 42))
    output_dir = Path(cfg.get("output_dir", "models/multitask/best_model"))
    
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\n{'='*60}")
    print("Multi-Task Training: QA + BIO Tagging")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"Grad accumulation: {grad_accum_steps}")
    print(f"Effective batch: {batch_size * grad_accum_steps}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    
    # Load datasets with CLEAN data
    print(f"\nLoading datasets...")
    train_dataset = MultiTaskDataset(
        qa_jsonl_path=Path("data/processed/funsd_layoutlmv3_train_clean.jsonl"),  # CLEAN
        bio_jsonl_path=Path("data/processed/funsd_bio_train.jsonl"),
        tokenizer_name=model_name,
        max_seq_length=512,
    )
    
    val_dataset = MultiTaskDataset(
        qa_jsonl_path=Path("data/processed/funsd_layoutlmv3_val.jsonl"),
        bio_jsonl_path=Path("data/processed/funsd_bio_val.jsonl"),
        tokenizer_name=model_name,
        max_seq_length=512,
    )
    
    # Load model config
    print(f"\nLoading model config...")
    config = LayoutLMv3Config.from_pretrained(model_name)
    config.num_labels = 7  # Number of BIO labels
    
    # Initialize model
    print(f"Initializing multi-task model...")
    model = LayoutLMv3ForMultiTask.from_pretrained(
        model_name,
        config=config
    )
    
    print(f"✓ Model loaded with {model.num_parameters():,} parameters")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )
    
    # Initialize trainer
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to {output_dir}")
    trainer.save_model(str(output_dir))
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()