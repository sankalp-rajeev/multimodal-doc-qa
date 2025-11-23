import sys
import json
import random
from pathlib import Path
from typing import Tuple

import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast


# Make sure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


def load_model_and_tokenizer(
    model_dir: Path = Path("models/bert_baseline/best_model"),
):
    """
    Load fine-tuned BERT QA model and tokenizer from disk.
    """
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. "
            "Make sure you have trained and saved the baseline model."
        )

    print(f"Loading model from: {model_dir}")
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForQuestionAnswering.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Using device: {device}")
    return model, tokenizer, device


def answer_question(
    model,
    tokenizer,
    device,
    question: str,
    context: str,
    max_length: int = 512,
) -> Tuple[str, float]:
    """
    Run inference for a single (question, context) pair.
    Returns:
      predicted_answer (str), confidence_score (float)
    """
    inputs = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits  # [1, L]
        end_logits = outputs.end_logits      # [1, L]

    start_probs = torch.nn.functional.softmax(start_logits, dim=-1)
    end_probs = torch.nn.functional.softmax(end_logits, dim=-1)

    start_idx = torch.argmax(start_probs, dim=-1).item()
    end_idx = torch.argmax(end_probs, dim=-1).item()

    # Ensure valid span
    if end_idx < start_idx:
        end_idx = start_idx

    input_ids = inputs["input_ids"][0]
    pred_tokens = input_ids[start_idx : end_idx + 1]

    predicted_answer = tokenizer.decode(
        pred_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Confidence: simple heuristic = start_prob * end_prob of chosen indices
    start_conf = start_probs[0, start_idx].item()
    end_conf = end_probs[0, end_idx].item()
    confidence = float(start_conf * end_conf)

    return predicted_answer.strip(), confidence


def load_random_val_example(
    val_path: Path = Path("data/processed/funsd_qa_val.jsonl"),
):
    """
    Load a random example from the validation split.
    Returns: (context, question, answer_text, doc_id, ex_id)
    """
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    with val_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    ex = json.loads(random.choice(lines))
    return (
        ex["context"],
        ex["question"],
        ex["answer_text"],
        ex["doc_id"],
        ex["id"],
    )


def main():
    model, tokenizer, device = load_model_and_tokenizer()

    print("\n=== Quick test on a random validation example ===")
    (
        context,
        question,
        gold_answer,
        doc_id,
        ex_id,
    ) = load_random_val_example()

    print(f"Doc ID   : {doc_id}")
    print(f"Example ID: {ex_id}")
    print(f"Question : {question}")
    print(f"Gold Ans : {gold_answer}")

    # (Optional) show a shortened context
    print("\nContext snippet:")
    print(context[:400] + ("..." if len(context) > 400 else ""))

    pred_answer, conf = answer_question(
        model=model,
        tokenizer=tokenizer,
        device=device,
        question=question,
        context=context,
        max_length=512,
    )

    print("\n=== Prediction ===")
    print(f"Predicted Ans: {pred_answer}")
    print(f"Confidence   : {conf:.4f}")

    print("\nYou can now import `answer_question` in other scripts to test your own questions.")


if __name__ == "__main__":
    main()
