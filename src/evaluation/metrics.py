import re
import string
from typing import List


def normalize_answer(s: str) -> str:
    """
    Lower text and remove punctuation, articles and extra whitespace.
    Standard SQuAD-style normalization.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    if s is None:
        return ""

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = {}
    for token in gold_tokens:
        common[token] = common.get(token, 0) + 1

    num_same = 0
    for token in pred_tokens:
        if common.get(token, 0) > 0:
            num_same += 1
            common[token] -= 1

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_em_f1(predictions: List[str], references: List[str]):
    assert len(predictions) == len(references)
    total = len(predictions)
    em_sum = 0.0
    f1_sum = 0.0

    for pred, gold in zip(predictions, references):
        em_sum += exact_match(pred, gold)
        f1_sum += f1_score(pred, gold)

    return {
        "em": em_sum / total if total > 0 else 0.0,
        "f1": f1_sum / total if total > 0 else 0.0,
    }
