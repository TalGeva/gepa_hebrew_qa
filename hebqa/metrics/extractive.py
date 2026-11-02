"""EM, token F1 and TLNLS for extractive QA.

TLNLS follows Cohen et al. (2023), "HeQ: a Large and Diverse Hebrew Reading Comprehension
Benchmark", Eq. 1: each gold token is matched to its most similar predicted token by
normalized Levenshtein similarity, and the sum is divided by the longer span's token count.
When more than half of either span's characters are digits, the score falls back to F1,
since "1948" vs "1921" would otherwise get 0.5.
"""

from collections import Counter
from collections.abc import Callable, Sequence

from hebqa.metrics.normalize import normalize_answer, tokenize

SpanMetric = Callable[[str, str], float]


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_toks, gold_toks = tokenize(pred), tokenize(gold)
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    overlap = sum((Counter(pred_toks) & Counter(gold_toks)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_toks), overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def normalized_levenshtein_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def _mostly_digits(text: str) -> bool:
    chars = normalize_answer(text).replace(" ", "")
    return bool(chars) and sum(c.isdigit() for c in chars) > len(chars) / 2


def tlnls(pred: str, gold: str) -> float:
    pred_toks, gold_toks = tokenize(pred), tokenize(gold)
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    if _mostly_digits(pred) or _mostly_digits(gold):
        return token_f1(pred, gold)
    total = sum(max(normalized_levenshtein_similarity(g, p) for p in pred_toks) for g in gold_toks)
    return total / max(len(pred_toks), len(gold_toks))


def best_over_golds(metric: SpanMetric, pred: str, golds: Sequence[str]) -> float:
    """Score against the closest gold span; with no gold spans, only an empty answer is correct."""
    if not golds:
        return float(not normalize_answer(pred))
    return max(metric(pred, gold) for gold in golds)


METRICS: dict[str, SpanMetric] = {"em": exact_match, "f1": token_f1, "tlnls": tlnls}


def score_answer(pred: str, golds: Sequence[str]) -> dict[str, float]:
    return {name: best_over_golds(fn, pred, golds) for name, fn in METRICS.items()}
