"""HeQ as training data for abstention and grounding.

The abstractive set has 39 unanswerable questions, 25 of them in its test split, so an optimizer
sees about seven while learning. HeQ has thousands, each labelled answerable or not, and for the
answerable ones a gold span that sits verbatim in the passage. Both labels can be checked without
a judge. This module draws balanced samples (half unanswerable) from the three official HeQ
splits, one question per passage, and writes them to data/heq_abstention/ so a run reads a fixed
file rather than re-sampling.

Examples carry the abstractive set's field names (difficulty 0 = unanswerable, 1 = answerable),
so the same program and the same reports apply.
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import dspy

from hebqa.config import DATA_DIR
from hebqa.data.heq import INPUT_FIELDS, HeQItem, load_heq_split
from hebqa.data.splits import Splits

HEQ_ABSTENTION_DIR = DATA_DIR / "heq_abstention"
# HeQ marks most questions "unchecked"; these labels mean a human looked at the question.
CHECKED_QUALITIES = ("good", "verified", "checked", "gold")


@dataclass(frozen=True)
class HeQAbstentionConfig:
    n_train: int = 150  # from HeQ train
    n_val: int = 300  # from HeQ val
    n_test: int = 300  # from HeQ test
    seed: int = 42
    qualities: tuple[str, ...] = CHECKED_QUALITIES


def _record(item: HeQItem, heq_split: str) -> dict:
    return {
        "id": item.id,
        "heq_split": heq_split,
        "title": item.title,
        "source": item.source,
        "quality": item.quality,
        "difficulty": 0 if item.is_impossible else 1,
        "answerable": not item.is_impossible,
        "context": item.context,
        "question": item.question,
        "answers": list(item.answers),
    }


def sample_balanced(items: list[HeQItem], n: int, seed: int, qualities: tuple[str, ...]) -> list[HeQItem]:
    """n items, half unanswerable, at most one question of each kind per passage.

    Passages are shuffled with the seed; from each passage one eligible unanswerable and one
    eligible answerable question may be taken, so the two kinds can share a passage.
    """
    if n % 2:
        raise ValueError("n must be even: half unanswerable, half answerable")
    rng = random.Random(seed)
    by_context: dict[str, list[HeQItem]] = {}
    for it in items:
        by_context.setdefault(it.context, []).append(it)
    contexts = list(by_context)
    rng.shuffle(contexts)
    picked: dict[bool, list[HeQItem]] = {True: [], False: []}
    for ctx in contexts:
        for want in (True, False):
            if len(picked[want]) >= n // 2:
                continue
            pool = [it for it in by_context[ctx] if it.is_impossible == want and it.quality in qualities]
            if pool:
                picked[want].append(rng.choice(pool))
    for want in (True, False):
        if len(picked[want]) < n // 2:
            kind = "unanswerable" if want else "answerable"
            raise ValueError(f"Only {len(picked[want])} {kind} passages with quality in {qualities}; need {n // 2}")
    out = picked[True] + picked[False]
    rng.shuffle(out)
    return out


def build(cfg: HeQAbstentionConfig = HeQAbstentionConfig(), out_dir: Path = HEQ_ABSTENTION_DIR) -> dict[str, Path]:
    """Sample the three splits and write them as JSON; returns the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = {"train": ("train", cfg.n_train), "val": ("val", cfg.n_val), "test": ("test", cfg.n_test)}
    paths = {}
    for name, (heq_split, n) in plan.items():
        items = load_heq_split(heq_split, include_unanswerable=True)
        chosen = sample_balanced(items, n, cfg.seed, cfg.qualities)
        records = [_record(it, heq_split) for it in chosen]
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps({"config": asdict(cfg), "n": len(records), "items": records}, ensure_ascii=False, indent=1), encoding="utf-8")
        paths[name] = path
    return paths


def _to_example(r: dict) -> dspy.Example:
    return dspy.Example(
        id=r["id"],
        difficulty=r["difficulty"],
        answerable=r["answerable"],
        context=r["context"],
        question=r["question"],
        answers=r["answers"],
        source=r["source"],
    ).with_inputs(*INPUT_FIELDS)


def load_heq_abstention_splits(data_dir: Path = HEQ_ABSTENTION_DIR) -> Splits:
    """The splits written by build(), as DSPy examples."""
    out = {}
    for name in ("train", "val", "test"):
        path = data_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `python -m scripts.build_heq_abstention_splits` first")
        out[name] = [_to_example(r) for r in json.loads(path.read_text(encoding="utf-8"))["items"]]
    return Splits(**out)
