"""Seeded sampling of experiment splits."""

import random
from collections import defaultdict
from typing import NamedTuple, Sequence, TypeVar

import dspy

from hebqa.config import AbstractiveSplitConfig, PilotConfig
from hebqa.data.abstractive import EXCLUDED, AbstractiveItem, read_abstractive
from hebqa.data.heq import load_heq_split

T = TypeVar("T")


class Splits(NamedTuple):
    train: list[dspy.Example]
    val: list[dspy.Example]
    test: list[dspy.Example]


def sample(items: Sequence[T], n: int, seed: int) -> list[T]:
    """Draw n items without replacement; the same seed and input always give the same sample."""
    if n > len(items):
        raise ValueError(f"Requested {n} items but only {len(items)} are available")
    return random.Random(seed).sample(list(items), n)


def load_heq_pilot(cfg: PilotConfig = PilotConfig()) -> Splits:
    """Sample the pilot's train/val/test sets, each from the matching official HeQ split."""

    def take(split: str, n: int) -> list[dspy.Example]:
        items = load_heq_split(split, include_unanswerable=cfg.include_unanswerable)
        return [it.to_example() for it in sample(items, n, cfg.seed)]

    return Splits(
        train=take("train", cfg.n_train),
        val=take("val", cfg.n_val),
        test=take("test", cfg.n_test),
    )


def split_by_document(
    items: Sequence[AbstractiveItem], n_train: int, n_val: int, seed: int
) -> tuple[list[AbstractiveItem], list[AbstractiveItem], list[AbstractiveItem]]:
    """Split so all questions on one document land in the same set.

    Documents are shuffled, then dealt to train, val and test. Each set gets about its share
    of unanswerable questions; a multi-question document can push a set one or two over.
    Excluded records are dropped after dealing, so excluding one does not move the others.
    """
    by_doc: dict[str, list[AbstractiveItem]] = defaultdict(list)
    for it in items:
        by_doc[it.document_id].append(it)
    docs = sorted(by_doc)
    random.Random(seed).shuffle(docs)

    share_unanswerable = sum(not it.answerable for it in items) / len(items)
    targets = {}
    for name, n in (("train", n_train), ("val", n_val)):
        unanswerable = round(n * share_unanswerable)
        targets[name] = {False: unanswerable, True: n - unanswerable}

    sets: dict[str, list[AbstractiveItem]] = {"train": [], "val": [], "test": []}
    taken = {(name, stratum): 0 for name in targets for stratum in (False, True)}
    for doc in docs:
        group = by_doc[doc]
        stratum = all(it.answerable for it in group)  # a doc with any unanswerable question counts as unanswerable
        for name in ("train", "val"):
            if taken[name, stratum] < targets[name][stratum]:
                sets[name].extend(group)
                taken[name, stratum] += len(group)
                break
        else:
            sets["test"].extend(group)
    train, val, test = ([it for it in sets[name] if it.id not in EXCLUDED] for name in ("train", "val", "test"))
    return train, val, test


def load_abstractive_splits(cfg: AbstractiveSplitConfig = AbstractiveSplitConfig()) -> Splits:
    train, val, test = split_by_document(read_abstractive(), cfg.n_train, cfg.n_val, cfg.seed)
    return Splits(*([it.to_example() for it in s] for s in (train, val, test)))
