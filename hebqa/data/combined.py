"""HeQ abstention samples and the abstractive set as one training pool.

Train and validation mix both sources; the abstractive test set stays the primary held-out
evaluation and the HeQ test sample is the secondary one. Every example carries `dataset`
("heq" or "abstractive") so the metric knows which checks apply, and `answers` (HeQ gold
spans, empty for abstractive items and for every unanswerable question).
"""

import dspy

from hebqa.data.heq import INPUT_FIELDS
from hebqa.data.heq_abstention import load_heq_abstention_splits
from hebqa.data.splits import Splits, load_abstractive_splits


def tag(examples: list[dspy.Example], dataset: str) -> list[dspy.Example]:
    out = []
    for ex in examples:
        fields = ex.toDict()
        fields.setdefault("answers", [])
        out.append(dspy.Example(**fields, dataset=dataset).with_inputs(*INPUT_FIELDS))
    return out


def load_combined_splits() -> tuple[Splits, list[dspy.Example]]:
    """(Splits with mixed train and val and the abstractive test, the HeQ test sample)."""
    heq = load_heq_abstention_splits()
    abstractive = load_abstractive_splits()
    splits = Splits(
        train=tag(heq.train, "heq") + tag(abstractive.train, "abstractive"),
        val=tag(heq.val, "heq") + tag(abstractive.val, "abstractive"),
        test=tag(abstractive.test, "abstractive"),
    )
    return splits, tag(heq.test, "heq")
