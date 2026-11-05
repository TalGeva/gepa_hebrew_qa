"""Parse HeQ (SQuAD-style JSON) into typed records and DSPy examples."""

import json
from dataclasses import dataclass
from pathlib import Path

import dspy

from hebqa.config import HEQ_FILES

# DSPy fields the program receives; everything else on the Example is label or metadata.
INPUT_FIELDS = ("context", "question")


@dataclass(frozen=True)
class HeQItem:
    id: str
    title: str
    source: str  # "Wikipedia" or "Geektime"
    context: str
    question: str
    answers: tuple[str, ...]  # all valid gold spans; empty for unanswerable questions
    is_impossible: bool
    wh_word: str
    quality: str

    def to_example(self) -> dspy.Example:
        return dspy.Example(
            id=self.id,
            title=self.title,
            source=self.source,
            context=self.context,
            question=self.question,
            answers=list(self.answers),
            is_impossible=self.is_impossible,
            wh_word=self.wh_word,
        ).with_inputs(*INPUT_FIELDS)


def _parse_answers(qa: dict, context: str) -> tuple[str, ...]:
    """Return the distinct gold spans, checking that each one sits at its offset in the context."""
    spans: list[str] = []
    for ans in qa["answers"]:
        # train v1.1 uses "answer.start", val/test v1.1 use "answer_start".
        start = ans["answer_start"] if "answer_start" in ans else ans["answer.start"]
        text = ans["text"]
        if context[start : start + len(text)] != text:
            raise ValueError(f"Answer {text!r} not found at offset {start} (question id {qa['id']})")
        text = text.strip()
        if text and text not in spans:
            spans.append(text)
    return tuple(spans)


def read_heq(path: Path) -> list[HeQItem]:
    """Read every question in a HeQ JSON file, answerable or not."""
    with open(path, encoding="utf-8") as f:
        articles = json.load(f)["data"]

    items = []
    for article in articles:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                items.append(
                    HeQItem(
                        id=qa["id"],
                        title=article["title"],
                        source=article["source"],
                        context=context,
                        question=qa["question"].strip(),
                        answers=_parse_answers(qa, context),
                        is_impossible=qa["is_impossible"],
                        wh_word=qa["WH Question"],
                        quality=qa["question quality"],
                    )
                )
    return items


def load_heq_split(split: str, *, include_unanswerable: bool = False) -> list[HeQItem]:
    """Load one official HeQ split ("train", "val" or "test")."""
    if split not in HEQ_FILES:
        raise ValueError(f"Unknown HeQ split {split!r}; expected one of {sorted(HEQ_FILES)}")
    items = read_heq(HEQ_FILES[split])
    if not include_unanswerable:
        items = [it for it in items if not it.is_impossible]
    return items
