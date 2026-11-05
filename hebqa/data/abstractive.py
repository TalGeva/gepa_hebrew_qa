"""Parse the Hebrew abstractive QA set into typed records and DSPy examples.

Each record carries one model-written answer (with claims and quotes) and four annotators'
0-2 ratings of it on the Part II criteria. That answer is not a gold answer: some ratings flag
it as wrong. Its ratings are what the judge is validated against.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import dspy

from hebqa.claims import Claim
from hebqa.config import ABSTRACTIVE_FILE
from hebqa.data.heq import INPUT_FIELDS

# The refusal the dataset's answers use when the document does not answer the question.
UNANSWERABLE_REFERENCE = "המידע המבוקש אינו מופיע במסמך"

# Fixes from an annotator's review of the judge calibration examples (2026-11-05).
# Excluded records stay in the data so the document split does not change; split_by_document drops them.
EXCLUDED = {
    "673d9846": "garbled document; the annotator could not find the participant list the answer describes",
}
# Records whose stored quotes point at the wrong sentences (sentence_id resolved against another
# segmentation), so the annotators' attribution scores refer to quotes the file does not hold.
# Replaced with the quotes the annotator located, one list per claim.
QUOTE_FIXES = {
    "b1ea313e": [
        ["פרייזר סבר כי המאגיה מכוונת אל הטבע והאנימיזם על ידי טקסים מסובכים האמורים להעביר כוחות אלו לטובת האדם."],
        ["פרייזר האמין שקיים דמיון בין מאגיה ומדע מכיוון ששניהם שמים דגש על התנסויות"],
        ["פרייזר האמין שקיים דמיון בין מאגיה ומדע מכיוון ששניהם שמים דגש על התנסויות"],
    ],
}


@dataclass(frozen=True)
class AbstractiveItem:
    id: str
    document_id: str
    genre: str  # "dialogue", "encyclopedic" or "journalistic"
    difficulty: int  # question level, 0-2 (Part I guidelines); 0 = the document has no answer
    context: str
    question: str
    dataset_answer: str  # the model answer the annotators rated
    dataset_claims: tuple[Claim, ...]
    human_scores: tuple[dict[str, int], ...]  # one dict per annotator, ordered by annotator id; criterion -> 0-2

    @property
    def answerable(self) -> bool:
        return self.difficulty != 0

    def to_example(self) -> dspy.Example:
        return dspy.Example(
            id=self.id,
            document_id=self.document_id,
            genre=self.genre,
            difficulty=self.difficulty,
            context=self.context,
            question=self.question,
            answerable=self.answerable,
        ).with_inputs(*INPUT_FIELDS)


def read_abstractive(path: Path = ABSTRACTIVE_FILE) -> list[AbstractiveItem]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    items = []
    for r in records:
        answer = r["reference_answer"].strip()
        if answer.startswith(UNANSWERABLE_REFERENCE) == bool(r["claims"]):
            raise ValueError(f"Question {r['id']}: a refusal must have no claims, and an answer must have some")
        quotes = QUOTE_FIXES.get(r["id"]) or [[a["source_excerpt"] for a in c.get("attribution", [])] for c in r["claims"]]
        if len(quotes) != len(r["claims"]):
            raise ValueError(f"Question {r['id']}: quote fix has {len(quotes)} claims, record has {len(r['claims'])}")
        claims = tuple(Claim(text=c["text"], quotes=q) for c, q in zip(r["claims"], quotes))
        items.append(
            AbstractiveItem(
                id=r["id"],
                document_id=r["document_id"],
                genre=r["genre"],
                difficulty=r["difficulty"],
                context=r["source_text"],
                question=r["question"].strip(),
                dataset_answer=answer,
                dataset_claims=claims,
                human_scores=tuple(
                    dict(h["scores"]) for h in sorted(r["human_ratings"], key=lambda h: h["annotator_id"])
                ),
            )
        )
    return items
