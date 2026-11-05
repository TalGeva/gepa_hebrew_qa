"""LLM judge for abstractive answers: a 0-2 score and a Hebrew rationale per Part II criterion.

The judge sees no reference answer, as the human annotators did not: it scores the answer and
its quotes against the document alone.
"""

from dataclasses import dataclass
from typing import Literal

import dspy
import pydantic

from hebqa.claims import Claim, render_answer, render_claims
from hebqa.judge.rubric import (
    CLAIM_CHECK_INSTRUCTIONS,
    CRITERIA,
    JUDGE_INSTRUCTIONS,
    QUESTION_LEVELS,
    SCORE_MAX,
    SCORE_MIN,
)


class ClaimVerdict(pydantic.BaseModel):
    claim: int
    verdict: Literal["נתמכת", "נתמכת חלקית", "לא נתמכת"]
    reason: str


class CheckClaims(dspy.Signature):
    # Claim and quotes only, no document: whether a quote exists is checked mechanically,
    # so this call only judges support, and stays cheap.
    claims: str = dspy.InputField()
    verdicts: list[ClaimVerdict] = dspy.OutputField(desc="one verdict per claim, in order")


CheckClaims = CheckClaims.with_instructions(CLAIM_CHECK_INSTRUCTIONS)


def _judge_signature() -> type[dspy.Signature]:
    fields = {
        "document": (str, dspy.InputField()),
        "question": (str, dspy.InputField()),
        "question_level": (str, dspy.InputField()),
        "answer": (str, dspy.InputField(desc="the answer to score, with its claims and quotes")),
    }
    for key in CRITERIA:  # rationale before score, so the score follows from the reasoning
        fields[f"{key}_reason"] = (str, dspy.OutputField())
        fields[key] = (int, dspy.OutputField(desc=f"{SCORE_MIN}-{SCORE_MAX}"))
    return dspy.Signature(fields, JUDGE_INSTRUCTIONS)


JudgeAnswer = _judge_signature()


@dataclass(frozen=True)
class Judgment:
    scores: dict[str, int]
    reasons: dict[str, str]

    @property
    def overall(self) -> float:
        """Mean of the criterion scores, on the 0-2 scale."""
        return sum(self.scores.values()) / len(self.scores)

    @property
    def normalized(self) -> float:
        """overall mapped to 0-1, the scale GEPA maximizes."""
        return (self.overall - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)

    def feedback(self) -> str:
        """Per-criterion score and rationale in Hebrew, the text GEPA's reflection model reads."""
        return "\n".join(f"{CRITERIA[k]} {self.scores[k]}/{SCORE_MAX}: {self.reasons[k]}" for k in CRITERIA)


class Judge:
    """Scores answers with a fixed LM, independent of the task LM set by dspy.configure.

    With check_claims, a first call checks claim by claim whether the quotes support it, and
    the verdicts are shown to the scoring call. Off by default: on the dataset's rated answers
    it made Sonnet stricter on attribution and lowered agreement with the annotators
    (runs/2026-11-04_13-21-23_judge_validation).
    """

    def __init__(self, lm: dspy.LM, check_claims: bool = False):
        self.lm = lm
        self.program = dspy.Predict(JudgeAnswer)
        self.claim_checker = dspy.Predict(CheckClaims) if check_claims else None

    def check(self, claims: list[Claim]) -> list[str] | None:
        """One verdict line per claim, or None when the checker is off or there are no claims."""
        if self.claim_checker is None or not claims:
            return None
        verdicts = {v.claim: v for v in self.claim_checker(claims=render_claims(claims)).verdicts}
        lines = []
        for i in range(1, len(claims) + 1):
            v = verdicts.get(i)
            lines.append("לא נבדקה" if v is None else v.verdict if v.verdict == "נתמכת" else f"{v.verdict}: {v.reason.strip()}")
        return lines

    def __call__(self, context: str, question: str, difficulty: int, answer: str, claims: list[Claim]) -> Judgment:
        with dspy.context(lm=self.lm):
            support = self.check(claims)
            pred = self.program(
                document=context,
                question=question,
                question_level=f"{difficulty} - {QUESTION_LEVELS[difficulty]}",
                answer=render_answer(answer, claims, context, support),
            )
        scores = {}
        for key in CRITERIA:
            score = int(pred[key])
            if not SCORE_MIN <= score <= SCORE_MAX:
                raise ValueError(f"Judge returned {key}={score}, outside {SCORE_MIN}-{SCORE_MAX}")
            scores[key] = score
        return Judgment(scores=scores, reasons={k: pred[f"{k}_reason"].strip() for k in CRITERIA})
