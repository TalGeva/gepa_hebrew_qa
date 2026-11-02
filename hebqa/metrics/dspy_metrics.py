"""Metric functions with the signatures that DSPy and GEPA expect."""

import dspy

from hebqa.metrics.extractive import best_over_golds, tlnls
from hebqa.metrics.feedback import hebrew_feedback

FEEDBACK_MODES = ("hebrew", "score")


def tlnls_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    return best_over_golds(tlnls, pred.get("answer") or "", example.answers)


def make_gepa_metric(feedback: str = "hebrew"):
    """TLNLS score plus textual feedback for GEPA.

    feedback="hebrew" diagnoses the error in Hebrew; feedback="score" reports only the
    number, the control condition for RQ2.
    """
    if feedback not in FEEDBACK_MODES:
        raise ValueError(f"Unknown feedback mode {feedback!r}; expected one of {FEEDBACK_MODES}")

    def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
        answer = pred.get("answer") or ""
        score = best_over_golds(tlnls, answer, gold.answers)
        text = f"ציון TLNLS: {score:.2f}."
        if feedback == "hebrew":
            text += " " + hebrew_feedback(answer, gold.answers, gold.context)
        return dspy.Prediction(score=score, feedback=text)

    return gepa_metric
