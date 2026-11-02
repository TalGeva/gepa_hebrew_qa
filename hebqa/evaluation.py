"""Run a program over a dataset and score every prediction with EM, F1 and TLNLS."""

import dspy

from hebqa.metrics import score_answer, tlnls_metric
from hebqa.metrics.extractive import METRICS


def evaluate(program: dspy.Module, examples: list[dspy.Example], num_threads: int = 8) -> tuple[dict, list[dict]]:
    """Return (summary, per-example rows). Summary scores are percentages, as in the HeQ paper."""
    evaluator = dspy.Evaluate(
        devset=examples,
        metric=tlnls_metric,
        num_threads=num_threads,
        display_progress=True,
        max_errors=len(examples),  # a failed call scores 0 instead of aborting the run
    )
    result = evaluator(program)

    rows = []
    for example, pred, _ in result.results:
        answer = pred.get("answer") or ""
        rows.append(
            {
                "id": example.id,
                "source": example.source,
                "wh_word": example.wh_word,
                "question": example.question,
                "gold": example.answers,
                "prediction": answer,
                **score_answer(answer, example.answers),
            }
        )

    summary = {"n": len(rows), "n_empty_predictions": sum(not r["prediction"] for r in rows)}
    for name in METRICS:
        summary[name] = round(100 * sum(r[name] for r in rows) / len(rows), 2)
    return summary, rows
