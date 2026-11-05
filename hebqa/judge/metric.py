"""The judge as a GEPA metric, and judge-scored evaluation of abstractive programs."""

from concurrent.futures import ThreadPoolExecutor

import dspy

from hebqa.judge.judge import Judge, Judgment
from hebqa.judge.rubric import CRITERIA

EMPTY_ANSWER_FEEDBACK = "ציון: 0.00.\nהתשובה ריקה: המודל לא החזיר תשובה, או שהפלט לא היה בפורמט הנדרש (תשובה ורשימת טענות עם ציטוטים)."


def judge_answer(judge: Judge, example: dspy.Example, pred: dspy.Prediction | None) -> Judgment | None:
    """The judge's scores for one prediction, or None when there is no answer to score."""
    answer = (pred.get("answer") or "").strip() if pred is not None else ""
    if not answer:
        return None
    return judge(example.context, example.question, example.difficulty, answer, list(pred.get("claims") or []))


def make_judge_metric(judge: Judge, errors: list[str] | None = None):
    """Normalized judge score (0-1) plus the judge's per-criterion rationales as GEPA feedback.

    A failed judge call scores 0 and is appended to errors, so a run can report how many
    scores came from failures rather than from the judge.
    """

    def judge_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
        try:
            j = judge_answer(judge, gold, pred)
        except Exception as e:  # a judge failure must not stop the optimization
            if errors is not None:
                errors.append(f"{gold.id}: {type(e).__name__}: {e}")
            return dspy.Prediction(score=0.0, feedback="ציון: 0.00.\nהשופט לא הצליח לדרג את התשובה.")
        if j is None:
            return dspy.Prediction(score=0.0, feedback=EMPTY_ANSWER_FEEDBACK)
        return dspy.Prediction(score=j.normalized, feedback=f"ציון: {j.normalized:.2f}.\n{j.feedback()}")

    return judge_metric


def evaluate_with_judges(
    program: dspy.Module, examples: list[dspy.Example], judges: dict[str, Judge], num_threads: int = 8
) -> tuple[dict, list[dict]]:
    """Answer every example once, then score each answer with every judge.

    Returns (summary per judge, per-example rows). An empty answer scores 0 on every criterion;
    a failed judge call is left out of that judge's means and counted.
    """
    preds = program.batch(examples, num_threads=num_threads, max_errors=len(examples))
    rows = [
        {
            "id": ex.id,
            "difficulty": ex.difficulty,
            "answer": (pred.get("answer") or "").strip() if pred is not None else "",
            "claims": [c.model_dump() for c in pred.get("claims") or []] if pred is not None else [],
            "judgments": {},
        }
        for ex, pred in zip(examples, preds)
    ]

    for name, judge in judges.items():

        def score(i: int) -> dict:
            try:
                j = judge_answer(judge, examples[i], preds[i])
            except Exception as e:
                return {"error": f"{type(e).__name__}: {e}"}
            if j is None:
                return {"scores": {k: 0 for k in CRITERIA}, "normalized": 0.0, "empty": True}
            return {"scores": j.scores, "reasons": j.reasons, "normalized": j.normalized}

        with ThreadPoolExecutor(num_threads) as pool:
            for row, result in zip(rows, pool.map(score, range(len(rows)))):
                row["judgments"][name] = result

    summary = {}
    for name in judges:
        ok = [r for r in rows if "error" not in r["judgments"][name]]
        js = [r["judgments"][name] for r in ok]
        by_level = {}
        for level in sorted({r["difficulty"] for r in ok}):
            level_js = [r["judgments"][name] for r in ok if r["difficulty"] == level]
            by_level[level] = round(sum(j["normalized"] for j in level_js) / len(level_js), 4)
        summary[name] = {
            "n": len(ok),
            "errors": len(rows) - len(ok),
            "empty_answers": sum(j.get("empty", False) for j in js),
            "normalized": round(sum(j["normalized"] for j in js) / len(js), 4),
            "criteria": {k: round(sum(j["scores"][k] for j in js) / len(js), 3) for k in CRITERIA},
            "by_level": by_level,
        }
    return summary, rows
