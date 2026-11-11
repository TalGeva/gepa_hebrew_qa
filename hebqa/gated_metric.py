"""The abstention-gated metric: the decision first, quality second.

Scores follow the evaluation guide's priorities:
- Unanswerable question (level 0): 1 for a clean abstention, 0 for anything else, including a
  hybrid answer that states a fact and then says the detail is missing.
- Answerable question answered with a refusal: 0.
- HeQ answerable question: no judge. 0.5 if a gold span appears in the answer, plus 0.5 times the
  share of quotes copied verbatim from the passage.
- Abstractive answerable question: the judge's four criteria from the guide (coherence dropped),
  weighted 0.35 factual correctness, 0.35 attribution, 0.15 completeness, 0.15 relevance, on a
  0-1 scale; an invented fact (factual correctness 0) zeroes the item.

Feedback is Hebrew text naming the failure mode, which is what the reflection model reads.
"""

import dspy

from hebqa.abstention import REFUSAL_SENTENCE, classify
from hebqa.claims import quote_in_document
from hebqa.judge.judge import Judge
from hebqa.judge.rubric import CRITERIA, SCORE_MAX
from hebqa.metrics.normalize import normalize_answer

WEIGHTS = {"faithfulness": 0.35, "attribution": 0.35, "completeness": 0.15, "relevance": 0.15}

_LEVEL0_FEEDBACK = {
    "abstain": "נכון: השאלה אינה נענית מהמסמך והמודל סירב.",
    "hybrid": "הזיה: התשובה מביאה מידע מהמסמך ואז מציינת שהפרט חסר. כשהפרט המבוקש אינו במסמך יש לכתוב רק את משפט הסירוב, בלי טענות.",
    "hedge": "ניחוש: התשובה משערת (\"ככל הנראה\", \"סביר להניח\") במקום לסרב. כשהפרט אינו במסמך יש לכתוב רק את משפט הסירוב.",
    "answer": "הזיה: המודל ענה על שאלה שהמסמך אינו עונה עליה. הפרט המבוקש אינו מופיע במסמך; התשובה הגיעה מידע חיצוני או מהסקה. יש לכתוב רק את משפט הסירוב.",
    "empty": "פלט ריק או לא בפורמט הנדרש (תשובה ורשימת טענות).",
}


def _quotes(claims) -> list[str]:
    out = []
    for c in claims or []:
        out += list(c.quotes if hasattr(c, "quotes") else c["quotes"])
    return out


def score_item(difficulty: int, dataset: str, answer: str, claims, context: str, answers: list[str],
               judge_scores: dict[str, int] | None) -> tuple[float, str]:
    """(score, feedback). judge_scores is required only for abstractive answerable answers."""
    kind = classify(answer, claims)
    if difficulty == 0:
        return (1.0 if kind == "abstain" else 0.0), _LEVEL0_FEEDBACK[kind]
    if kind == "empty":
        return 0.0, _LEVEL0_FEEDBACK["empty"]
    if kind == "abstain":
        return 0.0, f"סירוב שגוי: המסמך עונה על השאלה, אך המודל כתב שהמידע אינו מופיע. יש לענות ולצרף ציטוטים."
    if dataset == "heq":
        norm_answer = normalize_answer(answer)
        grounded = any(normalize_answer(s) and normalize_answer(s) in norm_answer for s in answers)
        quotes = _quotes(claims)
        verbatim = [quote_in_document(q, context) for q in quotes]
        attribution = sum(verbatim) / len(verbatim) if verbatim else 0.0
        parts = []
        parts.append("התשובה מכילה את הפרט הנכון מהקטע." if grounded else
                     f"התשובה אינה מכילה את הפרט שהקטע נותן ({' / '.join(answers[:2])}).")
        if not quotes:
            parts.append("אין ציטוטים.")
        elif all(verbatim):
            parts.append("כל הציטוטים מועתקים מילה במילה.")
        else:
            bad = [q for q, ok in zip(quotes, verbatim) if not ok]
            parts.append(f"{len(bad)} מתוך {len(quotes)} ציטוטים אינם מופיעים במסמך מילה במילה, למשל: \"{bad[0][:80]}\". יש להעתיק קטע רציף מהמסמך בלי לשנות אף תו.")
        return round(0.5 * grounded + 0.5 * attribution, 4), " ".join(parts)
    if judge_scores is None:
        raise ValueError("abstractive answerable answers need judge scores")
    if judge_scores["faithfulness"] == 0:
        return 0.0, "הזיה: התשובה מכילה עובדה שאינה במסמך (מהימנות 0), ולכן הציון 0."
    score = sum(WEIGHTS[k] * judge_scores[k] / SCORE_MAX for k in WEIGHTS)
    return round(score, 4), ""


def make_gated_metric(judge: Judge, errors: list[str] | None = None):
    """A GEPA metric over mixed HeQ and abstractive examples (see module docstring)."""

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction:
        answer = (pred.get("answer") or "").strip() if pred is not None else ""
        claims = list(pred.get("claims") or []) if pred is not None else []
        judge_scores = None
        judge_feedback = ""
        needs_judge = gold.difficulty != 0 and gold.dataset == "abstractive" and classify(answer, claims) not in ("abstain", "empty")
        if needs_judge:
            try:
                j = judge(gold.context, gold.question, gold.difficulty, answer, claims)
            except Exception as e:  # a judge failure must not stop the optimization
                if errors is not None:
                    errors.append(f"{gold.id}: {type(e).__name__}: {e}")
                return dspy.Prediction(score=0.0, feedback="ציון: 0.00.\nהשופט לא הצליח לדרג את התשובה.")
            judge_scores = j.scores
            judge_feedback = "\n".join(f"{CRITERIA[k]} {j.scores[k]}/{SCORE_MAX}: {j.reasons[k]}" for k in WEIGHTS)
        score, feedback = score_item(gold.difficulty, gold.dataset, answer, claims, gold.context, list(gold.answers), judge_scores)
        text = f"ציון: {score:.2f}.\n" + (feedback or "") + ("\n" + judge_feedback if judge_feedback else "")
        return dspy.Prediction(score=score, feedback=text.strip())

    return metric


__all__ = ["REFUSAL_SENTENCE", "WEIGHTS", "make_gated_metric", "score_item"]
