"""Hebrew textual feedback for extractive QA, read by GEPA's reflection model.

The feedback names the failure type (empty, affix-only difference, too long, too short,
not copied from the passage, wrong span) so the reflection model can turn recurring
failures into instructions.
"""

from collections.abc import Sequence

from hebqa.metrics.extractive import exact_match, tlnls
from hebqa.metrics.normalize import normalize_answer, tokenize

# One-letter Hebrew prefixes (ו, ב, כ, ל, מ, ש, ה) that attach to the following word.
HEBREW_PREFIXES = frozenset("ובכלמשה")


def _differs_by_prefix(a: str, b: str) -> bool:
    longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
    extra = longer[: len(longer) - len(shorter)]
    return longer.endswith(shorter) and 0 < len(extra) <= 3 and set(extra) <= HEBREW_PREFIXES


def _affix_only_difference(pred: str, gold: str) -> bool:
    pred_toks, gold_toks = tokenize(pred), tokenize(gold)
    return len(pred_toks) == len(gold_toks) and all(
        p == g or _differs_by_prefix(p, g) for p, g in zip(pred_toks, gold_toks)
    )


def _contains(outer: str, inner: str) -> bool:
    return f" {normalize_answer(inner)} " in f" {normalize_answer(outer)} "


def hebrew_feedback(pred: str, golds: Sequence[str], context: str) -> str:
    """Diagnose a prediction against its closest gold span."""
    golds_text = " / ".join(f"«{g}»" for g in golds)
    if not normalize_answer(pred):
        return f"לא הוחזרה תשובה. יש להחזיר קטע מתוך הטקסט שעונה על השאלה. תשובות תקינות: {golds_text}."
    if any(exact_match(pred, g) for g in golds):
        return "התשובה נכונה."

    gold = max(golds, key=lambda g: tlnls(pred, g))
    # Character-level check: a span that drops an attached prefix ("מנהלת" from "למנהלת") is still copied.
    verbatim = normalize_answer(pred) in normalize_answer(context)

    if _affix_only_difference(pred, gold):
        msg = f"התשובה כמעט נכונה: ההבדל הוא רק בתחיליות (ו/ב/כ/ל/מ/ש/ה). הצורה המצופה: «{gold}»."
    elif _contains(pred, gold):
        msg = f"התשובה ארוכה מדי: היא מכילה את התשובה הנכונה «{gold}» ועוד מילים מיותרות. יש להחזיר רק את הקטע המינימלי שעונה על השאלה."
    elif _contains(gold, pred):
        msg = f"התשובה חלקית: חסר בה חלק מהתשובה. התשובה המלאה: «{gold}»."
    elif not verbatim:
        msg = f"התשובה אינה מופיעה כלשונה בקטע. יש להעתיק קטע מהטקסט ולא לנסח מחדש. התשובה הנכונה: «{gold}»."
    else:
        msg = f"התשובה שגויה: נבחר קטע לא נכון מהטקסט. התשובה הנכונה: «{gold}»."

    if not verbatim and "אינה מופיעה כלשונה" not in msg:
        msg += " בנוסף, התשובה אינה מועתקת כלשונה מהקטע."
    if len(golds) > 1:
        msg += f" כל התשובות התקינות: {golds_text}."
    return msg
