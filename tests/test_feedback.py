import dspy
import pytest

from hebqa.metrics import hebrew_feedback, make_gepa_metric

CONTEXT = "ב-13 במאי הוגשה הצעתה של ועדת חמישה בראשות משה שרת למנהלת העם."


@pytest.mark.parametrize(
    "pred, golds, expected",
    [
        ("", ["משה שרת"], "לא הוחזרה תשובה"),
        ("משה שרת", ["משה שרת"], "התשובה נכונה"),
        ("למנהלת העם", ["מנהלת העם"], "רק בתחיליות"),
        ("ועדת חמישה בראשות משה שרת", ["משה שרת"], "ארוכה מדי"),
        ("שרת", ["משה שרת"], "חלקית"),
        ("דוד בן גוריון", ["משה שרת"], "אינה מופיעה כלשונה"),
        ("מנהלת העם", ["משה שרת"], "נבחר קטע לא נכון"),
    ],
)
def test_feedback_names_the_failure_type(pred, golds, expected):
    assert expected in hebrew_feedback(pred, golds, CONTEXT)


def test_feedback_lists_all_golds_when_several():
    text = hebrew_feedback("מנהלת העם", ["משה שרת", "שרת"], CONTEXT)
    assert "«משה שרת» / «שרת»" in text


def _gold():
    return dspy.Example(context=CONTEXT, question="מי?", answers=["משה שרת"]).with_inputs("context", "question")


def test_gepa_metric_returns_score_and_feedback():
    result = make_gepa_metric("hebrew")(_gold(), dspy.Prediction(answer="שרת"))
    assert 0 < result.score < 1
    assert "חלקית" in result.feedback


def test_score_only_feedback_has_no_diagnosis():
    result = make_gepa_metric("score")(_gold(), dspy.Prediction(answer="שרת"))
    assert result.feedback == f"ציון TLNLS: {result.score:.2f}."


def test_unknown_feedback_mode_raises():
    with pytest.raises(ValueError):
        make_gepa_metric("verbose")
