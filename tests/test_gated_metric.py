import dspy
import pytest

from hebqa.abstention import REFUSAL_SENTENCE
from hebqa.claims import Claim
from hebqa.gated_metric import make_gated_metric, score_item

DOC = "המפעל, שנפתח ב-1998 בעיר אשדוד, מייצר חלקי חילוף לרכב ומייצא לאירופה."
GOOD = [Claim(text="המפעל נפתח באשדוד", quotes=["שנפתח ב-1998 בעיר אשדוד"])]
BAD = [Claim(text="המפעל נפתח באשדוד", quotes=["נפתח בשנת 1998 באשדוד"])]


def test_level0_gate():
    assert score_item(0, "heq", REFUSAL_SENTENCE, [], DOC, [], None)[0] == 1.0
    s, fb = score_item(0, "heq", "המפעל נפתח באשדוד, אך המסמך אינו מציין את מספר העובדים.", GOOD, DOC, [], None)
    assert s == 0.0 and "הזיה" in fb
    assert score_item(0, "abstractive", "ככל הנראה 200 עובדים.", GOOD, DOC, [], None)[0] == 0.0
    assert score_item(0, "abstractive", "", [], DOC, [], None)[0] == 0.0


def test_wrong_refusal_scores_zero():
    s, fb = score_item(1, "heq", REFUSAL_SENTENCE, [], DOC, ["אשדוד"], None)
    assert s == 0.0 and "סירוב שגוי" in fb


def test_heq_answerable_combines_gold_span_and_verbatim_quotes():
    assert score_item(1, "heq", "המפעל נפתח באשדוד.", GOOD, DOC, ["אשדוד"], None)[0] == 1.0
    assert score_item(1, "heq", "המפעל נפתח באשדוד.", BAD, DOC, ["אשדוד"], None)[0] == 0.5
    s, fb = score_item(1, "heq", "המפעל נפתח בחיפה.", GOOD, DOC, ["אשדוד"], None)
    assert s == 0.5 and "אינה מכילה" in fb
    assert score_item(1, "heq", "המפעל נפתח באשדוד.", [], DOC, ["אשדוד"], None)[0] == 0.5


def test_abstractive_answerable_uses_weighted_judge_scores_with_zeroing():
    scores = {"coherence": 2, "relevance": 2, "completeness": 2, "faithfulness": 2, "attribution": 2}
    assert score_item(2, "abstractive", "תשובה.", GOOD, DOC, [], scores)[0] == 1.0
    scores["attribution"] = 0
    assert score_item(2, "abstractive", "תשובה.", GOOD, DOC, [], scores)[0] == pytest.approx(0.65)
    scores["faithfulness"] = 0
    assert score_item(2, "abstractive", "תשובה.", GOOD, DOC, [], scores)[0] == 0.0
    with pytest.raises(ValueError):
        score_item(2, "abstractive", "תשובה.", GOOD, DOC, [], None)


class _FakeJudge:
    def __init__(self):
        self.calls = 0

    def __call__(self, context, question, difficulty, answer, claims):
        self.calls += 1
        from hebqa.judge.judge import Judgment

        return Judgment(scores={k: 2 for k in ("coherence", "relevance", "completeness", "faithfulness", "attribution")},
                        reasons={k: "טוב" for k in ("coherence", "relevance", "completeness", "faithfulness", "attribution")})


def _gold(difficulty, dataset, answers=()):
    return dspy.Example(id="x", difficulty=difficulty, dataset=dataset, context=DOC, question="?", answers=list(answers)).with_inputs("context", "question")


def test_metric_calls_judge_only_for_abstractive_answerable_answers():
    judge = _FakeJudge()
    metric = make_gated_metric(judge)
    pred = dspy.Prediction(answer="המפעל נפתח באשדוד.", claims=GOOD)
    assert metric(_gold(0, "heq"), pred).score == 0.0
    assert metric(_gold(1, "heq", ["אשדוד"]), pred).score == 1.0
    assert judge.calls == 0
    out = metric(_gold(2, "abstractive"), pred)
    assert judge.calls == 1 and out.score == 1.0 and "ציון: 1.00" in out.feedback
    assert metric(_gold(2, "abstractive"), dspy.Prediction(answer=REFUSAL_SENTENCE, claims=[])).score == 0.0
    assert judge.calls == 1
