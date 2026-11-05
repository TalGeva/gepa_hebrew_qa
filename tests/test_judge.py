import dspy
import pytest
from dspy.utils.dummies import DummyLM

from hebqa.claims import Claim, quote_in_document, render_answer
from hebqa.judge import (
    CRITERIA,
    Judge,
    criterion_agreement,
    evaluate_with_judges,
    krippendorff_alpha,
    make_judge_metric,
    spearman,
)
from hebqa.programs import build_abstractive_qa, program_instructions
from hebqa.programs.prompts import ABSTRACTIVE_QA_SEED


def _reply(**scores):
    reply = {}
    for key in CRITERIA:
        reply[f"{key}_reason"] = f"נימוק {key}"
        reply[key] = str(scores.get(key, 2))
    return reply


CHECK = {"verdicts": '[{"claim": 1, "verdict": "נתמכת חלקית", "reason": "המספר אינו בציטוט"}]'}


def _judge(reply, lm_log=None):
    lm = DummyLM([CHECK, reply])
    judgment = Judge(lm, check_claims=True)("מסמך על חתולים.", "שאלה?", 1, "תשובה.", [Claim(text="טענה", quotes=["על חתולים"])])
    if lm_log is not None:
        lm_log.extend(lm.history)
    return judgment


def test_judgment_scores_overall_and_normalized():
    j = _judge(_reply(coherence=1, relevance=0))
    assert j.scores == {"coherence": 1, "relevance": 0, "completeness": 2, "faithfulness": 2, "attribution": 2}
    assert j.overall == 1.4 and j.normalized == 0.7


def test_feedback_lists_every_criterion_in_hebrew():
    text = _judge(_reply(coherence=0)).feedback()
    assert text.splitlines()[0] == "קוהרנטיות 0/2: נימוק coherence"
    assert all(name in text for name in CRITERIA.values())


def test_out_of_range_score_raises():
    with pytest.raises(ValueError, match="attribution"):
        _judge(_reply(attribution=3))


def test_judge_sees_level_checked_quotes_and_claim_verdicts_but_no_reference():
    history = []
    _judge(_reply(), history)
    check_prompt, prompt = str(history[0]["messages"]), str(history[1]["messages"])
    assert "מסמך על חתולים" not in check_prompt  # the claim check reads claims and quotes only
    assert "1 - איתור ושליפה" in prompt
    assert '"על חתולים" [נמצא במסמך]' in prompt
    assert "בדיקת תמיכה: נתמכת חלקית: המספר אינו בציטוט" in prompt
    assert "reference" not in prompt


def test_claim_check_is_off_by_default():
    lm = DummyLM([_reply()])
    Judge(lm)("מסמך", "שאלה?", 1, "תשובה.", [Claim(text="טענה", quotes=["מסמך"])])
    assert len(lm.history) == 1 and "   בדיקת תמיכה" not in str(lm.history[0]["messages"])


def test_judge_uses_its_own_lm_not_the_task_lm():
    judge = Judge(DummyLM([_reply()]))
    with dspy.context(lm=DummyLM([{"answer": "x"}])):
        assert judge("מסמך", "שאלה?", 0, "תשובה", []).overall == 2


def _example():
    return dspy.Example(id="q1", context="מסמך", question="שאלה?", difficulty=1).with_inputs("context", "question")


def test_judge_metric_returns_normalized_score_and_hebrew_feedback():
    metric = make_judge_metric(Judge(DummyLM([_reply(faithfulness=0)])))
    out = metric(_example(), dspy.Prediction(answer="תשובה", claims=[Claim(text="ט", quotes=["מסמך"])]))
    assert out.score == 0.8
    assert out.feedback.startswith("ציון: 0.80.") and "מהימנות 0/2" in out.feedback


def test_judge_metric_scores_empty_answer_and_judge_failure_zero():
    errors = []
    metric = make_judge_metric(Judge(DummyLM([_reply(attribution=3)])), errors)
    assert metric(_example(), dspy.Prediction(answer=" ", claims=[])).score == 0.0
    assert metric(_example(), dspy.Prediction(answer="תשובה", claims=[])).score == 0.0
    assert len(errors) == 1 and errors[0].startswith("q1: ValueError")


def test_evaluate_with_judges_scores_each_answer_with_every_judge():
    program = build_abstractive_qa()
    examples = [_example(), _example().copy(id="q2", difficulty=0)]
    judges = {"a": Judge(DummyLM([_reply()] * 2)), "b": Judge(DummyLM([_reply(coherence=0)] * 2))}
    with dspy.context(lm=DummyLM([{"answer": "תשובה.", "claims": "[]"}] * 2)):
        summary, rows = evaluate_with_judges(program, examples, judges, num_threads=1)
    assert summary["a"]["normalized"] == 1.0 and summary["b"]["normalized"] == 0.8
    assert summary["b"]["criteria"]["coherence"] == 0 and summary["b"]["by_level"] == {0: 0.8, 1: 0.8}
    assert set(rows[0]["judgments"]) == {"a", "b"}


def test_quote_check_tolerates_whitespace_quote_marks_and_ellipsis():
    doc = 'הוא אמר: "שלום  עולם"\nואז הלך הביתה מוקדם.'
    assert quote_in_document("אמר: ״שלום עולם״", doc)
    assert quote_in_document("הוא אמר ... הלך הביתה", doc)
    assert not quote_in_document("הלך הביתה ... הוא אמר", doc)
    assert not quote_in_document("שלום לכולם", doc)
    assert not quote_in_document("...", doc)


def test_render_marks_missing_quotes_and_empty_claims():
    text = render_answer("תשובה", [Claim(text="ט", quotes=["לא קיים"]), Claim(text="ט2", quotes=[])], "מסמך")
    assert "[לא נמצא במסמך]" in text and "(אין ציטוט)" in text
    assert "(אין טענות)" in render_answer("המידע אינו מופיע", [], "מסמך")


def test_alpha_and_spearman():
    assert krippendorff_alpha([[2, 2], [0, 0], [1, 1]]) == 1.0
    assert krippendorff_alpha([[2, 0], [0, 2]]) < 0
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_criterion_agreement_flaw_detection():
    humans = [[2, 2, 2, 2], [0, 1, 0, 1], [2, 2, 1, 2], [1, 1, 2, 2]]
    a = criterion_agreement(humans, [2, 0, 2, 2])
    assert a["n_flawed"] == 2 and a["n_flagged"] == 1
    assert a["flaw_recall"] == 0.5 and a["flaw_precision"] == 1.0


def test_build_abstractive_qa_seed_and_custom_instructions():
    for lang, text in ABSTRACTIVE_QA_SEED.items():
        assert program_instructions(build_abstractive_qa(lang)) == {"self": text}
    assert program_instructions(build_abstractive_qa(instructions="ענה.")) == {"self": "ענה."}


def test_abstractive_program_returns_claims():
    lm = DummyLM([{"answer": "תשובה.", "claims": '[{"text": "טענה", "quotes": ["ציטוט"]}]'}])
    with dspy.context(lm=lm):
        pred = build_abstractive_qa()(context="מסמך", question="שאלה?")
    assert pred.claims == [Claim(text="טענה", quotes=["ציטוט"])]
