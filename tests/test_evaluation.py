import dspy
from dspy.utils.dummies import DummyLM

from hebqa.evaluation import evaluate
from hebqa.programs import build_extractive_qa, program_instructions
from hebqa.programs.prompts import EXTRACTIVE_QA_SEED


def _example(question, answers):
    return dspy.Example(
        id=question, source="Wikipedia", wh_word="מי", context="ועדת חמישה בראשות משה שרת.",
        question=question, answers=answers,
    ).with_inputs("context", "question")


def test_build_extractive_qa_uses_seed_instructions():
    for lang, text in EXTRACTIVE_QA_SEED.items():
        assert program_instructions(build_extractive_qa(lang)) == {"self": text}


def test_evaluate_scores_each_prediction():
    examples = [_example("מי עמד בראש הוועדה?", ["משה שרת"]), _example("כמה חברים?", ["חמישה"])]
    lm = DummyLM([{"answer": "משה שרת"}, {"answer": "שישה"}])
    with dspy.context(lm=lm):
        summary, rows = evaluate(build_extractive_qa("he"), examples, num_threads=1)

    assert [r["prediction"] for r in rows] == ["משה שרת", "שישה"]
    assert rows[0]["em"] == 1.0 and rows[1]["em"] == 0.0
    assert summary["n"] == 2 and summary["em"] == 50.0
