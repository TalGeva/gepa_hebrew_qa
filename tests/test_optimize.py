"""End-to-end GEPA run on DummyLMs: checks the wiring, not optimization quality."""

import dspy
from dspy.utils.dummies import DummyLM

from hebqa.config import GEPAConfig
from hebqa.optimize import run_gepa, summarize_gepa
from hebqa.programs import build_extractive_qa

CONTEXT = "ועדת חמישה בראשות משה שרת ניסחה את הטיוטה."


def _examples(n):
    return [
        dspy.Example(id=str(i), source="Wikipedia", wh_word="מי", context=CONTEXT,
                     question=f"שאלה {i}", answers=["משה שרת"]).with_inputs("context", "question")
        for i in range(n)
    ]


def test_run_gepa_returns_program_with_candidates(tmp_path):
    task_lm = DummyLM([{"answer": "ועדת חמישה"}] * 5000)
    reflection_lm = DummyLM([{"new_instruction": "החזר רק את שם האדם מתוך הקטע."}] * 500)

    with dspy.context(lm=task_lm):
        optimized = run_gepa(build_extractive_qa("he"), _examples(4), _examples(3), reflection_lm,
                             GEPAConfig(num_threads=1), str(tmp_path / "gepa"))

    summary = summarize_gepa(optimized)
    assert summary["candidates"][0]["parents"] == [None]
    assert summary["total_metric_calls"] > 0
    assert set(summary["candidates"][summary["best_idx"]]["instructions"]) == {"self"}
