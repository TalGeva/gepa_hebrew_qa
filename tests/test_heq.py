import json

import pytest

from hebqa.data.heq import read_heq
from hebqa.data.splits import sample

CONTEXT = "ועדת חמישה בראשות משה שרת ניסחה את הטיוטה הרביעית."


def _write_heq(tmp_path, qas, start_key="answer_start"):
    data = {
        "version": "v1.1",
        "data": [{"title": "t", "source": "Wikipedia", "paragraphs": [{"context": CONTEXT, "qas": qas}]}],
    }
    for qa in qas:
        qa["answers"] = [{"text": a["text"], start_key: a["start"]} for a in qa["answers"]]
        qa.setdefault("is_impossible", not qa["answers"])
        qa.setdefault("WH Question", "מי")
        qa.setdefault("question quality", "checked")
    path = tmp_path / "heq.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("start_key", ["answer_start", "answer.start"])
def test_reads_both_offset_key_styles_and_dedupes_spans(tmp_path, start_key):
    qas = [{"id": "q1", "question": " מי עמד בראש הוועדה? ", "answers": [
        {"text": "משה שרת", "start": 18},
        {"text": "משה שרת", "start": 18},
        {"text": "שרת", "start": 22},
    ]}]
    [item] = read_heq(_write_heq(tmp_path, qas, start_key))
    assert item.question == "מי עמד בראש הוועדה?"
    assert item.answers == ("משה שרת", "שרת")
    assert not item.is_impossible


def test_unanswerable_question_has_no_answers(tmp_path):
    qas = [{"id": "q1", "question": "מתי?", "answers": []}]
    [item] = read_heq(_write_heq(tmp_path, qas))
    assert item.is_impossible and item.answers == ()


def test_offset_mismatch_raises(tmp_path):
    qas = [{"id": "q1", "question": "מי?", "answers": [{"text": "משה שרת", "start": 0}]}]
    with pytest.raises(ValueError, match="q1"):
        read_heq(_write_heq(tmp_path, qas))


def test_example_exposes_only_context_and_question_as_inputs(tmp_path):
    qas = [{"id": "q1", "question": "מי?", "answers": [{"text": "שרת", "start": 22}]}]
    ex = read_heq(_write_heq(tmp_path, qas))[0].to_example()
    assert set(ex.inputs().keys()) == {"context", "question"}
    assert ex.answers == ["שרת"]


def test_sample_is_deterministic_and_rejects_oversized_requests():
    items = list(range(50))
    assert sample(items, 10, seed=1) == sample(items, 10, seed=1)
    assert sample(items, 10, seed=1) != sample(items, 10, seed=2)
    with pytest.raises(ValueError):
        sample(items, 51, seed=1)
