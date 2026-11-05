import json

import pytest

from hebqa.claims import Claim, quote_in_document
from hebqa.config import ABSTRACTIVE_FILE, AbstractiveSplitConfig
from hebqa.data import splits
from hebqa.data.abstractive import EXCLUDED, QUOTE_FIXES, UNANSWERABLE_REFERENCE, read_abstractive
from hebqa.data.splits import split_by_document

CLAIM = {"claim_id": 0, "text": "טענה", "attribution": [{"source_excerpt": "ציטוט"}]}


def _record(i, doc, answerable=True):
    return {
        "id": f"q{i}",
        "document_id": doc,
        "genre": "journalistic",
        "difficulty": 1 if answerable else 0,
        "source_text": f"מסמך {doc}",
        "question": f" שאלה {i}? ",
        "reference_answer": "תשובה." if answerable else UNANSWERABLE_REFERENCE + ".",
        "claims": [CLAIM] if answerable else [],
        "human_ratings": [
            {"annotator_id": "annotator_2", "scores": {"coherence": 1}},
            {"annotator_id": "annotator_1", "scores": {"coherence": 2}},
        ],
    }


def _write(tmp_path, records):
    path = tmp_path / "abs.json"
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


def test_reads_answer_claims_and_ratings(tmp_path):
    a, b = read_abstractive(_write(tmp_path, [_record(1, "d1"), _record(2, "d2", answerable=False)]))
    assert a.question == "שאלה 1?" and a.answerable and not b.answerable
    assert a.dataset_claims == (Claim(text="טענה", quotes=["ציטוט"]),) and b.dataset_claims == ()
    assert [h["coherence"] for h in a.human_scores] == [2, 1]  # ordered by annotator id
    assert set(a.to_example().inputs().keys()) == {"context", "question"}


def test_answerability_comes_from_the_level_not_the_answer(tmp_path):
    # The model answered a level-0 question: the question is still unanswerable.
    record = _record(1, "d1") | {"difficulty": 0}
    assert not read_abstractive(_write(tmp_path, [record]))[0].answerable


def test_refusal_with_claims_is_rejected(tmp_path):
    bad = _record(1, "d1", answerable=False) | {"claims": [CLAIM]}
    with pytest.raises(ValueError, match="q1"):
        read_abstractive(_write(tmp_path, [bad]))


@pytest.mark.skipif(not ABSTRACTIVE_FILE.exists(), reason="dataset not downloaded")
def test_every_dataset_quote_is_found_in_its_document():
    missing = [
        (it.id, q) for it in read_abstractive() for c in it.dataset_claims for q in c.quotes
        if not quote_in_document(q, it.context)
    ]
    assert missing == []


@pytest.mark.skipif(not ABSTRACTIVE_FILE.exists(), reason="dataset not downloaded")
def test_excluded_records_leave_the_split_without_moving_others(monkeypatch):
    items = read_abstractive()
    cfg = AbstractiveSplitConfig()
    before = split_by_document(items, cfg.n_train, cfg.n_val, cfg.seed)
    monkeypatch.setattr(splits, "EXCLUDED", {})
    unfiltered = split_by_document(items, cfg.n_train, cfg.n_val, cfg.seed)
    for kept, full in zip(before, unfiltered):
        assert [it.id for it in kept] == [it.id for it in full if it.id not in EXCLUDED]
    assert not EXCLUDED.keys() & {it.id for s in before for it in s}


@pytest.mark.skipif(not ABSTRACTIVE_FILE.exists(), reason="dataset not downloaded")
def test_quote_fixes_replace_the_stored_quotes():
    by_id = {it.id: it for it in read_abstractive()}
    for record_id, quotes in QUOTE_FIXES.items():
        assert [c.quotes for c in by_id[record_id].dataset_claims] == quotes


def _items(tmp_path):
    # 60 single-question docs (a third unanswerable) plus 5 docs with 3 questions each
    records = [_record(i, f"d{i}", answerable=i % 3 != 0) for i in range(60)]
    records += [_record(100 + 3 * j + k, f"m{j}") for j in range(5) for k in range(3)]
    return read_abstractive(_write(tmp_path, records))


def test_split_keeps_documents_together_and_is_deterministic(tmp_path):
    items = _items(tmp_path)
    train, val, test = split_by_document(items, n_train=20, n_val=20, seed=7)
    docs = [{it.document_id for it in s} for s in (train, val, test)]
    assert not (docs[0] & docs[1] or docs[0] & docs[2] or docs[1] & docs[2])
    assert len(train) + len(val) + len(test) == len(items)
    assert 20 <= len(train) <= 22 and 20 <= len(val) <= 22
    assert [it.id for it in train] == [it.id for it in split_by_document(items, 20, 20, seed=7)[0]]


def test_split_spreads_unanswerable_questions(tmp_path):
    items = _items(tmp_path)
    share = sum(not it.answerable for it in items) / len(items)
    for s in split_by_document(items, n_train=20, n_val=20, seed=7)[:2]:
        assert abs(sum(not it.answerable for it in s) - 20 * share) <= 1
