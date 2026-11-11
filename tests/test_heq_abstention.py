import pytest

from hebqa.data.heq import HeQItem
from hebqa.data.heq_abstention import HeQAbstentionConfig, build, load_heq_abstention_splits, sample_balanced


def _item(i: int, ctx: str, impossible: bool, quality: str = "good") -> HeQItem:
    return HeQItem(id=f"q{i}", title="t", source="Wikipedia", context=ctx, question=f"שאלה {i}?",
                   answers=() if impossible else ("תשובה",), is_impossible=impossible, wh_word="מה", quality=quality)


def test_sample_balanced_is_half_unanswerable_and_at_most_one_per_kind_per_passage():
    items = []
    for p in range(10):
        items.append(_item(2 * p, f"קטע {p}", impossible=True))
        items.append(_item(2 * p + 1, f"קטע {p}", impossible=False))
    picked = sample_balanced(items, 6, seed=1, qualities=("good",))
    assert len(picked) == 6
    assert sum(it.is_impossible for it in picked) == 3
    per_passage_kind = {(it.context, it.is_impossible) for it in picked}
    assert len(per_passage_kind) == 6


def test_sample_balanced_skips_unchecked_quality_and_reports_shortage():
    items = [_item(i, f"קטע {i}", impossible=i % 2 == 0, quality="unchecked") for i in range(10)]
    with pytest.raises(ValueError, match="Only 0 unanswerable"):
        sample_balanced(items, 4, seed=0, qualities=("good",))
    assert len(sample_balanced(items, 4, seed=0, qualities=("good", "unchecked"))) == 4


def test_build_and_load_round_trip(tmp_path, monkeypatch):
    items = [_item(i, f"קטע {i}", impossible=i % 2 == 0) for i in range(40)]
    monkeypatch.setattr("hebqa.data.heq_abstention.load_heq_split", lambda split, include_unanswerable: items)
    paths = build(HeQAbstentionConfig(n_train=4, n_val=6, n_test=8), out_dir=tmp_path)
    assert sorted(paths) == ["test", "train", "val"]
    splits = load_heq_abstention_splits(tmp_path)
    assert (len(splits.train), len(splits.val), len(splits.test)) == (4, 6, 8)
    ex = splits.train[0]
    assert sorted(ex.inputs().keys()) == ["context", "question"]
    assert ex.difficulty in (0, 1) and ex.answerable == (ex.difficulty == 1)
