import pytest

from hebqa.metrics import exact_match, score_answer, tlnls, token_f1
from hebqa.metrics.normalize import normalize_answer


def test_normalize_strips_punctuation_niqqud_and_splits_on_maqaf():
    assert normalize_answer('  "צה״ל"  ') == "צהל"
    assert normalize_answer("שָׁלוֹם") == "שלום"
    assert normalize_answer("תל־אביב") == "תל אביב"


def test_exact_match_ignores_quotes():
    assert exact_match('"הוד מעלתו"', "הוד מעלתו") == 1.0


def test_f1_is_zero_for_prefixed_word():
    # The motivating case from the HeQ paper: F1 gives no credit for an affixed form.
    assert token_f1("בבית", "בית") == 0.0


def test_tlnls_matches_paper_examples():
    assert tlnls("ביתינו", "בית") == pytest.approx(0.5)  # Table 3
    assert tlnls("בבית", "בית") == pytest.approx(0.75)


def test_tlnls_falls_back_to_f1_for_numbers():
    # Without the fallback "1948" vs "1921" would get 0.5.
    assert tlnls("1948", "1921") == 0.0
    # "שנת 2000" is 4/7 digits, so the gold span triggers the F1 fallback.
    assert tlnls("בשנת 2000", "שנת 2000") == pytest.approx(0.5)
    # "בשנת 2000" alone is 4/8 digits; with a non-numeric gold, TLNLS credits the prefixed word.
    assert tlnls("בשנת", "שנת") == pytest.approx(0.75)


def test_tlnls_penalizes_extra_tokens():
    assert tlnls("משה שרת", "משה שרת") == 1.0
    assert tlnls("משה שרת ראש הוועדה", "משה שרת") == pytest.approx(0.5)


def test_score_answer_takes_best_gold():
    scores = score_answer("שרת", ["משה שרת", "שרת"])
    assert scores == {"em": 1.0, "f1": 1.0, "tlnls": 1.0}


def test_empty_prediction_scores_zero_on_answerable_question():
    assert score_answer("", ["שרת"]) == {"em": 0.0, "f1": 0.0, "tlnls": 0.0}
