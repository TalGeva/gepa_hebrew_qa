from hebqa.abstention import REFUSAL_SENTENCE, abstention_report, attribution_report, classify, format_reports

DOC = "המפעל, שנפתח ב-1998 בעיר אשדוד, מייצר חלקי חילוף לרכב ומייצא לאירופה."
CLAIM = {"text": "המפעל נפתח באשדוד", "quotes": ["שנפתח ב-1998 בעיר אשדוד"]}
BAD_CLAIM = {"text": "המפעל נפתח באשדוד", "quotes": ["נפתח בשנת 1998 באשדוד"]}


def test_classify_kinds():
    assert classify(REFUSAL_SENTENCE, []) == "abstain"
    assert classify("המסמך אינו מציין את השנה.", []) == "abstain"
    assert classify("המסמך מציין 12 ראשי ערים, אך אינו מפרט סנקציות.", [CLAIM]) == "hybrid"
    assert classify("ככל הנראה ההכנסות גדלו.", [CLAIM]) == "hedge"
    assert classify("המפעל נפתח באשדוד.", [CLAIM]) == "answer"
    assert classify("   ", []) == "empty"


def test_prescribed_sentence_with_claims_is_not_an_abstention():
    assert classify(REFUSAL_SENTENCE, [CLAIM]) == "hybrid"


def test_abstention_report_counts_correct_and_wrong():
    rows = [
        {"id": "a", "difficulty": 0, "answer": REFUSAL_SENTENCE, "claims": []},
        {"id": "b", "difficulty": 0, "answer": "המפעל נפתח באשדוד.", "claims": [CLAIM]},
        {"id": "c", "difficulty": 1, "answer": REFUSAL_SENTENCE, "claims": []},
        {"id": "d", "difficulty": 2, "answer": "המפעל נפתח באשדוד.", "claims": [CLAIM]},
    ]
    rep = abstention_report(rows)
    assert rep["correct_abstention"] == {"n": 1, "of": 2}
    assert rep["wrong_abstention"] == {"n": 1, "of": 2}
    assert rep["by_level"][0] == {"abstain": 1, "answer": 1}


def test_attribution_report_checks_quotes_verbatim():
    rows = [
        {"id": "x", "difficulty": 1, "answer": "..", "claims": [CLAIM]},
        {"id": "y", "difficulty": 2, "answer": "..", "claims": [CLAIM, BAD_CLAIM]},
        {"id": "z", "difficulty": 0, "answer": "..", "claims": [BAD_CLAIM]},  # level 0 is skipped
        {"id": "w", "difficulty": 1, "answer": "..", "claims": [{"text": "t", "quotes": []}]},
    ]
    rep = attribution_report(rows, {"x": DOC, "y": DOC, "z": DOC, "w": DOC})
    assert rep["answers_with_claims"] == 3
    assert rep["answers_all_quotes_verbatim"] == 1
    assert rep["quotes"] == {"verbatim": 2, "of": 3}
    assert rep["claims"] == {"total": 4, "without_quote": 1}
    text = format_reports(abstention_report(rows), rep)
    assert "correct abstention 0/1" in text and "1/3 answers fully verbatim" in text
