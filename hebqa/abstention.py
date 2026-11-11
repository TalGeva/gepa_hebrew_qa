"""Programmatic checks for abstention and verbatim attribution.

The judge cannot be trusted with the abstention decision: it scores a fluent, well-quoted answer
to an unanswerable question on four other criteria, so the decision is one fifth of the score.
These checks are deterministic and free, and they report what the evaluation guide asks for:
how often the model abstains when it should, answers when it should abstain, and copies quotes
exactly.

An answer counts as an abstention only when it has no claims AND states that the information is
missing. A hybrid answer (states a fact, then says the detail is missing) carries claims, so it is
not an abstention: the guide treats it as a hallucination.
"""

import re

from hebqa.claims import quote_in_document

REFUSAL_SENTENCE = "המידע המבוקש אינו מופיע במסמך."

# Free-wording refusals the model produces instead of the prescribed sentence.
_REFUSAL = re.compile(
    r"אינו מופיע במסמך|אינם מופיעים במסמך|לא מופיע במסמך|אינו מכיל|אינו מציין|אינו מזכיר|אינו מפרט"
    r"|לא מצוין במסמך|לא ניתן לקבוע|לא ניתן לספק|לא ניתן לענות|אין במסמך|אין מידע|לא פורסם"
)
_HEDGE = re.compile(r"ככל הנראה|סביר להניח|כנראה|ייתכן|יתכן|נראה ש|אפשר להניח|ניתן להניח|לכאורה")


def classify(answer: str, claims: list) -> str:
    """One of: empty, abstain, hybrid, hedge, answer."""
    text = (answer or "").strip()
    if not text:
        return "empty"
    refuses = bool(_REFUSAL.search(text))
    if refuses and not claims:
        return "abstain"
    if refuses and claims:
        return "hybrid"
    if _HEDGE.search(text):
        return "hedge"
    return "answer"


def is_abstention(answer: str, claims: list) -> bool:
    return classify(answer, claims) == "abstain"


def abstention_report(rows: list[dict]) -> dict:
    """Counts per level of the answer kinds, plus the two rates the guide asks for.

    rows: dicts with difficulty, answer, claims (as written by evaluate_with_judges).
    """
    kinds = {}
    for r in rows:
        kinds.setdefault(r["difficulty"], {}).setdefault(classify(r["answer"], r["claims"]), 0)
        kinds[r["difficulty"]][classify(r["answer"], r["claims"])] += 1
    unanswerable = [r for r in rows if r["difficulty"] == 0]
    answerable = [r for r in rows if r["difficulty"] != 0]
    correct = sum(is_abstention(r["answer"], r["claims"]) for r in unanswerable)
    wrong = sum(is_abstention(r["answer"], r["claims"]) for r in answerable)
    return {
        "by_level": {lv: dict(sorted(k.items())) for lv, k in sorted(kinds.items())},
        "correct_abstention": {"n": correct, "of": len(unanswerable)},
        "wrong_abstention": {"n": wrong, "of": len(answerable)},
    }


def attribution_report(rows: list[dict], documents: dict[str, str]) -> dict:
    """Verbatim-quote statistics over answerable questions that produced claims.

    documents: item id -> document text.
    """
    answers = quotes = quotes_ok = all_ok = claims = claims_no_quote = 0
    for r in rows:
        if r["difficulty"] == 0 or not r["claims"]:
            continue
        answers += 1
        every = True
        for c in r["claims"]:
            claims += 1
            if not c["quotes"]:
                claims_no_quote += 1
                every = False
            for q in c["quotes"]:
                quotes += 1
                found = quote_in_document(q, documents[r["id"]])
                quotes_ok += found
                every &= found
        all_ok += every
    return {
        "answers_with_claims": answers,
        "answers_all_quotes_verbatim": all_ok,
        "quotes": {"verbatim": quotes_ok, "of": quotes},
        "claims": {"total": claims, "without_quote": claims_no_quote},
    }


def format_reports(abst: dict, attr: dict) -> str:
    lines = []
    for lv, kinds in abst["by_level"].items():
        lines.append(f"level {lv}: " + ", ".join(f"{k}={v}" for k, v in kinds.items()))
    ca, wa = abst["correct_abstention"], abst["wrong_abstention"]
    lines.append(f"correct abstention {ca['n']}/{ca['of']}   wrong abstention {wa['n']}/{wa['of']}")
    q = attr["quotes"]
    lines.append(
        f"attribution: {attr['answers_all_quotes_verbatim']}/{attr['answers_with_claims']} answers fully verbatim, "
        f"{q['verbatim']}/{q['of']} quotes verbatim, {attr['claims']['without_quote']}/{attr['claims']['total']} claims without a quote"
    )
    return "\n".join(lines)
