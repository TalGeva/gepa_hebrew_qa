"""Claims with supporting quotes: the attribution part of an abstractive answer.

The task program outputs an answer plus a list of claims, each backed by verbatim quotes from
the document (the format of the dataset's original answers). The judge reads both, rendered by
render_answer, with every quote checked mechanically against the document.
"""

import re

import pydantic


class Claim(pydantic.BaseModel):
    text: str
    quotes: list[str]


_QUOTE_CHARS = str.maketrans({"״": '"', "“": '"', "”": '"', "„": '"', "׳": "'", "‘": "'", "’": "'", "`": "'"})
_ELLIPSIS = re.compile(r"\.{3,}|…")


def _normalize(text: str) -> str:
    return " ".join(text.translate(_QUOTE_CHARS).split())


def quote_in_document(quote: str, document: str) -> bool:
    """True if the quote appears verbatim in the document, up to whitespace and quote-mark style.

    An ellipsis splits the quote into fragments that must appear in order.
    """
    doc = _normalize(document)
    fragments = [f.strip(" \"'") for f in _ELLIPSIS.split(_normalize(quote))]
    fragments = [f for f in fragments if f]
    if not fragments:
        return False
    pos = 0
    for fragment in fragments:
        pos = doc.find(fragment, pos)
        if pos < 0:
            return False
        pos += len(fragment)
    return True


def render_claims(claims: list[Claim]) -> str:
    """Numbered claims with their quotes, as the claim checker reads them."""
    lines = []
    for i, claim in enumerate(claims, 1):
        lines.append(f"{i}. {claim.text.strip()}")
        lines += [f'   ציטוט: "{q.strip()}"' for q in claim.quotes] or ["   (אין ציטוט)"]
    return "\n".join(lines)


def render_answer(answer: str, claims: list[Claim], document: str, support: list[str] | None = None) -> str:
    """The answer as the judge sees it: text, then each claim with its quotes and their checks.

    support, if given, holds one claim-checker verdict per claim.
    """
    lines = [answer.strip(), "", "טענות וציטוטים:"]
    if not claims:
        lines.append("(אין טענות)")
    for i, claim in enumerate(claims, 1):
        lines.append(f"{i}. {claim.text.strip()}")
        if not claim.quotes:
            lines.append("   (אין ציטוט)")
        for quote in claim.quotes:
            found = "נמצא במסמך" if quote_in_document(quote, document) else "לא נמצא במסמך"
            lines.append(f'   ציטוט: "{quote.strip()}" [{found}]')
        if support is not None:
            lines.append(f"   בדיקת תמיכה: {support[i - 1]}")
    return "\n".join(lines)
