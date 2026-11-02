"""Answer normalization shared by all span metrics (SQuAD-style, adapted to Hebrew)."""

import re
import string

# Hyphens and maqaf join words, so they become spaces rather than vanishing.
_SEPARATORS = re.compile(r"[-־–—/]")
# Niqqud and cantillation marks (U+0591-U+05C7), applied after the maqaf (U+05BE) is replaced.
_NIQQUD = re.compile(r"[֑-ׇ]")
# ASCII punctuation plus Hebrew geresh/gershayim and typographic quotes.
_PUNCT = str.maketrans("", "", string.punctuation + "׳״“”„‘’«»…")


def normalize_answer(text: str) -> str:
    text = _SEPARATORS.sub(" ", text)
    text = _NIQQUD.sub("", text)
    text = text.translate(_PUNCT).lower()
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    """Whitespace tokens of the normalized text, as in the HeQ paper."""
    return normalize_answer(text).split()
