"""Hand-written seed instructions. GEPA starts from these; RQ3 compares the two languages."""

EXTRACTIVE_QA_SEED = {
    "he": (
        "ענה על השאלה על סמך הקטע בלבד. "
        "התשובה חייבת להיות קטע רציף המועתק מילה במילה מתוך הקטע, קצר ככל האפשר, "
        "ללא הסברים וללא מילים נוספות."
    ),
    "en": (
        "Answer the question using only the passage. "
        "The answer must be a contiguous span copied verbatim from the passage, as short as possible, "
        "with no explanation and no extra words."
    ),
}
