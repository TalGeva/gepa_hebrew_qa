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

ABSTRACTIVE_QA_SEED = {
    "he": (
        "ענה על השאלה בעברית על סמך המסמך בלבד. "
        "כתוב תשובה מלאה ותמציתית במשפטים שלמים, במילים שלך, ואל תוסיף מידע שאינו מופיע במסמך. "
        "לאחר מכן פרט את הטענות שבתשובה: לכל טענה צרף ציטוט אחד או יותר, המועתקים מילה במילה מהמסמך ותומכים בה. "
        "אם המסמך אינו עונה על השאלה, כתוב: \"המידע המבוקש אינו מופיע במסמך.\" והשאר את רשימת הטענות ריקה."
    ),
    "en": (
        "Answer the question in Hebrew using only the document. "
        "Write a complete, concise answer in full sentences, in your own words, and add nothing the document does not say. "
        "Then list the claims the answer makes: back each claim with one or more quotes copied verbatim from the document that support it. "
        "If the document does not answer the question, write: \"המידע המבוקש אינו מופיע במסמך.\" and leave the claim list empty."
    ),
}
