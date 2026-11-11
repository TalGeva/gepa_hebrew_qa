"""Build a single-file HTML page for reviewing test answers by hand.

One card per test question: question and level, the document (collapsed, with every quote the
prompts used highlighted in it), the dataset answer the annotators rated with their scores, and
one column per prompt with its answer, claims, quotes and judge scores. Each column has verdict
buttons; verdicts live in the browser (localStorage) and export as JSON.

Run from the project root:
    python -m scripts.build_review_page \
        seed=runs/<comparison dir>:seed final=runs/<comparison dir>:gepa_gemma ...

Each argument is label=<comparison run dir>:<program key>, where the program key names the
test_<key>.jsonl file written by compare_gepa_judges.
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path

from hebqa.claims import _normalize
from hebqa.data import load_abstractive_splits
from hebqa.data.abstractive import read_abstractive
from hebqa.judge.rubric import CRITERIA
from hebqa.runlog import RunLog

LEVELS = {0: "0 - אין תשובה במסמך", 1: "1 - פרט מפורש אחד", 2: "2 - חיבור של 2-3 פרטים"}
GENRES = {"dialogue": "דיאלוג", "encyclopedic": "אנציקלופדי", "journalistic": "עיתונאי"}
JUDGES = {"sonnet": "Sonnet", "gemma": "Gemma"}
VERDICTS = [
    ("refusal_ok", "סירוב נכון"),
    ("answer_ok", "תשובה נכונה"),
    ("hallucination", "הזיה / היברידי"),
    ("wrong_other", "שגוי אחר"),
]
_ELLIPSIS = re.compile(r"\.{3,}|…")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("columns", nargs="+", help="label=<comparison run dir>:<program key>")
    parser.add_argument("--title", default="סקירת תשובות - סט המבחן")
    return parser.parse_args()


def load_column(spec: str) -> tuple[str, dict, dict]:
    """(label, rows by item id, column metadata) for one label=<dir>:<key> argument."""
    label, _, source = spec.partition("=")
    run_dir, _, key = source.rpartition(":")
    if not (label and run_dir and key):
        raise ValueError(f"Expected label=<run dir>:<program key>, got {spec!r}")
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rows = {}
    with open(run_dir / f"test_{key}.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["id"]] = row
    instructions = meta["instructions"][key]
    return label, rows, {"source": str(run_dir), "key": key, "instructions": instructions, "judges": list(meta["judges"])}


def quote_spans(document: str, quote: str) -> list[tuple[int, int]] | None:
    """Character spans of the quote's fragments in the normalized document, or None if not found."""
    fragments = [f.strip(" \"'") for f in _ELLIPSIS.split(_normalize(quote))]
    fragments = [f for f in fragments if f]
    if not fragments:
        return None
    spans, pos = [], 0
    for fragment in fragments:
        pos = document.find(fragment, pos)
        if pos < 0:
            return None
        spans.append((pos, pos + len(fragment)))
        pos += len(fragment)
    return spans


def render_document(document: str, marks: list[tuple[int, int, str]]) -> str:
    """The document as HTML, with each (start, end, css class) span wrapped in <mark>."""
    boundaries = sorted({0, len(document), *(b for s, e, _ in marks for b in (s, e))})
    out = []
    for start, end in zip(boundaries, boundaries[1:]):
        classes = sorted({c for s, e, c in marks if s <= start and end <= e})
        text = html.escape(document[start:end])
        out.append(f'<mark class="{" ".join(classes)}">{text}</mark>' if classes else text)
    return "".join(out)


def render_claims(claims: list[dict], document: str) -> str:
    if not claims:
        return '<div class="claims empty">(אין טענות)</div>'
    parts = ['<ol class="claims">']
    for claim in claims:
        parts.append(f"<li>{html.escape(claim['text'])}")
        for quote in claim["quotes"]:
            found = quote_spans(document, quote) is not None
            cls, tag = ("found", "נמצא") if found else ("missing", "לא נמצא")
            parts.append(f'<div class="quote {cls}"><span class="tag">{tag}</span> "{html.escape(quote)}"</div>')
        if not claim["quotes"]:
            parts.append('<div class="quote missing"><span class="tag">אין ציטוט</span></div>')
        parts.append("</li>")
    parts.append("</ol>")
    return "".join(parts)


def render_judgments(judgments: dict) -> str:
    present = [j for j in JUDGES if j in judgments]
    if not present:
        return ""
    head = "".join(f"<th>{html.escape(CRITERIA[k])}</th>" for k in CRITERIA)
    rows = []
    for name in present:
        j = judgments[name]
        if "error" in j:
            rows.append(f'<tr><th>{JUDGES[name]}</th><td colspan="{len(CRITERIA) + 1}" class="err">שגיאת שופט</td></tr>')
            continue
        cells = []
        for k in CRITERIA:
            other = next((judgments[o]["scores"][k] for o in present if o != name and "scores" in judgments[o]), None)
            cls = "disagree" if other is not None and abs(other - j["scores"][k]) >= 2 else ""
            reason = html.escape((j.get("reasons") or {}).get(k, ""))
            cells.append(f'<td class="score {cls}" title="{reason}">{j["scores"][k]}</td>')
        rows.append(f'<tr><th>{JUDGES[name]}</th>{"".join(cells)}<td class="overall">{j["normalized"]:.2f}</td></tr>')
    reasons = []
    for name in present:
        j = judgments[name]
        if "reasons" in j:
            items = "".join(f"<li><b>{html.escape(CRITERIA[k])}</b>: {html.escape(j['reasons'][k])}</li>" for k in CRITERIA)
            reasons.append(f"<details><summary>נימוקי {JUDGES[name]}</summary><ul>{items}</ul></details>")
    return (
        f'<table class="judge"><tr><th></th>{head}<th>כללי</th></tr>{"".join(rows)}</table>'
        + "".join(reasons)
    )


def render_human(item) -> str:
    if not item.human_scores:
        return ""
    head = "".join(f"<th>{html.escape(CRITERIA[k])}</th>" for k in CRITERIA)
    rows = "".join(
        f"<tr><th>מתייגת {i}</th>" + "".join(f'<td class="score">{s.get(k, "")}</td>' for k in CRITERIA) + "</tr>"
        for i, s in enumerate(item.human_scores, 1)
    )
    return f'<table class="judge"><tr><th></th>{head}</tr>{rows}</table>'


def card(item, columns: list[tuple[str, dict, dict]]) -> str:
    document = _normalize(item.context)
    marks = []
    for ci, (label, rows, _) in enumerate(columns):
        row = rows.get(item.id)
        for claim in (row or {}).get("claims") or []:
            for quote in claim["quotes"]:
                for s, e in quote_spans(document, quote) or []:
                    marks.append((s, e, f"c{ci}"))
    for claim in item.dataset_claims:
        for quote in claim.quotes:
            for s, e in quote_spans(document, quote) or []:
                marks.append((s, e, "cref"))

    flags = set()
    cols = []
    for ci, (label, rows, _) in enumerate(columns):
        row = rows.get(item.id)
        if row is None:
            cols.append(f'<section class="col c{ci}"><h4>{html.escape(label)}</h4><p class="empty">(אין תשובה בקובץ)</p></section>')
            continue
        answer = row["answer"] or ""
        if not row["claims"]:
            flags.add("refused")
        js = row["judgments"]
        if all(k in js and "scores" in js[k] for k in JUDGES):
            if any(abs(js["sonnet"]["scores"][k] - js["gemma"]["scores"][k]) >= 2 for k in CRITERIA):
                flags.add("disagree")
        buttons = "".join(
            f'<button class="verdict" data-item="{item.id}" data-col="{html.escape(label)}" data-verdict="{code}">{name}</button>'
            for code, name in VERDICTS
        )
        cols.append(
            f'<section class="col c{ci}"><h4>{html.escape(label)}</h4>'
            f'<div class="answer">{html.escape(answer) or "<i>(תשובה ריקה)</i>"}</div>'
            f"{render_claims(row['claims'], document)}{render_judgments(js)}"
            f'<div class="verdicts" data-item="{item.id}" data-col="{html.escape(label)}">{buttons}'
            f'<input class="note" placeholder="הערה" data-item="{item.id}" data-col="{html.escape(label)}"></div>'
            "</section>"
        )

    ref_claims = render_claims([c.model_dump() for c in item.dataset_claims], document)
    return (
        f'<article class="card" data-id="{item.id}" data-level="{item.difficulty}" data-genre="{item.genre}" '
        f'data-flags="{" ".join(sorted(flags))}">'
        f'<header><span class="level l{item.difficulty}">{LEVELS[item.difficulty]}</span>'
        f'<span class="genre">{GENRES.get(item.genre, item.genre)}</span><span class="id">{item.id}</span></header>'
        f"<h3>{html.escape(item.question)}</h3>"
        f'<details class="doc"><summary>המסמך ({len(document)} תווים)</summary><div class="doctext">{render_document(document, marks)}</div></details>'
        f'<details class="ref"><summary>התשובה שדורגה במאגר, עם ציוני המתייגות</summary>'
        f'<div class="answer">{html.escape(item.dataset_answer)}</div>{ref_claims}{render_human(item)}</details>'
        f'<div class="cols">{"".join(cols)}</div>'
        "</article>"
    )


CSS = """
body{font-family:system-ui,Arial,sans-serif;margin:0;background:#f4f4f6;color:#222;direction:rtl}
header.top{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccc;padding:8px 16px;z-index:5;display:flex;gap:16px;flex-wrap:wrap;align-items:center}
header.top label{margin-inline-end:8px}
main{padding:12px 16px;max-width:1700px;margin:auto}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin-bottom:16px}
.card header{display:flex;gap:12px;font-size:0.9em;color:#555}
.level{font-weight:bold;padding:1px 6px;border-radius:4px}
.l0{background:#ffe3e3}.l1{background:#e3f0ff}.l2{background:#e8ffe3}
.id{font-family:monospace;direction:ltr}
h3{margin:6px 0 10px}
details.doc>summary,details.ref>summary{cursor:pointer;color:#345;font-weight:bold}
.doctext{white-space:pre-wrap;line-height:1.6;background:#fafaf5;padding:10px;border:1px solid #eee;max-height:60vh;overflow:auto}
mark{background:transparent;border-bottom:3px solid}
mark.c0{border-color:#888;background:#8882}mark.c1{border-color:#2a7;background:#2a72}mark.c2{border-color:#27c;background:#27c2}mark.c3{border-color:#c72;background:#c722}mark.cref{background:#fff3b0}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px;margin-top:10px}
.col{border:1px solid #e3e3e3;border-radius:6px;padding:8px;border-top-width:4px}
.col.c0{border-top-color:#888}.col.c1{border-top-color:#2a7}.col.c2{border-top-color:#27c}.col.c3{border-top-color:#c72}
.col h4{margin:0 0 6px}
.answer{background:#f7f7fb;padding:8px;border-radius:4px;white-space:pre-wrap}
ol.claims{padding-inline-start:20px;margin:8px 0}
.claims.empty{color:#888;margin:8px 0}
.quote{font-size:0.9em;margin:2px 0;padding-inline-start:8px;border-inline-start:3px solid}
.quote.found{border-color:#2a7;color:#255}.quote.missing{border-color:#d33;color:#a22}
.quote .tag{font-size:0.8em;background:#eee;padding:0 4px;border-radius:3px;margin-inline-end:4px}
table.judge{border-collapse:collapse;margin:6px 0;font-size:0.9em}
table.judge th,table.judge td{border:1px solid #ddd;padding:2px 6px;text-align:center}
td.score{cursor:help}td.disagree{background:#ffd9a8}td.overall{font-weight:bold}td.err{color:#a22}
details summary{cursor:pointer;font-size:0.9em;color:#456}
.verdicts{margin-top:8px;display:flex;flex-wrap:wrap;gap:4px;align-items:center}
button.verdict{border:1px solid #bbb;background:#fff;border-radius:4px;padding:3px 8px;cursor:pointer}
button.verdict.on{background:#345;color:#fff;border-color:#345}
input.note{flex:1;min-width:120px;border:1px solid #ccc;border-radius:4px;padding:3px 6px}
.hidden{display:none}
#prompts{display:none}#prompts.show{display:block}
.prompt{white-space:pre-wrap;background:#fff;border:1px solid #ddd;padding:12px;border-radius:6px;margin-bottom:12px}
.count{color:#666}
button.tool{padding:4px 10px}
"""

JS = """
const KEY='hebqa_review_verdicts';
const state=JSON.parse(localStorage.getItem(KEY)||'{}');
function save(){localStorage.setItem(KEY,JSON.stringify(state));refresh();}
function k(el){return el.dataset.item+'|'+el.dataset.col;}
document.querySelectorAll('button.verdict').forEach(b=>{
  b.addEventListener('click',()=>{const s=state[k(b)]||{};s.verdict=(s.verdict===b.dataset.verdict)?null:b.dataset.verdict;state[k(b)]=s;save();});
});
document.querySelectorAll('input.note').forEach(i=>{
  i.addEventListener('change',()=>{const s=state[k(i)]||{};s.note=i.value;state[k(i)]=s;save();});
});
function refresh(){
  document.querySelectorAll('button.verdict').forEach(b=>{const s=state[k(b)];b.classList.toggle('on',!!s&&s.verdict===b.dataset.verdict);});
  document.querySelectorAll('input.note').forEach(i=>{const s=state[k(i)];if(s&&s.note!==undefined&&document.activeElement!==i)i.value=s.note;});
  document.querySelectorAll('.card').forEach(c=>{
    const done=[...c.querySelectorAll('.verdicts')].every(v=>{const s=state[v.dataset.item+'|'+v.dataset.col];return s&&s.verdict;});
    c.dataset.reviewed=done?'1':'0';
  });
  filter();
}
function filter(){
  const lv=document.querySelector('#level').value, g=document.querySelector('#genre').value, f=document.querySelector('#flag').value, r=document.querySelector('#reviewed').value;
  let n=0;
  document.querySelectorAll('.card').forEach(c=>{
    let show=(lv===''||c.dataset.level===lv)&&(g===''||c.dataset.genre===g)&&(f===''||c.dataset.flags.split(' ').includes(f))&&(r===''||c.dataset.reviewed===r);
    c.classList.toggle('hidden',!show); if(show)n++;
  });
  document.querySelector('#count').textContent=n+' שאלות מוצגות, '+Object.values(state).filter(s=>s.verdict).length+' פסקי דין נשמרו';
}
document.querySelectorAll('#level,#genre,#flag,#reviewed').forEach(s=>s.addEventListener('change',filter));
document.querySelector('#export').addEventListener('click',()=>{
  const blob=new Blob([JSON.stringify(state,null,1)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='review_verdicts.json';a.click();
});
document.querySelector('#import').addEventListener('change',e=>{
  const f=e.target.files[0]; if(!f)return; f.text().then(t=>{Object.assign(state,JSON.parse(t));save();});
});
document.querySelector('#toggle-prompts').addEventListener('click',()=>{
  document.querySelector('#prompts').classList.toggle('show');document.querySelector('#cards').classList.toggle('hidden');
});
document.querySelectorAll('#open-all').forEach(b=>b.addEventListener('click',()=>document.querySelectorAll('details.doc').forEach(d=>d.open=!d.open)));
refresh();
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    columns = [load_column(spec) for spec in args.columns]
    if len(columns) > 4:
        raise ValueError("At most 4 columns (the page has 4 highlight colours)")
    test_ids = [ex.id for ex in load_abstractive_splits().test]
    items = {it.id: it for it in read_abstractive()}

    cards = "".join(card(items[i], columns) for i in test_ids)
    legend = " ".join(
        f'<span class="col c{ci}" style="padding:0 6px">{html.escape(label)}</span>' for ci, (label, _, _) in enumerate(columns)
    )
    prompts = "".join(
        f"<h3>{html.escape(label)} <small>({html.escape(meta['source'])})</small></h3>"
        + "".join(f'<div class="prompt">{html.escape(text)}</div>' for text in meta["instructions"].values())
        for label, _, meta in columns
    )
    level_opts = "".join(f'<option value="{k}">{v}</option>' for k, v in LEVELS.items())
    genre_opts = "".join(f'<option value="{k}">{v}</option>' for k, v in GENRES.items())
    page = f"""<!DOCTYPE html>
<html lang="he" dir="rtl"><head><meta charset="utf-8"><title>{html.escape(args.title)}</title><style>{CSS}</style></head>
<body>
<header class="top">
<b>{html.escape(args.title)}</b>
<label>רמה <select id="level"><option value="">הכול</option>{level_opts}</select></label>
<label>ז'אנר <select id="genre"><option value="">הכול</option>{genre_opts}</select></label>
<label>סינון <select id="flag"><option value="">הכול</option><option value="refused">יש סירוב (רשימת טענות ריקה)</option><option value="disagree">השופטים חלוקים (פער 2)</option></select></label>
<label>נסקר <select id="reviewed"><option value="">הכול</option><option value="0">עדיין לא</option><option value="1">כן</option></select></label>
<span id="count" class="count"></span>
<button class="tool" id="open-all">פתח/סגור את כל המסמכים</button>
<button class="tool" id="toggle-prompts">הפרומפטים</button>
<button class="tool" id="export">ייצוא פסקי הדין</button>
<label class="tool">ייבוא <input type="file" id="import" accept=".json"></label>
<span>מקרא: {legend} <mark class="cref" style="padding:0 6px">ציטוטי התשובה במאגר</mark></span>
</header>
<main>
<div id="prompts">{prompts}</div>
<div id="cards">{cards}</div>
</main>
<script>{JS}</script>
</body></html>
"""
    log = RunLog("review_page")
    out = log.dir / "review.html"
    out.write_text(page, encoding="utf-8")
    log.write_json(
        "run.json",
        {"stage": "abstractive/review_page", "columns": {label: {k: v for k, v in meta.items() if k != "instructions"} for label, _, meta in columns}, "n_test": len(test_ids)},
    )
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
