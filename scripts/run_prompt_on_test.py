"""Answer the test set once with a given prompt and report abstention and verbatim attribution.

No judge: the abstention decision and the quote check are programmatic (hebqa.abstention), so
this costs only task-model calls. Use it to see whether a prompt moves the two numbers the
evaluation guide cares about most before spending an optimization budget.

Run from the project root:
    python -m scripts.run_prompt_on_test --instructions-file prompt.txt --label handwritten \
        --task-model HebArabNlpProject/Hebatron --task-api-base http://localhost:8002/v1 \
        --compare runs/<comparison dir>/test_seed.jsonl
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import dspy

from hebqa.abstention import abstention_report, attribution_report, format_reports
from hebqa.config import TASK_LM
from hebqa.data import load_abstractive_splits
from hebqa.data.abstractive import read_abstractive
from hebqa.lm import THINKING_MODES, describe_lm, lm_usage, make_lm
from hebqa.programs import build_abstractive_qa, program_instructions
from hebqa.runlog import RunLog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instructions-file", type=Path, default=None, help="prompt text; default: the Hebrew seed")
    parser.add_argument("--label", default="prompt")
    parser.add_argument("--task-model", default=TASK_LM.model)
    parser.add_argument("--task-api-base", default=None)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--thinking", choices=sorted(THINKING_MODES), default="template",
                        help="reasoning mode of the task model (see hebqa.lm.THINKING_MODES)")
    parser.add_argument("--compare", type=Path, action="append", default=[],
                        help="test_*.jsonl of an earlier run to report alongside (repeatable)")
    parser.add_argument("--max-answerable", type=int, default=None,
                        help="keep every level-0 question but only the first N answerable ones (a quick abstention probe)")
    return parser.parse_args()


def rows_for(program: dspy.Module, examples: list[dspy.Example], threads: int) -> list[dict]:
    preds = program.batch(examples, num_threads=threads, max_errors=len(examples))
    return [
        {
            "id": ex.id,
            "difficulty": ex.difficulty,
            "answer": (pred.get("answer") or "").strip() if pred is not None else "",
            "claims": [c.model_dump() for c in pred.get("claims") or []] if pred is not None else [],
            "judgments": {},
        }
        for ex, pred in zip(examples, preds)
    ]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    instructions = args.instructions_file.read_text(encoding="utf-8").strip() if args.instructions_file else None
    mode = THINKING_MODES[args.thinking]  # a mode's own max_tokens (room for the reasoning trace) wins
    task_cfg = replace(TASK_LM, model=args.task_model, api_base=args.task_api_base,
                       **({"max_tokens": args.max_tokens} | mode))
    if args.thinking != "template":
        args.label = f"{args.label}_think-{args.thinking}"
    task_lm = make_lm(task_cfg)
    dspy.configure(lm=task_lm)

    test = load_abstractive_splits().test
    if args.max_answerable is not None:
        answerable = [ex for ex in test if ex.difficulty != 0][:args.max_answerable]
        test = [ex for ex in test if ex.difficulty == 0] + answerable
    documents = {it.id: it.context for it in read_abstractive()}
    program = build_abstractive_qa("he", instructions=instructions)

    rows = rows_for(program, test, args.threads)
    reports = {}
    for label, rs in [(args.label, rows)] + [(p.stem.removeprefix("test_"), [json.loads(l) for l in open(p, encoding="utf-8")]) for p in args.compare]:
        reports[label] = {"abstention": abstention_report(rs), "attribution": attribution_report(rs, documents)}
        print(f"== {label} (test, n={len(rs)})")
        print(format_reports(reports[label]["abstention"], reports[label]["attribution"]))
        print()

    log = RunLog(f"prompt_test_{args.label}")
    log.write_jsonl(f"test_{args.label}.jsonl", rows)
    log.write_json("run.json", {
        "stage": "abstractive/prompt_test",
        "label": args.label,
        "instructions": program_instructions(program),
        "task_lm": describe_lm(task_cfg),
        "thinking": args.thinking,
        "n_test": len(test),
        "compare": [str(p) for p in args.compare],
        "reports": reports,
        "usage": {"task_lm": lm_usage(task_lm)},
    })
    print(f"saved to {log.dir}")


if __name__ == "__main__":
    main()
