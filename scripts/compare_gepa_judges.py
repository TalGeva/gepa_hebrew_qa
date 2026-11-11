"""Stage 3: score the seed prompt and GEPA-optimized prompts on the test set with every judge.

Each program answers the test questions once; every answer is then scored by every judge.
Scoring each prompt with the judge it was not optimized against shows whether a gain is real
or only fits one judge's taste.

Run from the project root:
    python -m scripts.compare_gepa_judges runs/<sonnet run> runs/<gemma run>
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import dspy

from hebqa.config import JUDGE_LMS, TASK_LM
from hebqa.data import load_abstractive_splits
from hebqa.judge import CRITERIA, Judge, evaluate_with_judges
from hebqa.lm import describe_lm, lm_usage, make_lm
from hebqa.programs import build_abstractive_qa, program_instructions
from hebqa.runlog import RunLog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run_gepa_abstractive output directories")
    parser.add_argument("--judges", nargs="+", choices=sorted(JUDGE_LMS), default=sorted(JUDGE_LMS))
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--task-model", default=TASK_LM.model, help="task model id (OpenRouter id, or the server's id with --task-api-base)")
    parser.add_argument("--task-api-base", default=None, help="OpenAI-compatible base URL for the task model instead of OpenRouter")
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    task_cfg = replace(TASK_LM, model=args.task_model, api_base=args.task_api_base, max_tokens=args.max_tokens)
    task_lm = make_lm(task_cfg)
    dspy.configure(lm=task_lm)
    judge_lms = {name: make_lm(JUDGE_LMS[name]) for name in args.judges}
    judges = {name: Judge(lm) for name, lm in judge_lms.items()}
    test = load_abstractive_splits().test

    seed_lang = None
    programs = {}
    for run in args.runs:
        meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
        if seed_lang not in (None, meta["seed_language"]):
            raise ValueError("All runs must start from the same seed prompt")
        seed_lang = meta["seed_language"]
        program = build_abstractive_qa(seed_lang)
        program.load(str(run / "program.json"))
        programs[f"gepa_{meta['judge']}"] = program
    programs = {"seed": build_abstractive_qa(seed_lang)} | programs

    log = RunLog("gepa_judge_comparison")
    record = {
        "stage": "abstractive/compare",
        "runs": [str(r) for r in args.runs],
        "task_lm": describe_lm(task_cfg),
        "judges": {name: describe_lm(JUDGE_LMS[name]) for name in args.judges},
        "n_test": len(test),
        "instructions": {name: program_instructions(p) for name, p in programs.items()},
        "results": {},
    }
    for name, program in programs.items():
        summary, rows = evaluate_with_judges(program, test, judges, num_threads=args.threads)
        record["results"][name] = summary
        log.write_jsonl(f"test_{name}.jsonl", rows)
        log.write_json("run.json", record)

    record["usage"] = {"task_lm": lm_usage(task_lm)} | {f"judge_{n}": lm_usage(lm) for n, lm in judge_lms.items()}
    log.write_json("run.json", record)

    for judge in args.judges:
        print(f"\n== scored by {judge} (test, n={len(test)})")
        print(f"{'program':14s} {'overall':>8s} " + " ".join(f"{k[:6]:>7s}" for k in CRITERIA) + "  by level  errors/empty")
        for name, res in record["results"].items():
            s = res[judge]
            print(f"{name:14s} {s['normalized']:8.3f} " + " ".join(f"{s['criteria'][k]:7.2f}" for k in CRITERIA)
                  + f"  {s['by_level']}  {s['errors']}/{s['empty_answers']}")
    print(f"\nusage: {record['usage']}")
    print(f"saved to {log.dir}")


if __name__ == "__main__":
    main()
