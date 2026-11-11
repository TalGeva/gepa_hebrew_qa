"""Evaluate the seed prompt and optimized programs on the abstractive test set and the HeQ test sample.

For every program: abstention and verbatim-attribution reports (programmatic), the gated score,
and on the abstractive test the judge's criteria. Writes test_<key>.jsonl per program in the
layout compare_gepa_judges uses, so build_review_page can render the answers.

Run from the project root:
    python -m scripts.eval_abstention runs/<gepa_abstention run> ... --task-model ... --task-api-base ... --thinking off
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import dspy

from hebqa.abstention import abstention_report, attribution_report, format_reports
from hebqa.config import JUDGE_LMS, TASK_LM
from hebqa.data.combined import load_combined_splits
from hebqa.gated_metric import score_item
from hebqa.judge import CRITERIA, Judge, evaluate_with_judges
from hebqa.lm import THINKING_MODES, describe_lm, lm_usage, make_lm
from hebqa.programs import build_abstractive_qa, program_instructions
from hebqa.runlog import RunLog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="run_gepa_abstention output directories (seed is always included)")
    parser.add_argument("--judge", choices=sorted(JUDGE_LMS), default="gemma")
    parser.add_argument("--task-model", default=TASK_LM.model)
    parser.add_argument("--task-api-base", default=None)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--thinking", choices=sorted(THINKING_MODES), default="off")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-heq", action="store_true", help="only the abstractive test")
    return parser.parse_args()


def gated_scores(rows: list[dict], examples: dict, judge_name: str) -> list[float]:
    out = []
    for r in rows:
        ex = examples[r["id"]]
        j = r["judgments"].get(judge_name, {})
        scores = j.get("scores") if "scores" in j else None
        try:
            s, _ = score_item(ex.difficulty, ex.dataset, r["answer"], r["claims"], ex.context, list(ex.answers), scores)
        except ValueError:  # judge error on an abstractive answerable item
            s = 0.0
        out.append(s)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    mode = THINKING_MODES[args.thinking]
    task_cfg = replace(TASK_LM, model=args.task_model, api_base=args.task_api_base, **({"max_tokens": args.max_tokens} | mode))
    task_lm = make_lm(task_cfg)
    dspy.configure(lm=task_lm)
    judge_lm = make_lm(JUDGE_LMS[args.judge])
    judges = {args.judge: Judge(judge_lm)}

    splits, heq_test = load_combined_splits()
    abs_examples = {ex.id: ex for ex in splits.test}
    heq_examples = {ex.id: ex for ex in heq_test}
    documents = {**{i: ex.context for i, ex in abs_examples.items()}, **{i: ex.context for i, ex in heq_examples.items()}}

    programs = {"seed": build_abstractive_qa("he")}
    for run in args.runs:
        meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
        program = build_abstractive_qa(meta.get("seed_language", "he"))
        program.load(str(run / "program.json"))
        programs[f"gepa_{run.name[:19]}"] = program  # timestamp prefix keeps several runs apart

    log = RunLog("abstention_eval")
    results = {}
    for key, program in programs.items():
        print(f"== {key}: abstractive test (n={len(splits.test)})")
        summary, rows = evaluate_with_judges(program, splits.test, judges, num_threads=args.threads)
        log.write_jsonl(f"test_{key}.jsonl", rows)
        abst, attr = abstention_report(rows), attribution_report(rows, documents)
        gated = gated_scores(rows, abs_examples, args.judge)
        print(format_reports(abst, attr))
        print(f"gated score {sum(gated) / len(gated):.3f}; judge errors {summary[args.judge]['errors']}; "
              f"criteria {summary[args.judge]['criteria']}")
        results[key] = {"abstractive": {"abstention": abst, "attribution": attr, "gated": round(sum(gated) / len(gated), 4),
                                        "judge": summary[args.judge]}}
        if not args.skip_heq:
            print(f"== {key}: HeQ test (n={len(heq_test)})")
            preds = program.batch(heq_test, num_threads=args.threads, max_errors=len(heq_test))
            heq_rows = [{"id": ex.id, "difficulty": ex.difficulty,
                         "answer": (p.get("answer") or "").strip() if p is not None else "",
                         "claims": [c.model_dump() for c in p.get("claims") or []] if p is not None else [],
                         "judgments": {}} for ex, p in zip(heq_test, preds)]
            log.write_jsonl(f"heq_test_{key}.jsonl", heq_rows)
            abst_h, attr_h = abstention_report(heq_rows), attribution_report(heq_rows, documents)
            gated_h = gated_scores(heq_rows, heq_examples, args.judge)
            print(format_reports(abst_h, attr_h))
            print(f"gated score {sum(gated_h) / len(gated_h):.3f}")
            results[key]["heq"] = {"abstention": abst_h, "attribution": attr_h, "gated": round(sum(gated_h) / len(gated_h), 4)}
        print()

    log.write_json("run.json", {
        "stage": "abstention/eval",
        "runs": [str(r) for r in args.runs],
        "task_lm": describe_lm(task_cfg),
        "thinking": args.thinking,
        "judges": list(judges),
        "n_test": len(splits.test),
        "n_heq_test": 0 if args.skip_heq else len(heq_test),
        "instructions": {key: program_instructions(p) for key, p in programs.items()},
        "results": results,
        "usage": {"task_lm": lm_usage(task_lm), f"judge_{args.judge}": lm_usage(judge_lm)},
    })
    print(f"saved to {log.dir}")


if __name__ == "__main__":
    main()
