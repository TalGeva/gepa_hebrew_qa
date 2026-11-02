"""Stage 1, steps 3-4: optimize the seed prompt with GEPA, then score the result on the pilot test set.

Run from the project root:  python -m scripts.run_gepa [--seed-lang he] [--feedback hebrew] [--auto light]
"""

import argparse
import sys
from dataclasses import asdict, replace

import dspy

from hebqa.config import REFLECTION_LM, TASK_LM, GEPAConfig, PilotConfig
from hebqa.data import load_heq_pilot
from hebqa.evaluation import evaluate
from hebqa.lm import describe_lm, lm_usage, make_lm
from hebqa.optimize import run_gepa, summarize_gepa
from hebqa.programs import build_extractive_qa, program_instructions
from hebqa.runlog import RunLog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-lang", choices=["he", "en"], default="he")
    parser.add_argument("--feedback", choices=["hebrew", "score"], default=GEPAConfig.feedback)
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=GEPAConfig.auto)
    parser.add_argument("--task-model", default=TASK_LM.model)
    parser.add_argument("--reflection-model", default=REFLECTION_LM.model)
    parser.add_argument("--threads", type=int, default=GEPAConfig.num_threads)
    parser.add_argument("--skip-test", action="store_true", help="optimize only; do not score on the test set")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    pilot_cfg = PilotConfig()
    gepa_cfg = GEPAConfig(auto=args.auto, feedback=args.feedback, num_threads=args.threads)
    task_cfg = replace(TASK_LM, model=args.task_model)
    reflection_cfg = replace(REFLECTION_LM, model=args.reflection_model)

    splits = load_heq_pilot(pilot_cfg)
    task_lm, reflection_lm = make_lm(task_cfg), make_lm(reflection_cfg)
    dspy.configure(lm=task_lm)
    seed_program = build_extractive_qa(args.seed_lang)

    log = RunLog(f"gepa_{args.auto}_{args.feedback}_{args.seed_lang}")
    record = {
        "stage": "pilot/gepa",
        "seed_language": args.seed_lang,
        "pilot": asdict(pilot_cfg),
        "gepa": asdict(gepa_cfg),
        "task_lm": describe_lm(task_cfg),
        "reflection_lm": describe_lm(reflection_cfg),
        "seed_instructions": program_instructions(seed_program),
    }
    log.write_json("run.json", record)  # written early so a crashed run still records its config

    optimized = run_gepa(seed_program, splits.train, splits.val, reflection_lm, gepa_cfg, str(log.dir / "gepa"))
    optimized.save(str(log.dir / "program.json"))
    gepa_summary = summarize_gepa(optimized)
    log.write_json("gepa_candidates.json", gepa_summary)

    best = gepa_summary["candidates"][gepa_summary["best_idx"]]
    record.update(
        optimized_instructions=program_instructions(optimized),
        best_val_tlnls=best["val_score"],
        seed_val_tlnls=gepa_summary["candidates"][0]["val_score"],
        total_metric_calls=gepa_summary["total_metric_calls"],
        usage={"optimization": {"task_lm": lm_usage(task_lm), "reflection_lm": lm_usage(reflection_lm)}},
    )
    log.write_json("run.json", record)

    if not args.skip_test:
        task_calls_before_test = len(task_lm.history)
        test_scores, rows = evaluate(optimized, splits.test, num_threads=args.threads)
        record["test_scores"] = test_scores
        record["usage"]["test"] = {"task_lm": lm_usage(task_lm, since=task_calls_before_test)}
        log.write_jsonl("test_predictions.jsonl", rows)
        log.write_json("run.json", record)

    print("\n=== Evolved instructions ===")
    for name, text in record["optimized_instructions"].items():
        print(f"[{name}]\n{text}\n")
    print(f"val TLNLS: seed {record['seed_val_tlnls']:.3f} -> best {record['best_val_tlnls']:.3f} "
          f"({len(gepa_summary['candidates'])} candidates, {record['total_metric_calls']} metric calls)")
    if "test_scores" in record:
        s = record["test_scores"]
        print(f"test n={s['n']}  EM={s['em']}  F1={s['f1']}  TLNLS={s['tlnls']}")
    print(f"usage: {record['usage']}")
    print(f"saved to {log.dir}")


if __name__ == "__main__":
    main()
