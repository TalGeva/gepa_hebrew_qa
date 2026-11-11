"""Stage 3: optimize the abstractive seed prompt with GEPA, scored by an LLM judge.

The judge's normalized score (mean of the five Part II criteria, mapped to 0-1) is the metric,
and its Hebrew per-criterion rationales are the feedback the reflection model reads. Only the
judge differs between runs; scoring on the test set is done by scripts.compare_gepa_judges,
with every judge, so no prompt is graded only by the judge it was optimized against.

Run from the project root:  python -m scripts.run_gepa_abstractive --judge sonnet [--auto light]
"""

import argparse
from pathlib import Path
import sys
from dataclasses import asdict, replace

import dspy

from hebqa.config import JUDGE_LMS, REFLECTION_LM, TASK_LM, AbstractiveSplitConfig, GEPAConfig
from hebqa.data import load_abstractive_splits
from hebqa.judge import JUDGE_INSTRUCTIONS, Judge, make_judge_metric
from hebqa.lm import describe_lm, lm_usage, make_lm
from hebqa.optimize import run_gepa, summarize_gepa
from hebqa.programs import build_abstractive_qa, program_instructions
from hebqa.runlog import RunLog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", choices=sorted(JUDGE_LMS), required=True)
    parser.add_argument("--seed-lang", choices=["he", "en"], default="he")
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=GEPAConfig.auto)
    # Qwen thinks at length on long documents; 4000 tokens left some answers empty.
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--task-model", default=TASK_LM.model, help="task model id (OpenRouter id, or the server's id with --task-api-base)")
    parser.add_argument("--task-api-base", default=None, help="OpenAI-compatible base URL for the task model instead of OpenRouter")
    parser.add_argument("--reflection-model", default=REFLECTION_LM.model,
                        help="OpenRouter id of the model that proposes new prompts (default: Sonnet)")
    parser.add_argument("--seed", type=int, default=GEPAConfig.seed, help="GEPA random seed (candidate sampling, minibatches)")
    parser.add_argument("--reflection-minibatch", type=int, default=GEPAConfig.reflection_minibatch_size,
                        help="failed examples shown to the reflection model per rewrite")
    parser.add_argument("--reflection-guidance-file", type=Path, default=None,
                        help="text file with extra guidance for the reflection model, appended to every proposal prompt")
    parser.add_argument("--threads", type=int, default=GEPAConfig.num_threads)
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    split_cfg = AbstractiveSplitConfig()
    guidance = args.reflection_guidance_file.read_text(encoding="utf-8").strip() if args.reflection_guidance_file else None
    gepa_cfg = GEPAConfig(auto=args.auto, num_threads=args.threads, seed=args.seed,
                          reflection_minibatch_size=args.reflection_minibatch, reflection_guidance=guidance)
    task_cfg = replace(TASK_LM, model=args.task_model, api_base=args.task_api_base, max_tokens=args.max_tokens)
    judge_cfg = JUDGE_LMS[args.judge]
    reflection_cfg = replace(REFLECTION_LM, model=args.reflection_model)

    splits = load_abstractive_splits(split_cfg)
    task_lm, reflection_lm, judge_lm = make_lm(task_cfg), make_lm(reflection_cfg), make_lm(judge_cfg)
    dspy.configure(lm=task_lm)
    seed_program = build_abstractive_qa(args.seed_lang)
    judge_errors: list[str] = []
    metric = make_judge_metric(Judge(judge_lm), judge_errors)

    run_name = f"gepa_abstractive_{args.judge}_{args.auto}_{args.seed_lang}_{args.task_model.split('/')[-1]}"
    if args.reflection_model != REFLECTION_LM.model:
        run_name += f"_refl-{args.reflection_model.split('/')[-1]}"
    if args.seed != GEPAConfig.seed:
        run_name += f"_s{args.seed}"
    if args.reflection_minibatch != GEPAConfig.reflection_minibatch_size:
        run_name += f"_mb{args.reflection_minibatch}"
    if guidance:
        run_name += "_steer"
    log = RunLog(run_name)
    record = {
        "stage": "abstractive/gepa",
        "judge": args.judge,
        "seed_language": args.seed_lang,
        "splits": asdict(split_cfg),
        "n_train": len(splits.train),
        "n_val": len(splits.val),
        "gepa": asdict(gepa_cfg),
        "task_lm": describe_lm(task_cfg),
        "reflection_lm": describe_lm(reflection_cfg),
        "judge_lm": describe_lm(judge_cfg),
        "rubric": JUDGE_INSTRUCTIONS,
        "seed_instructions": program_instructions(seed_program),
    }
    log.write_json("run.json", record)  # written early so a crashed run still records its config

    optimized = run_gepa(
        seed_program, splits.train, splits.val, reflection_lm, gepa_cfg, str(log.dir / "gepa"), metric=metric
    )
    optimized.save(str(log.dir / "program.json"))
    gepa_summary = summarize_gepa(optimized)
    log.write_json("gepa_candidates.json", gepa_summary)

    best = gepa_summary["candidates"][gepa_summary["best_idx"]]
    record.update(
        optimized_instructions=program_instructions(optimized),
        best_val_score=best["val_score"],
        seed_val_score=gepa_summary["candidates"][0]["val_score"],
        n_candidates=len(gepa_summary["candidates"]),
        total_metric_calls=gepa_summary["total_metric_calls"],
        judge_errors=judge_errors,
        usage={"task_lm": lm_usage(task_lm), "reflection_lm": lm_usage(reflection_lm), "judge_lm": lm_usage(judge_lm)},
    )
    log.write_json("run.json", record)

    print("\n=== Evolved instructions ===")
    for name, text in record["optimized_instructions"].items():
        print(f"[{name}]\n{text}\n")
    print(f"val judge score ({args.judge}): seed {record['seed_val_score']:.3f} -> best {record['best_val_score']:.3f} "
          f"({record['n_candidates']} candidates, {record['total_metric_calls']} metric calls, "
          f"{len(judge_errors)} judge errors)")
    print(f"usage: {record['usage']}")
    print(f"saved to {log.dir}")


if __name__ == "__main__":
    main()
