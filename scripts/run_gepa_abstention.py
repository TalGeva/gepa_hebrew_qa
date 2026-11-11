"""GEPA on the combined HeQ + abstractive pool with the abstention-gated metric.

Train and validation mix HeQ abstention samples with the abstractive items (hebqa.data.combined);
the metric (hebqa.gated_metric) puts the abstention decision first and needs the judge only for
abstractive answerable answers. Judge and reflection default to Gemma, so the loop is open-weights.

Run from the project root (task model on a vLLM server through a tunnel):
    python -m scripts.run_gepa_abstention --task-model HebArabNlpProject/Hebatron \
        --task-api-base http://localhost:8002/v1 --thinking off
"""

import argparse
import sys
from dataclasses import asdict, replace

import dspy

from hebqa.config import JUDGE_LMS, REFLECTION_LM, TASK_LM, GEPAConfig
from hebqa.data.combined import load_combined_splits
from hebqa.gated_metric import WEIGHTS, make_gated_metric
from hebqa.judge import Judge
from hebqa.lm import THINKING_MODES, describe_lm, lm_usage, make_lm
from hebqa.optimize import run_gepa, summarize_gepa
from hebqa.programs import build_abstractive_qa, program_instructions
from hebqa.runlog import RunLog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", choices=sorted(JUDGE_LMS), default="gemma")
    parser.add_argument("--reflection-model", default="google/gemma-4-31b-it")
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default=GEPAConfig.auto)
    parser.add_argument("--task-model", default=TASK_LM.model)
    parser.add_argument("--task-api-base", default=None)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--thinking", choices=sorted(THINKING_MODES), default="off")
    parser.add_argument("--seed", type=int, default=GEPAConfig.seed)
    parser.add_argument("--reflection-minibatch", type=int, default=GEPAConfig.reflection_minibatch_size)
    parser.add_argument("--threads", type=int, default=GEPAConfig.num_threads)
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    gepa_cfg = GEPAConfig(auto=args.auto, num_threads=args.threads, seed=args.seed,
                          reflection_minibatch_size=args.reflection_minibatch)
    mode = THINKING_MODES[args.thinking]
    task_cfg = replace(TASK_LM, model=args.task_model, api_base=args.task_api_base, **({"max_tokens": args.max_tokens} | mode))
    judge_cfg = JUDGE_LMS[args.judge]
    reflection_cfg = replace(REFLECTION_LM, model=args.reflection_model)

    splits, _ = load_combined_splits()
    task_lm, reflection_lm, judge_lm = make_lm(task_cfg), make_lm(reflection_cfg), make_lm(judge_cfg)
    dspy.configure(lm=task_lm)
    seed_program = build_abstractive_qa("he")
    judge_errors: list[str] = []
    metric = make_gated_metric(Judge(judge_lm), judge_errors)

    run_name = f"gepa_abstention_{args.judge}_{args.auto}_{args.task_model.split('/')[-1]}_think-{args.thinking}"
    if args.seed != GEPAConfig.seed:
        run_name += f"_s{args.seed}"
    log = RunLog(run_name)
    record = {
        "stage": "abstention/gepa",
        "judge": args.judge,
        "seed_language": "he",
        "thinking": args.thinking,
        "metric": {"name": "gated", "weights": WEIGHTS},
        "n_train": len(splits.train),
        "n_val": len(splits.val),
        "n_train_unanswerable": sum(ex.difficulty == 0 for ex in splits.train),
        "n_val_unanswerable": sum(ex.difficulty == 0 for ex in splits.val),
        "gepa": asdict(gepa_cfg),
        "task_lm": describe_lm(task_cfg),
        "reflection_lm": describe_lm(reflection_cfg),
        "judge_lm": describe_lm(judge_cfg),
        "seed_instructions": program_instructions(seed_program),
    }
    log.write_json("run.json", record)

    optimized = run_gepa(seed_program, splits.train, splits.val, reflection_lm, gepa_cfg, str(log.dir / "gepa"), metric=metric)
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
    print(f"val gated score: seed {record['seed_val_score']:.3f} -> best {record['best_val_score']:.3f} "
          f"({record['n_candidates']} candidates, {record['total_metric_calls']} metric calls, {len(judge_errors)} judge errors)")
    print(f"usage: {record['usage']}")
    print(f"saved to {log.dir}")


if __name__ == "__main__":
    main()
