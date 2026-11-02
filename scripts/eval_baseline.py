"""Stage 1, step 2: score the hand-written seed prompt on the pilot test set.

Run from the project root:  python -m scripts.eval_baseline [--split test] [--seed-lang he] [--limit 20]
"""

import argparse
import sys
from dataclasses import asdict, replace

import dspy

from hebqa.config import TASK_LM, PilotConfig
from hebqa.data import load_heq_pilot
from hebqa.evaluation import evaluate
from hebqa.lm import describe_lm, lm_usage, make_lm
from hebqa.programs import build_extractive_qa, program_instructions
from hebqa.runlog import RunLog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--seed-lang", choices=["he", "en"], default="he")
    parser.add_argument("--model", default=TASK_LM.model, help="OpenRouter model id of the task model")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, help="score only the first N examples (smoke test)")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    pilot_cfg, lm_cfg = PilotConfig(), replace(TASK_LM, model=args.model)

    examples = getattr(load_heq_pilot(pilot_cfg), args.split)[: args.limit]
    lm = make_lm(lm_cfg)
    dspy.configure(lm=lm)
    program = build_extractive_qa(args.seed_lang)

    summary, rows = evaluate(program, examples, num_threads=args.threads)
    usage = lm_usage(lm)

    log = RunLog(f"baseline_{args.seed_lang}_{args.split}")
    log.write_json(
        "run.json",
        {
            "stage": "pilot/baseline",
            "split": args.split,
            "seed_language": args.seed_lang,
            "pilot": asdict(pilot_cfg),
            "task_lm": describe_lm(lm_cfg),
            "instructions": program_instructions(program),
            "scores": summary,
            "usage": usage,
        },
    )
    log.write_jsonl("predictions.jsonl", rows)
    program.save(str(log.dir / "program.json"))

    print(f"\n{args.split} n={summary['n']}  EM={summary['em']}  F1={summary['f1']}  TLNLS={summary['tlnls']}")
    print(f"empty predictions: {summary['n_empty_predictions']}")
    print(f"usage: {usage}")
    print(f"saved to {log.dir}")


if __name__ == "__main__":
    main()
