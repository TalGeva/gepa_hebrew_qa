"""Score candidate task models with the seed prompt, to pick one with room for GEPA to improve.

Each model answers the development questions (train + val; the test set stays unseen) with
the Hebrew seed prompt, and the frozen judge scores every answer.

Run from the project root:
    python -m scripts.screen_task_models --models google/gemma-3-27b-it mistralai/mistral-small-24b-instruct-2501@4000
A model id may end in @N to set its max_tokens (for models with a short context window).
A model OpenRouter does not serve runs on an OpenAI-compatible endpoint:
    python -m scripts.screen_task_models --models dicta-il/DictaLM-3.0-24B-Thinking \\
        --api-base https://<endpoint>/v1 --api-key-env HF_TOKEN --temperature 0.6
"""

import argparse
import sys

import dspy

from hebqa.config import JUDGE_LM, LMConfig
from hebqa.data import load_abstractive_splits
from hebqa.judge import CRITERIA, Judge, evaluate_with_judges
from hebqa.lm import describe_lm, lm_usage, make_lm
from hebqa.programs import build_abstractive_qa, program_instructions
from hebqa.runlog import RunLog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--max-tokens", type=int, default=8000, help="default for models without @N")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-base", help="OpenAI-compatible endpoint serving the models (default: OpenRouter)")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY", help="environment variable with its key")
    parser.add_argument("--seed-lang", choices=["he", "en"], default="he")
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    splits = load_abstractive_splits()
    dev = splits.train + splits.val
    judge_lm = make_lm(JUDGE_LM)
    judges = {"judge": Judge(judge_lm)}
    program = build_abstractive_qa(args.seed_lang)

    log = RunLog("task_model_screen")
    record = {
        "stage": "abstractive/screen_task_models",
        "n_dev": len(dev),
        "judge_lm": describe_lm(JUDGE_LM),
        "instructions": program_instructions(program),
        "models": {},
    }
    for spec in args.models:
        model, _, max_tokens = spec.partition("@")
        cfg = LMConfig(
            model=model, temperature=args.temperature, max_tokens=int(max_tokens or args.max_tokens),
            api_base=args.api_base, api_key_env=args.api_key_env,
        )
        task_lm = make_lm(cfg)
        judge_calls_before = len(judge_lm.history)
        dspy.configure(lm=task_lm)
        try:
            summary, rows = evaluate_with_judges(program, dev, judges, num_threads=args.threads)
        except Exception as e:  # one model failing must not lose the others
            record["models"][model] = {"lm": describe_lm(cfg), "error": f"{type(e).__name__}: {e}"}
            log.write_json("run.json", record)
            print(f"{model}: failed: {e}")
            continue
        record["models"][model] = {
            "lm": describe_lm(cfg),
            **summary["judge"],
            "usage": {"task_lm": lm_usage(task_lm), "judge_lm": lm_usage(judge_lm, since=judge_calls_before)},
        }
        log.write_jsonl(f"dev_{model.replace('/', '__')}.jsonl", rows)
        log.write_json("run.json", record)
        s = record["models"][model]
        print(f"{model}: {s['normalized']:.3f}  " + " ".join(f"{k[:6]} {s['criteria'][k]:.2f}" for k in CRITERIA)
              + f"  by level {s['by_level']}  empty {s['empty_answers']} errors {s['errors']}", flush=True)

    print(f"\nsaved to {log.dir}")


if __name__ == "__main__":
    main()
