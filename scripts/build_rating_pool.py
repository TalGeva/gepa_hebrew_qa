"""Stage 2b: Qwen answers for human rating, to validate the judge on the answers GEPA will see.

The dataset's rated answers come from a strong model and are almost all 2s. This draws
questions from train and val only (the test set stays unseen), stratified by level, and
answers them with the Hebrew seed program. Writes pool.jsonl (answers with claims, level,
split) for the rating page and for the judge.

Run from the project root:  python -m scripts.build_rating_pool [--n 40]
"""

import argparse
import random
import sys
from collections import defaultdict
from dataclasses import asdict, replace

import dspy

from hebqa.config import TASK_LM, AbstractiveSplitConfig
from hebqa.data import load_abstractive_splits
from hebqa.lm import describe_lm, lm_usage, make_lm
from hebqa.programs import build_abstractive_qa, program_instructions
from hebqa.runlog import RunLog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=40, help="questions drawn from train and val together")
    # Qwen thinks at length on long documents; 4000 tokens left some answers empty.
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def stratified_sample(examples: list[dspy.Example], n: int, seed: int) -> list[dspy.Example]:
    """n examples with each level's share of the pool, in a seeded random order."""
    by_level: dict[int, list[dspy.Example]] = defaultdict(list)
    for ex in examples:
        by_level[ex.difficulty].append(ex)
    rng = random.Random(seed)
    picked = []
    for level in sorted(by_level):
        group = by_level[level]
        picked += rng.sample(group, round(n * len(group) / len(examples)))
    rng.shuffle(picked)
    return picked[:n]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    split_cfg, lm_cfg = AbstractiveSplitConfig(), replace(TASK_LM, max_tokens=args.max_tokens)
    splits = load_abstractive_splits(split_cfg)
    split_of = {ex.id: "train" for ex in splits.train} | {ex.id: "val" for ex in splits.val}
    examples = stratified_sample(splits.train + splits.val, args.n, args.seed)

    lm = make_lm(lm_cfg)
    dspy.configure(lm=lm)
    program = build_abstractive_qa("he")
    preds = program.batch(examples, num_threads=args.threads, max_errors=len(examples))

    rows = []
    for i, (ex, pred) in enumerate(zip(examples, preds), 1):
        ok = pred is not None and (pred.get("answer") or "").strip()
        rows.append(
            {
                "pool_id": f"q{i:03d}",
                "question_id": ex.id,
                "split": split_of[ex.id],
                "difficulty": ex.difficulty,
                "genre": ex.genre,
                "question": ex.question,
                "context": ex.context,
                "answer": pred.answer.strip() if ok else "",
                "claims": [c.model_dump() for c in pred.claims] if ok else [],
            }
        )

    log = RunLog("rating_pool")
    log.write_json(
        "run.json",
        {
            "stage": "judge/rating_pool",
            "splits": asdict(split_cfg),
            "n": args.n,
            "seed": args.seed,
            "task_lm": describe_lm(lm_cfg),
            "instructions": program_instructions(program),
            "usage": lm_usage(lm),
        },
    )
    log.write_jsonl("pool.jsonl", rows)

    levels = defaultdict(int)
    for r in rows:
        levels[r["difficulty"]] += 1
    print(f"{len(rows)} answers; empty: {sum(not r['answer'] for r in rows)}; by level: {dict(sorted(levels.items()))}")
    print(f"usage: {lm_usage(lm)}")
    print(f"saved to {log.dir}")


if __name__ == "__main__":
    main()
