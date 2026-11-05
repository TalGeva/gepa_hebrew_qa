"""Stage 2: validate candidate judges against the dataset's human ratings.

Each judge scores the 132 model answers in the dataset (with their claims and quotes) on the
five Part II criteria, and its scores are compared with the four annotators'. Agreement is
reported on all items and separately on the test split, which judge tuning must not look at.

Run from the project root:  python -m scripts.validate_judge [--judges sonnet gemma] [--limit N]
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

from hebqa.config import JUDGE_LMS, AbstractiveSplitConfig
from hebqa.data import read_abstractive, split_by_document
from hebqa.data.abstractive import EXCLUDED
from hebqa.judge import CRITERIA, JUDGE_INSTRUCTIONS, Judge, criterion_agreement
from hebqa.lm import describe_lm, lm_usage, make_lm
from hebqa.runlog import RunLog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judges", nargs="+", choices=sorted(JUDGE_LMS), default=sorted(JUDGE_LMS))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int, help="score only the first N items (smoke test)")
    parser.add_argument("--claim-check", action="store_true", help="add the claim-by-claim support check")
    return parser.parse_args()


def agreement_table(items, judgments: dict[str, dict]) -> dict:
    ok = [it for it in items if "scores" in judgments[it.id]]
    return {
        key: criterion_agreement(
            [[h[key] for h in it.human_scores] for it in ok], [judgments[it.id]["scores"][key] for it in ok]
        )
        for key in CRITERIA
    } | {"n": len(ok)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    items = [it for it in read_abstractive() if it.id not in EXCLUDED][: args.limit]
    split_cfg = AbstractiveSplitConfig()
    test_ids = {it.id for it in split_by_document(read_abstractive(), split_cfg.n_train, split_cfg.n_val, split_cfg.seed)[2]}

    log = RunLog("judge_validation")
    record = {
        "stage": "judge/validate",
        "rubric": JUDGE_INSTRUCTIONS,
        "claim_check": args.claim_check,
        "n_items": len(items),
        "judges": {},
    }
    for name in args.judges:
        lm = make_lm(JUDGE_LMS[name])
        judge = Judge(lm, check_claims=args.claim_check)

        def score(it) -> dict:
            out = {"id": it.id, "difficulty": it.difficulty, "split": "test" if it.id in test_ids else "dev"}
            try:
                j = judge(it.context, it.question, it.difficulty, it.dataset_answer, list(it.dataset_claims))
                out.update(scores=j.scores, reasons=j.reasons)
            except Exception as e:  # a failed judgment is recorded, not fatal
                out["error"] = f"{type(e).__name__}: {e}"
            out["human_scores"] = list(it.human_scores)
            return out

        with ThreadPoolExecutor(args.threads) as pool:
            results = list(pool.map(score, items))
        log.write_jsonl(f"judgments_{name}.jsonl", results)

        by_id = {r["id"]: r for r in results}
        record["judges"][name] = {
            "lm": describe_lm(JUDGE_LMS[name]),
            "errors": sum("error" in r for r in results),
            "agreement_all": agreement_table(items, by_id),
            "agreement_test": agreement_table([it for it in items if it.id in test_ids], by_id),
            "usage": lm_usage(lm),
        }

        print(f"\n== {name}: {record['judges'][name]['errors']} errors, usage {record['judges'][name]['usage']}")
        table = record["judges"][name]["agreement_all"]
        print(f"{'criterion':13s} {'α humans':>9s} {'α h+judge':>9s} {'pair h-h':>9s} {'pair j-h':>9s} "
              f"{'spearman':>9s} {'flaw rec':>9s} {'flaw prec':>9s}  means h/j")
        for key in CRITERIA:
            a = table[key]
            print(f"{key:13s} {a['alpha_humans']:9.3f} {a['alpha_humans_plus_judge']:9.3f} "
                  f"{a['pairwise_alpha_human_human']:9.3f} {a['pairwise_alpha_judge_human']:9.3f} "
                  f"{a['spearman_vs_human_mean']:9.3f} {a['flaw_recall']!s:>9s} {a['flaw_precision']!s:>9s}  "
                  f"{a['human_mean']:.2f}/{a['judge_mean']:.2f} (flawed {a['n_flawed']}, flagged {a['n_flagged']})")

    log.write_json("run.json", record)
    print(f"\nsaved to {log.dir}")


if __name__ == "__main__":
    main()
