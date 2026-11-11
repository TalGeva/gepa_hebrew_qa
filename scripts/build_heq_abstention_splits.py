"""Write the HeQ abstention splits (data/heq_abstention/{train,val,test}.json) and print a summary.

Run from the project root:
    python -m scripts.build_heq_abstention_splits [--n-train 150 --n-val 300 --n-test 300 --seed 42]
"""

import argparse
import json
import sys
from collections import Counter

from hebqa.data.heq_abstention import HeQAbstentionConfig, build


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-train", type=int, default=HeQAbstentionConfig.n_train)
    parser.add_argument("--n-val", type=int, default=HeQAbstentionConfig.n_val)
    parser.add_argument("--n-test", type=int, default=HeQAbstentionConfig.n_test)
    parser.add_argument("--seed", type=int, default=HeQAbstentionConfig.seed)
    args = parser.parse_args()

    paths = build(HeQAbstentionConfig(n_train=args.n_train, n_val=args.n_val, n_test=args.n_test, seed=args.seed))
    for name, path in paths.items():
        items = json.loads(path.read_text(encoding="utf-8"))["items"]
        print(f"{name}: {len(items)} questions, {sum(not r['answerable'] for r in items)} unanswerable, "
              f"{len({r['context'] for r in items})} passages, sources {dict(Counter(r['source'] for r in items))}, "
              f"quality {dict(Counter(r['quality'] for r in items))}")
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
