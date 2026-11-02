"""Load the HeQ pilot splits and print a summary.

Run from the project root:  python -m scripts.inspect_heq
"""

import sys
from collections import Counter

from hebqa.config import PilotConfig
from hebqa.data import load_heq_pilot


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = PilotConfig()
    splits = load_heq_pilot(cfg)

    print(f"Pilot config: {cfg}\n")
    for name, examples in splits._asdict().items():
        n_answers = Counter(len(ex.answers) for ex in examples)
        sources = Counter(ex.source for ex in examples)
        print(f"{name:5}  n={len(examples):3}  gold spans/question={dict(sorted(n_answers.items()))}  sources={dict(sources)}")

    ex = splits.test[0]
    print("\nSample test example:")
    print("  inputs: ", list(ex.inputs().keys()))
    print("  question:", ex.question)
    print("  answers: ", ex.answers)
    print("  context: ", ex.context[:200], "...")


if __name__ == "__main__":
    main()
