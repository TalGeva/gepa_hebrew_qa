"""Agreement between the judge and the dataset's human annotators.

Ratings are 0-2 and mostly 2, so raw agreement is inflated; chance-corrected agreement
(Krippendorff's alpha) and flaw detection carry the signal.
"""

import itertools
from statistics import mean

# An answer counts as flawed on a criterion when the annotators' mean is at or below this:
# at least two of four gave less than 2, or one gave 0.
FLAW_THRESHOLD = 1.5


def krippendorff_alpha(units: list[list[float]]) -> float:
    """Interval alpha. Each unit holds the ratings it received; units with fewer than two are skipped."""
    units = [u for u in units if len(u) >= 2]
    observed = [(a - b) ** 2 / (len(u) - 1) for u in units for a, b in itertools.permutations(u, 2)]
    values = [v for u in units for v in u]
    n = len(values)
    d_o = sum(observed) / n
    d_e = sum((a - b) ** 2 for a, b in itertools.permutations(values, 2)) / (n * (n - 1))
    return 1.0 if d_e == 0 else 1 - d_o / d_e


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=xs.__getitem__)
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return float("nan") if var == 0 else cov / var


def criterion_agreement(humans: list[list[int]], judge: list[int]) -> dict:
    """humans[i] holds item i's annotator scores (same annotator order on every item); judge[i] the judge's."""
    n_raters = len(humans[0])
    human_mean = [mean(h) for h in humans]

    def pair_alpha(a: list[int], b: list[int]) -> float:
        return krippendorff_alpha([[x, y] for x, y in zip(a, b)])

    columns = [[h[r] for h in humans] for r in range(n_raters)]
    human_pairs = [pair_alpha(a, b) for a, b in itertools.combinations(columns, 2)]
    judge_pairs = [pair_alpha(judge, c) for c in columns]

    flawed = [m <= FLAW_THRESHOLD for m in human_mean]
    flagged = [j < 2 for j in judge]
    tp = sum(f and g for f, g in zip(flawed, flagged))
    return {
        "human_mean": round(mean(human_mean), 3),
        "judge_mean": round(mean(judge), 3),
        "alpha_humans": round(krippendorff_alpha([list(h) for h in humans]), 3),
        "alpha_humans_plus_judge": round(krippendorff_alpha([list(h) + [j] for h, j in zip(humans, judge)]), 3),
        "pairwise_alpha_human_human": round(mean(human_pairs), 3),
        "pairwise_alpha_judge_human": round(mean(judge_pairs), 3),
        "spearman_vs_human_mean": round(spearman(judge, human_mean), 3),
        "n_flawed": sum(flawed),
        "n_flagged": sum(flagged),
        "flaw_recall": round(tp / sum(flawed), 3) if any(flawed) else None,
        "flaw_precision": round(tp / sum(flagged), 3) if any(flagged) else None,
    }
