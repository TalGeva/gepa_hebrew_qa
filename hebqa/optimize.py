"""Run GEPA on a program and summarize what it found."""

import dspy

from hebqa.config import GEPAConfig
from hebqa.metrics import make_gepa_metric
from hebqa.programs import program_instructions


def run_gepa(
    program: dspy.Module,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    reflection_lm: dspy.LM,
    cfg: GEPAConfig,
    log_dir: str,
    metric=None,
) -> dspy.Module:
    """Optimize the program's instructions. The task LM is whatever dspy.configure set.

    metric defaults to the extractive TLNLS metric with cfg.feedback.
    """
    optimizer = dspy.GEPA(
        metric=metric or make_gepa_metric(cfg.feedback),
        auto=cfg.auto,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=cfg.reflection_minibatch_size,
        num_threads=cfg.num_threads,
        seed=cfg.seed,
        track_stats=True,  # attaches detailed_results to the returned program
        log_dir=log_dir,  # GEPA checkpoints here, so an interrupted run resumes from the same dir
    )
    return optimizer.compile(program, trainset=trainset, valset=valset)


def summarize_gepa(optimized: dspy.Module) -> dict:
    """Every candidate GEPA proposed, with its lineage, val score and the budget spent to find it."""
    results = optimized.detailed_results
    candidates = [
        {
            "idx": i,
            "parents": results.parents[i],
            "val_score": results.val_aggregate_scores[i],
            "found_after_metric_calls": results.discovery_eval_counts[i],
            "instructions": program_instructions(candidate),
        }
        for i, candidate in enumerate(results.candidates)
    ]
    return {
        "best_idx": results.best_idx,
        "total_metric_calls": results.total_metric_calls,
        "num_full_val_evals": results.num_full_val_evals,
        "candidates": candidates,
    }
