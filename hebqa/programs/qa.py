"""DSPy programs under optimization."""

import dspy

from hebqa.claims import Claim
from hebqa.programs.prompts import ABSTRACTIVE_QA_SEED, EXTRACTIVE_QA_SEED


class ExtractiveQA(dspy.Signature):
    # Instructions are set per seed language in build_extractive_qa; fields carry no
    # descriptions so the instruction text is the only language-specific part of the prompt.
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class AbstractiveQA(dspy.Signature):
    # The answer plus the claims it makes, each backed by verbatim quotes (judged as attribution).
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    claims: list[Claim] = dspy.OutputField()


def _build(signature: type[dspy.Signature], seeds: dict[str, str], seed_language: str) -> dspy.Module:
    if seed_language not in seeds:
        raise ValueError(f"Unknown seed language {seed_language!r}; expected one of {sorted(seeds)}")
    return dspy.Predict(signature.with_instructions(seeds[seed_language]))


def build_extractive_qa(seed_language: str = "he") -> dspy.Module:
    return _build(ExtractiveQA, EXTRACTIVE_QA_SEED, seed_language)


def build_abstractive_qa(seed_language: str = "he", instructions: str | None = None) -> dspy.Module:
    """The seed program, or one with custom instructions."""
    if instructions is not None:
        return dspy.Predict(AbstractiveQA.with_instructions(instructions))
    return _build(AbstractiveQA, ABSTRACTIVE_QA_SEED, seed_language)


def program_instructions(program: dspy.Module) -> dict[str, str]:
    """Current instructions of every predictor, keyed by predictor name (what GEPA rewrites)."""
    return {name: pred.signature.instructions for name, pred in program.named_predictors()}
