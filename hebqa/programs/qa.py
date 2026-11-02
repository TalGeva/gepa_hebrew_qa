"""DSPy programs under optimization."""

import dspy

from hebqa.programs.prompts import EXTRACTIVE_QA_SEED


class ExtractiveQA(dspy.Signature):
    # Instructions are set per seed language in build_extractive_qa; fields carry no
    # descriptions so the instruction text is the only language-specific part of the prompt.
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def build_extractive_qa(seed_language: str = "he") -> dspy.Module:
    if seed_language not in EXTRACTIVE_QA_SEED:
        raise ValueError(f"Unknown seed language {seed_language!r}; expected one of {sorted(EXTRACTIVE_QA_SEED)}")
    return dspy.Predict(ExtractiveQA.with_instructions(EXTRACTIVE_QA_SEED[seed_language]))


def program_instructions(program: dspy.Module) -> dict[str, str]:
    """Current instructions of every predictor, keyed by predictor name (what GEPA rewrites)."""
    return {name: pred.signature.instructions for name, pred in program.named_predictors()}
