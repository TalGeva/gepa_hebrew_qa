"""Language model clients. All calls go through OpenRouter."""

import os
from dataclasses import asdict

import dspy
from dotenv import load_dotenv

from hebqa.config import PROJECT_ROOT, LMConfig


def make_lm(cfg: LMConfig) -> dspy.LM:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to .env in the project root (see .env.example).")
    return dspy.LM(f"openrouter/{cfg.model}", temperature=cfg.temperature, max_tokens=cfg.max_tokens)


def lm_usage(lm: dspy.LM, since: int = 0) -> dict:
    """Billed tokens and dollars for lm.history[since:]. Cache hits are counted but not billed."""
    usage = {
        "calls": 0,
        "cached_calls": 0,
        "truncated_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }
    for entry in lm.history[since:]:
        response = entry["response"]
        usage["calls"] += 1
        usage["truncated_calls"] += response.finish_reason == "length"
        tokens = entry["usage"]
        if not tokens:  # DSPy records empty usage on a cache hit; the stored response keeps its old cost
            usage["cached_calls"] += 1
            continue
        usage["prompt_tokens"] += tokens.get("prompt_tokens") or 0
        usage["completion_tokens"] += tokens.get("completion_tokens") or 0
        usage["reasoning_tokens"] += (tokens.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        # DSPy's entry["cost"] stays empty on OpenRouter; the billed amount is in the raw response.
        usage["cost_usd"] += (response.provider_data or {}).get("usage", {}).get("cost") or 0.0
    usage["cost_usd"] = round(usage["cost_usd"], 6)
    return usage


def describe_lm(cfg: LMConfig) -> dict:
    return {"provider": "openrouter", **asdict(cfg)}
