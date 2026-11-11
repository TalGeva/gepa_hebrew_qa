"""Language model clients. All calls go through OpenRouter."""

import os
from dataclasses import asdict

import dspy
from dotenv import load_dotenv

from hebqa.config import PROJECT_ROOT, LMConfig


def make_lm(cfg: LMConfig) -> dspy.LM:
    extra = {}
    if cfg.top_p is not None:
        extra["top_p"] = cfg.top_p
    if cfg.extra_body:
        extra["extra_body"] = cfg.extra_body  # forwarded by LiteLLM into the request body
    if cfg.api_base:  # a local OpenAI-compatible server; no key needed
        return dspy.LM(f"openai/{cfg.model}", api_base=cfg.api_base, api_key="EMPTY",
                       temperature=cfg.temperature, max_tokens=cfg.max_tokens, **extra)
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to .env in the project root (see .env.example).")
    return dspy.LM(f"openrouter/{cfg.model}", temperature=cfg.temperature, max_tokens=cfg.max_tokens, **extra)


# Hebatron (Nemotron 3 Nano) decoding modes from the model card: thinking off = greedy;
# thinking on = temperature 1.0, top_p 1.0, a large output budget for the reasoning trace.
THINKING_MODES = {
    "template": {},  # whatever the chat template does by default (thinking on, but Hebatron closes the block at once)
    "off": {"temperature": 0.0, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    "on": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 16000, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
}


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
        finish = getattr(response, "finish_reason", None)
        if finish is None and getattr(response, "choices", None):  # LiteLLM ModelResponse keeps it per choice
            finish = getattr(response.choices[0], "finish_reason", None)
        usage["truncated_calls"] += finish == "length"
        tokens = entry["usage"]
        if not tokens:  # DSPy records empty usage on a cache hit; the stored response keeps its old cost
            usage["cached_calls"] += 1
            continue
        usage["prompt_tokens"] += tokens.get("prompt_tokens") or 0
        usage["completion_tokens"] += tokens.get("completion_tokens") or 0
        usage["reasoning_tokens"] += (tokens.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        # DSPy's entry["cost"] stays empty on OpenRouter; the billed amount is in the raw response.
        usage["cost_usd"] += (getattr(response, "provider_data", None) or {}).get("usage", {}).get("cost") or 0.0
    usage["cost_usd"] = round(usage["cost_usd"], 6)
    return usage


def describe_lm(cfg: LMConfig) -> dict:
    return {"provider": "openai-compatible" if cfg.api_base else "openrouter", **asdict(cfg)}
