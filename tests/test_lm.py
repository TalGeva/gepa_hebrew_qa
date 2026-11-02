from types import SimpleNamespace

from hebqa.lm import lm_usage


def _entry(usage, cost, finish_reason="stop"):
    response = SimpleNamespace(finish_reason=finish_reason, provider_data={"usage": {"cost": cost}})
    return {"usage": usage, "response": response}


BILLED = {"prompt_tokens": 10, "completion_tokens": 20, "completion_tokens_details": {"reasoning_tokens": 15}}


def test_cache_hits_are_counted_but_not_billed():
    lm = SimpleNamespace(history=[_entry(BILLED, 0.5), _entry({}, 0.5)])
    usage = lm_usage(lm)
    assert usage["calls"] == 2 and usage["cached_calls"] == 1
    assert usage["cost_usd"] == 0.5
    assert usage["prompt_tokens"] == 10 and usage["completion_tokens"] == 20 and usage["reasoning_tokens"] == 15


def test_since_and_truncation():
    lm = SimpleNamespace(history=[_entry(BILLED, 1.0), _entry(BILLED, 0.25, finish_reason="length")])
    usage = lm_usage(lm, since=1)
    assert usage["calls"] == 1 and usage["truncated_calls"] == 1 and usage["cost_usd"] == 0.25
