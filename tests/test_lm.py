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


def test_make_lm_forwards_top_p_and_extra_body():
    from dataclasses import replace

    from hebqa.config import LMConfig
    from hebqa.lm import THINKING_MODES, make_lm

    cfg = LMConfig(model="m", temperature=0.0, max_tokens=10, api_base="http://localhost:1/v1")
    cfg = replace(cfg, **THINKING_MODES["on"])
    lm = make_lm(cfg)
    assert lm.kwargs["temperature"] == 1.0 and lm.kwargs["top_p"] == 1.0 and lm.kwargs["max_tokens"] == 16000
    assert lm.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    off = make_lm(LMConfig(model="m", temperature=0.0, max_tokens=10, api_base="http://localhost:1/v1"))
    assert "extra_body" not in off.kwargs and "top_p" not in off.kwargs


def test_lm_usage_reads_finish_reason_from_choices_when_missing_at_top_level():
    from types import SimpleNamespace

    from hebqa.lm import lm_usage

    resp = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")], provider_data=None)
    lm = SimpleNamespace(history=[{"response": resp, "usage": {"prompt_tokens": 1, "completion_tokens": 2}}])
    assert lm_usage(lm)["truncated_calls"] == 1
