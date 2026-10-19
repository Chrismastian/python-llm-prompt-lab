"""Tests for the pure prompt layer. No network, no API key needed."""
import pytest

from llmlab import prompts


def test_build_messages_has_system_then_user():
    messages = prompts.build_messages("What is a token?")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == prompts.SYSTEM_PROMPT
    assert messages[1]["content"] == "What is a token?"


def test_strict_messages_adds_output_contract():
    messages = prompts.strict_messages("What is a token?")
    assert messages[0]["content"].startswith(prompts.SYSTEM_PROMPT)
    assert "JSON object only" in messages[0]["content"]
    assert messages[0]["content"] != prompts.SYSTEM_PROMPT


def test_extract_json_accepts_fenced_reply():
    # Models wrap output in a code fence even when told not to.
    text = '```json\n{"answer": "400", "confidence": 0.9}\n```'
    assert prompts.extract_json(text) == {"answer": "400", "confidence": 0.9}


def test_extract_json_accepts_clean_object():
    assert prompts.extract_json('{"answer": "400"}') == {"answer": "400"}


def test_extract_json_rejects_prose():
    # The failure that started all this: a model that ignores the contract.
    with pytest.raises(ValueError):
        prompts.extract_json("**400 Bad Request** is the answer.")


def test_usage_totals_defaults_to_zero():
    assert prompts.usage_totals({}) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_usage_totals_reads_wire_format():
    raw = {"usage": {"prompt_tokens": 24, "completion_tokens": 92, "total_tokens": 116}}
    assert prompts.usage_totals(raw)["total_tokens"] == 116