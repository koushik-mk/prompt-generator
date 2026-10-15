import pytest

from ai_service import (
    MODEL_ALLOWLIST,
    AiServiceError,
    UnsupportedModelError,
    classify_ai_exception,
    extract_prompt,
    generate_prompt,
    resolve_model_id,
)
from helpers import FakeAIBinding
from prompt_engine import DEFAULT_GENERATION_PARAMS, MAX_OUTPUT_TOKENS, SYSTEM_PROMPT


class JsLikeResponse:
    """Mimics the JS proxy object returned by the AI binding."""

    def __init__(self, response):
        self.response = response


def test_allowlist_contains_two_verified_models():
    assert MODEL_ALLOWLIST == {
        "llama8b": "@cf/meta/llama-3.1-8b-instruct-fp8",
        "llama1b": "@cf/meta/llama-3.2-1b-instruct",
    }


def test_resolve_model_id_valid():
    assert resolve_model_id("llama8b") == "@cf/meta/llama-3.1-8b-instruct-fp8"
    assert resolve_model_id("llama1b") == "@cf/meta/llama-3.2-1b-instruct"


def test_resolve_model_id_rejects_unknown():
    with pytest.raises(UnsupportedModelError):
        resolve_model_id("@cf/meta/llama-3.3-70b-instruct-fp8-fast")


def test_resolve_model_id_rejects_non_string():
    with pytest.raises(UnsupportedModelError):
        resolve_model_id(None)


def test_extract_prompt_from_dict():
    assert extract_prompt({"response": " Role: test "}) == "Role: test"


def test_extract_prompt_from_js_like_object():
    assert extract_prompt(JsLikeResponse("Role: test")) == "Role: test"


def test_extract_prompt_rejects_missing_response():
    with pytest.raises(AiServiceError) as exc_info:
        extract_prompt({"unexpected": "shape"})
    assert exc_info.value.status == 502
    assert exc_info.value.code == "INVALID_AI_RESPONSE"


def test_extract_prompt_rejects_empty_response():
    for empty in ({"response": ""}, {"response": "   \n "}, {"response": None}):
        with pytest.raises(AiServiceError):
            extract_prompt(empty)


def test_extract_prompt_rejects_non_string_response():
    with pytest.raises(AiServiceError):
        extract_prompt({"response": 42})


def test_classify_quota_exhaustion():
    exc = Exception(
        "Error: You have used up your daily free allocation of 10,000 neurons. "
        "Please upgrade to Cloudflare's Workers Paid plan if you would like to continue usage."
    )
    assert classify_ai_exception(exc) == "quota"


def test_classify_capacity_is_transient():
    assert classify_ai_exception(Exception("Capacity temporarily exceeded, please try again.")) == "transient"


def test_classify_timeout_is_transient():
    assert classify_ai_exception(Exception("Request timeout")) == "transient"


def test_classify_unknown_is_provider_error():
    assert classify_ai_exception(Exception("Something unexpected happened")) == "provider"


async def test_generate_prompt_uses_selected_model_llama8b():
    ai = FakeAIBinding(responses={"response": "Role: generated"})
    result = await generate_prompt(ai, "llama8b", "Write an email.")
    assert result == "Role: generated"
    assert ai.model_ids == ["@cf/meta/llama-3.1-8b-instruct-fp8"]


async def test_generate_prompt_uses_selected_model_llama1b():
    ai = FakeAIBinding(responses={"response": "Role: generated"})
    result = await generate_prompt(ai, "llama1b", "Write an email.")
    assert result == "Role: generated"
    assert ai.model_ids == ["@cf/meta/llama-3.2-1b-instruct"]


async def test_generate_prompt_sends_system_prompt_and_limits():
    ai = FakeAIBinding()
    await generate_prompt(ai, "llama8b", "Write an email.")
    payload = ai.calls[0]["payload"]
    messages = payload["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    assert payload["max_tokens"] == MAX_OUTPUT_TOKENS
    assert payload["temperature"] == DEFAULT_GENERATION_PARAMS["temperature"]


async def test_generate_prompt_unsupported_model_raises_before_ai_call():
    ai = FakeAIBinding()
    with pytest.raises(UnsupportedModelError):
        await generate_prompt(ai, "gpt-everything", "Write an email.")
    assert ai.calls == []


async def test_generate_prompt_empty_ai_response():
    ai = FakeAIBinding(responses={"response": ""})
    with pytest.raises(AiServiceError) as exc_info:
        await generate_prompt(ai, "llama8b", "Write an email.")
    assert exc_info.value.status == 502


async def test_generate_prompt_retries_once_on_transient_failure():
    ai = FakeAIBinding(
        errors=[Exception("Capacity temporarily exceeded, please try again."), None],
        responses={"response": "Role: recovered"},
    )
    # errors list is consumed first; second call must succeed.
    result = await generate_prompt(ai, "llama8b", "Write an email.")
    assert result == "Role: recovered"
    assert len(ai.calls) == 2


async def test_generate_prompt_quota_exhaustion_is_not_retried():
    ai = FakeAIBinding(
        errors=Exception(
            "You have used up your daily free allocation of 10,000 neurons."
        )
    )
    with pytest.raises(AiServiceError) as exc_info:
        await generate_prompt(ai, "llama8b", "Write an email.")
    assert exc_info.value.status == 429
    assert exc_info.value.code == "AI_QUOTA_EXCEEDED"
    assert "usage limit has been reached" in exc_info.value.message
    assert len(ai.calls) == 1


async def test_generate_prompt_persistent_transient_failure_returns_503():
    ai = FakeAIBinding(errors=Exception("Capacity temporarily exceeded, please try again."))
    with pytest.raises(AiServiceError) as exc_info:
        await generate_prompt(ai, "llama8b", "Write an email.")
    assert exc_info.value.status == 503
    assert exc_info.value.code == "AI_UNAVAILABLE"
    assert len(ai.calls) == 2  # bounded: initial + one retry


async def test_generate_prompt_provider_error_is_not_retried():
    ai = FakeAIBinding(errors=Exception("Internal model failure"))
    with pytest.raises(AiServiceError) as exc_info:
        await generate_prompt(ai, "llama8b", "Write an email.")
    assert exc_info.value.status == 502
    assert exc_info.value.code == "AI_PROVIDER_ERROR"
    assert len(ai.calls) == 1
