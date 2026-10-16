import pytest

from ai_service import (
    MODEL_ALLOWLIST,
    AiServiceError,
    UnsupportedModelError,
    classify_ai_exception,
    classify_unusable_output,
    extract_prompt,
    generate_prompt,
    resolve_model_id,
    strip_leading_commentary,
)
from helpers import FakeAIBinding
from prompt_engine import (
    CORRECTIVE_SUFFIXES,
    DEFAULT_GENERATION_PARAMS,
    MAX_OUTPUT_TOKENS,
    SYSTEM_PROMPT,
    build_messages,
)


class JsLikeResponse:
    """Mimics the JS proxy object returned by the AI binding."""

    def __init__(self, response):
        self.response = response


CLEAN_PROMPT = """Role: Act as an experienced customer support specialist.

Context: A customer is waiting for a delayed shipment.

Task: Write a professional apology email. Use placeholders such as
[Customer Name] and [Product Name] for missing details.

Constraints: Apologetic and empathetic tone; no promotional content.

Output Format: A well-structured plain-text email."""

OBSERVED_LABELLED_LEAK = """**Role:** Professional Job Application Email Writer
**Context:** A job applicant is applying for a position at XYZ Corporation.
**Task:** Write an email to apply for the Marketing Manager position.
**Constraints:**
* Use a formal business tone
* Mention the company name (XYZ Corporation)
**Output Format:** Email body

**Generated Prompt:**

Subject: Application for Marketing Manager Position at XYZ Corporation

Dear Hiring Manager,

I am excited to submit my application for the Marketing Manager position.

Best regards,
[Applicant Name]"""

OBSERVED_SELF_NOTE = """Role: Professional Job Application Email Writer
Task: Draft the application email.

Note: This prompt preserves the original scenario's intent, deliverable,
audience, and factual details while providing a well-formatted email."""

OBSERVED_LEAK_WITH_NOTE = (
    "Role: Email Writer\n"
    "Task: Generate an apology email.\n\n"
    "Dear [Customer],\n\n"
    "I am excited to apologize for the delay in shipping your product.\n\n"
    "Best regards,\n[AI System]\n\n"
    "Note: This prompt preserves the original scenario's intent while "
    "providing a well-formatted email application."
)


def test_allowlist_matches_supported_models():
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


def test_strip_leading_commentary_removes_prefaces():
    commented = "Here is the generated prompt:\n\n" + CLEAN_PROMPT
    assert strip_leading_commentary(commented) == CLEAN_PROMPT
    double = "Here's the prompt:\nHere is the result:\n\n" + CLEAN_PROMPT
    assert strip_leading_commentary(double) == CLEAN_PROMPT
    assert strip_leading_commentary(CLEAN_PROMPT) == CLEAN_PROMPT


def test_classify_flags_labelled_prompt_leak():
    assert classify_unusable_output(OBSERVED_LABELLED_LEAK) == "leak"


def test_classify_flags_self_referential_note():
    assert classify_unusable_output(OBSERVED_SELF_NOTE) == "leak"


def test_classify_flags_leak_with_trailing_note():
    assert classify_unusable_output(OBSERVED_LEAK_WITH_NOTE) == "leak"


def test_classify_flags_leading_refusal():
    refusal = "I cannot write an email to the HR department."
    assert classify_unusable_output(refusal) == "refusal"


def test_classify_flags_degenerate_repetition():
    repeated = "\n".join(
        ["Role: HR Representative"]
        + ["The email should be written in a format that is easy to follow."] * 3
    )
    assert classify_unusable_output(repeated) == "repetition"


def test_clean_prompt_is_not_flagged():
    assert classify_unusable_output(CLEAN_PROMPT) is None


def test_legitimate_prompt_mentioning_refusal_is_kept():
    legitimate = (
        "Role: Customer Support Specialist\n\n"
        "Task: Draft a professional response to the customer.\n\n"
        "Constraints:\n"
        "- If the request is outside company policy, explain why and offer "
        "the closest supported alternative.\n"
        "- Use the customer's name placeholder.\n\n"
        "Output Format: Subject line, greeting, body, signature."
    )
    assert classify_unusable_output(legitimate) is None


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
    assert payload["repetition_penalty"] == DEFAULT_GENERATION_PARAMS["repetition_penalty"]


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
    assert len(ai.calls) == 2


async def test_generate_prompt_provider_error_is_not_retried():
    ai = FakeAIBinding(errors=Exception("Internal model failure"))
    with pytest.raises(AiServiceError) as exc_info:
        await generate_prompt(ai, "llama8b", "Write an email.")
    assert exc_info.value.status == 502
    assert exc_info.value.code == "AI_PROVIDER_ERROR"
    assert len(ai.calls) == 1


OBSERVED_EXAMPLE_RESPONSE_LEAK = """**Role:** Professional Job Application Email Writer
**Context:** Requesting a Meeting with Team
**Task:** Write a professional email to request a meeting with the team
**Output Format:** Email body with subject line and signature

**Example Response:**

Subject: Request for Meeting with Team

Dear Team,

I hope this email finds you well. I am reaching out to request a meeting
with the team to discuss some important updates on our current project.

Best regards,
[Your Name]"""


def test_classify_flags_example_response_leak():
    assert classify_unusable_output(OBSERVED_EXAMPLE_RESPONSE_LEAK) == "leak"


async def test_example_response_leak_triggers_corrective_regeneration():
    ai = FakeAIBinding(
        responses=[
            {"response": OBSERVED_EXAMPLE_RESPONSE_LEAK},
            {"response": CLEAN_PROMPT},
        ]
    )
    result = await generate_prompt(ai, "llama1b", "Write an email requesting a meeting.")
    assert result == CLEAN_PROMPT
    assert len(ai.calls) == 2
    second_user = ai.calls[1]["payload"]["messages"][1]["content"]
    assert second_user.endswith(CORRECTIVE_SUFFIXES["leak"])


async def test_leaked_answer_triggers_corrective_regeneration():
    ai = FakeAIBinding(
        responses=[{"response": OBSERVED_LABELLED_LEAK}, {"response": CLEAN_PROMPT}]
    )
    result = await generate_prompt(ai, "llama1b", "Write an email applying for a job.")
    assert result == CLEAN_PROMPT
    assert len(ai.calls) == 2
    first_user = ai.calls[0]["payload"]["messages"][1]["content"]
    second_user = ai.calls[1]["payload"]["messages"][1]["content"]
    assert CORRECTIVE_SUFFIXES["leak"] not in first_user
    assert second_user.endswith(CORRECTIVE_SUFFIXES["leak"])
    assert ai.calls[0]["payload"]["messages"][0]["content"] == SYSTEM_PROMPT


async def test_persistent_leak_is_rejected():
    ai = FakeAIBinding(responses={"response": OBSERVED_LABELLED_LEAK})
    with pytest.raises(AiServiceError) as exc_info:
        await generate_prompt(ai, "llama1b", "Write an email applying for a job.")
    assert exc_info.value.status == 502
    assert exc_info.value.code == "INVALID_AI_RESPONSE"
    assert len(ai.calls) == 2  # initial + one corrective regeneration


async def test_leading_preface_is_stripped_without_regeneration():
    ai = FakeAIBinding(
        responses={"response": "Here is the generated prompt:\n\n" + CLEAN_PROMPT}
    )
    result = await generate_prompt(ai, "llama1b", "Write an email.")
    assert result == CLEAN_PROMPT
    assert len(ai.calls) == 1


def test_build_messages_appends_corrective_suffix():
    plain = build_messages("Write an email.")
    corrective = build_messages("Write an email.", corrective="leak")
    assert plain[1]["content"] == corrective[1]["content"].removesuffix(
        CORRECTIVE_SUFFIXES["leak"]
    )
    assert corrective[0]["content"] == SYSTEM_PROMPT


def test_build_messages_unknown_corrective_is_ignored():
    plain = build_messages("Write an email.")
    corrective = build_messages("Write an email.", corrective="bogus")
    assert plain[1]["content"] == corrective[1]["content"]
