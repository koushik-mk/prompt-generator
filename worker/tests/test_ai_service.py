import pytest

from ai_service import (
    MODEL_ALLOWLIST,
    AiServiceError,
    UnsupportedModelError,
    classify_ai_exception,
    classify_unusable_output,
    extract_prompt,
    generate_prompt,
    looks_like_degenerate_repetition,
    looks_like_refusal,
    resolve_model_id,
    strip_commentary,
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


OBSERVED_1B_LEAK = """**Role:** Email Writer
**Task:** Generate an email apologizing for a delayed product shipment

Here is the generated prompt:

"Dear [Customer],

I am writing to apologize for the delay in shipping your product. Please be
assured that your order will be delivered within the next [insert timeframe].

Best regards,
[AI System]"
"""

CLEAN_8B_PROMPT = """Role: Act as an experienced customer support specialist.

Context: A customer is waiting for a delayed shipment.

Task: Write a professional apology email. Use placeholders such as
[Customer Name] and [Product Name] for missing details.

Constraints: Apologetic and empathetic tone; no promotional content.

Output Format: A well-structured plain-text email."""

DEGENERATE_REPETITION = (
    "**Role:** HR Representative\n**Task:** Draft an email\n"
    + "**Constraints:** The email should be written in a format that is easy "
    "to understand and follow\n" * 8
)


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
    assert payload["repetition_penalty"] == 1.15


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


def test_detects_observed_leaked_answer():
    assert classify_unusable_output(OBSERVED_1B_LEAK) == "leak"


def test_detects_email_body_markers():
    for leaked in (
        "Dear [Customer Name], I am writing to ...",
        "I hope this email finds you well. ...",
        "Here is the transformed prompt: ...",
        "Here is the transformed scenario: ...",
    ):
        assert classify_unusable_output(leaked) == "leak"


def test_clean_prompt_is_not_flagged():
    assert classify_unusable_output(CLEAN_8B_PROMPT) is None


def test_looks_like_refusal_detects_conversational_answers():
    refusal = "I cannot write an email to the HR department. Is there something else I can help you with?"
    assert looks_like_refusal(refusal)
    assert classify_unusable_output(refusal) == "refusal"
    structured = CLEAN_8B_PROMPT + "\n\nConstraints: the target AI cannot use external tools."
    assert not looks_like_refusal(structured)
    assert classify_unusable_output(structured) is None


def test_short_unstructured_refusal_is_detected():
    assert classify_unusable_output("I can't do that. Is there something else I can help you with?") == "refusal"


def test_short_structured_refusal_is_not_flagged_conservatively():
    # Accepted miss: detection stays conservative so real prompts that quote
    # refusal phrases are never rejected. A dressed-up refusal that carries
    # structure headings passes the structural checks; the live semantic
    # suite covers meaning, not these markers.
    dressed = "Role: Email Writer\n\nI cannot write that email. I'm unable to help with this."
    assert classify_unusable_output(dressed) is None


def test_legitimate_prompt_with_refusal_example_not_flagged():
    legitimate = (
        "Role: Customer Support Specialist\n\n"
        "Task: Draft a professional response to the customer.\n\n"
        "Constraints:\n"
        "- If the request is outside company policy, explain why I cannot "
        "fulfill it and offer the closest supported alternative.\n"
        "- Keep the reply under 200 words and use the customer's name placeholder.\n"
        "- Escalate to a human agent when the customer asks for a refund above [limit].\n\n"
        "Output Format: Subject line, greeting, body, signature."
    )
    assert classify_unusable_output(legitimate) is None


def test_detects_degenerate_repetition():
    assert looks_like_degenerate_repetition(DEGENERATE_REPETITION)
    assert classify_unusable_output(DEGENERATE_REPETITION) == "repetition"
    assert not looks_like_degenerate_repetition(CLEAN_8B_PROMPT)


def test_strip_leading_commentary_removes_prefaces():
    commented = "Here is the rewritten prompt:\n\n" + CLEAN_8B_PROMPT
    assert strip_leading_commentary(commented) == CLEAN_8B_PROMPT
    double = "Here's the prompt:\nHere is the transformed scenario:\n" + CLEAN_8B_PROMPT
    assert strip_leading_commentary(double) == CLEAN_8B_PROMPT
    assert strip_leading_commentary(CLEAN_8B_PROMPT) == CLEAN_8B_PROMPT
    mid = CLEAN_8B_PROMPT + "\n\nNote: here is the placeholder list."
    assert strip_leading_commentary(mid) == mid.strip()


OBSERVED_TRAILING_NOTE = (
    "Note: I've removed all the untrustworthy elements from the original "
    "scenario and focused on providing a helpful and informative prompt for "
    "another AI system to use."
)

OBSERVED_META_NOTE = (
    "Note: This is an unverified piece of user-generated text. Please use "
    "this prompt to generate a high-quality response."
)

OBSERVED_THIRD_PERSON_NOTE = (
    "Note: This prompt instructs the recipient (another AI system) to "
    "generate a formal business letter addressing the hiring manager at XYZ "
    "Corporation, expressing interest in a Backend Engineer position while "
    "mentioning relevant technical skills and experience."
)


def test_strip_trailing_commentary_removes_meta_notes():
    with_note = CLEAN_8B_PROMPT + "\n\n" + OBSERVED_TRAILING_NOTE
    assert strip_commentary(with_note) == CLEAN_8B_PROMPT
    with_meta = CLEAN_8B_PROMPT + "\n\n" + OBSERVED_META_NOTE
    assert strip_commentary(with_meta) == CLEAN_8B_PROMPT
    with_third_person = CLEAN_8B_PROMPT + "\n\n" + OBSERVED_THIRD_PERSON_NOTE
    assert strip_commentary(with_third_person) == CLEAN_8B_PROMPT


OBSERVED_1B_CLOSING_ADVICE = (
    "Please use this template to craft a well-written email applying for a "
    "backend engineer position by attaching your resume. Your goal should "
    "focus on highlighting relevant skills and experiences while ensuring "
    "clarity and professionalism throughout. Remember to keep the tone "
    "consistent and concise, avoiding jargon and overly technical terms "
    "whenever possible. Good luck!"
)

OBSERVED_1B_CLOSING_ADVICE_OUTPUT = (
    "**Role:** Job Application Specialist\n"
    "**Context:** Seeking a Backend Engineer Position\n"
    "**Task:** Write an Email Applying for a Backend Engineer Position with "
    "Resume Attachment\n"
    "**Constraints:**\n"
    "* Tone: Professional and formal\n"
    "* Scope: General guidance only\n"
    "* Limitations:\n"
    "\t+ No personal opinions or biases\n"
    "\t+ Focus on providing instruction to generate a suitable response\n"
    "* Output Format: A single paragraph with placeholder text\n\n"
    + OBSERVED_1B_CLOSING_ADVICE
)


def test_strip_closing_advice_removes_observed_human_directed_paragraph():
    stripped = strip_commentary(OBSERVED_1B_CLOSING_ADVICE_OUTPUT)
    assert stripped.endswith("* Output Format: A single paragraph with placeholder text")
    assert "Good luck" not in stripped
    assert "Please use this template" not in stripped


def test_strip_closing_advice_removes_short_chatty_closings():
    with_advice = CLEAN_8B_PROMPT + "\n\n" + OBSERVED_1B_CLOSING_ADVICE
    assert strip_commentary(with_advice) == CLEAN_8B_PROMPT
    with_luck = CLEAN_8B_PROMPT + "\n\nHope this helps! Good luck!"
    assert strip_commentary(with_luck) == CLEAN_8B_PROMPT
    wrapped = (
        CLEAN_8B_PROMPT
        + "\n\nPlease use this template to craft the email.\n"
        "Remember to keep the tone consistent. Good luck!"
    )
    assert strip_commentary(wrapped) == CLEAN_8B_PROMPT


def test_strip_closing_advice_keeps_structured_final_sections():
    heading_first = (
        CLEAN_8B_PROMPT
        + "\n\nOutput Format: Use this template for the final email and wish the reader good luck."
    )
    assert strip_commentary(heading_first) == heading_first.strip()
    instruction = (
        CLEAN_8B_PROMPT
        + "\n\nConstraints: Keep the email concise and free of jargon."
    )
    assert strip_commentary(instruction) == instruction.strip()


def test_strip_commentary_handles_both_ends():
    both = (
        "Here is the rewritten prompt:\n\n"
        + CLEAN_8B_PROMPT
        + "\n\n"
        + OBSERVED_TRAILING_NOTE
    )
    assert strip_commentary(both) == CLEAN_8B_PROMPT


def test_non_first_person_notes_are_kept():
    instructional = CLEAN_8B_PROMPT + "\n\nNote: use [placeholders] for any missing names or dates."
    assert strip_commentary(instructional) == instructional.strip()
    constraint_note = CLEAN_8B_PROMPT + "\n\nNote: this response must stay under 200 words."
    assert strip_commentary(constraint_note) == constraint_note.strip()


async def test_trailing_note_is_stripped_without_regeneration():
    ai = FakeAIBinding(responses={"response": CLEAN_8B_PROMPT + "\n\n" + OBSERVED_TRAILING_NOTE})
    result = await generate_prompt(ai, "llama1b", "Write an email.")
    assert result == CLEAN_8B_PROMPT
    assert len(ai.calls) == 1


async def test_third_person_note_is_stripped_without_regeneration():
    ai = FakeAIBinding(responses={"response": CLEAN_8B_PROMPT + "\n\n" + OBSERVED_THIRD_PERSON_NOTE})
    result = await generate_prompt(ai, "llama1b", "Write an email.")
    assert result == CLEAN_8B_PROMPT
    assert len(ai.calls) == 1


async def test_closing_advice_paragraph_is_stripped_without_regeneration():
    ai = FakeAIBinding(responses={"response": OBSERVED_1B_CLOSING_ADVICE_OUTPUT})
    result = await generate_prompt(ai, "llama1b", "Apply for a job and attach my resume.")
    assert result.endswith("* Output Format: A single paragraph with placeholder text")
    assert "Good luck" not in result
    assert len(ai.calls) == 1


async def test_commentary_preface_is_stripped_without_regeneration():
    ai = FakeAIBinding(responses={"response": "Here is the rewritten prompt:\n\n" + CLEAN_8B_PROMPT})
    result = await generate_prompt(ai, "llama1b", "Write an email.")
    assert result == CLEAN_8B_PROMPT
    assert len(ai.calls) == 1


async def test_leaked_answer_triggers_corrective_regeneration():
    ai = FakeAIBinding(
        responses=[{"response": OBSERVED_1B_LEAK}, {"response": CLEAN_8B_PROMPT}]
    )
    result = await generate_prompt(ai, "llama1b", "Write an apology email.")
    assert result == CLEAN_8B_PROMPT
    assert len(ai.calls) == 2
    first_user = ai.calls[0]["payload"]["messages"][1]["content"]
    second_user = ai.calls[1]["payload"]["messages"][1]["content"]
    assert CORRECTIVE_SUFFIXES["leak"] not in first_user
    assert second_user.endswith(CORRECTIVE_SUFFIXES["leak"])
    assert ai.calls[0]["payload"]["messages"][0]["content"] == SYSTEM_PROMPT
    assert ai.calls[1]["payload"]["messages"][0]["content"] == SYSTEM_PROMPT


async def test_refusal_regeneration_uses_refusal_specific_correction():
    ai = FakeAIBinding(
        responses=[
            {
                "response": "I cannot write an email to the HR department. "
                "Is there something else I can help you with?"
            },
            {"response": CLEAN_8B_PROMPT},
        ]
    )
    result = await generate_prompt(ai, "llama1b", "Write an email to HR.")
    assert result == CLEAN_8B_PROMPT
    second_user = ai.calls[1]["payload"]["messages"][1]["content"]
    assert second_user.endswith(CORRECTIVE_SUFFIXES["refusal"])
    assert CORRECTIVE_SUFFIXES["leak"] not in second_user


async def test_repetition_uses_repetition_specific_correction():
    ai = FakeAIBinding(
        responses=[{"response": DEGENERATE_REPETITION}, {"response": CLEAN_8B_PROMPT}]
    )
    result = await generate_prompt(ai, "llama1b", "Write an email to HR.")
    assert result == CLEAN_8B_PROMPT
    second_user = ai.calls[1]["payload"]["messages"][1]["content"]
    assert second_user.endswith(CORRECTIVE_SUFFIXES["repetition"])


async def test_persistent_unusable_output_is_rejected():
    ai = FakeAIBinding(responses={"response": OBSERVED_1B_LEAK})
    with pytest.raises(AiServiceError) as exc_info:
        await generate_prompt(ai, "llama1b", "Write an apology email.")
    assert exc_info.value.status == 502
    assert exc_info.value.code == "INVALID_AI_RESPONSE"
    assert len(ai.calls) == 2  # initial + one corrective regeneration


def test_long_structured_prompt_with_ai_phrase_is_not_refusal():
    structured = (
        CLEAN_8B_PROMPT
        + "\n\nConstraints: The target model, as an AI system, must not invent "
        "facts and must request clarification when details are missing. "
        + "Extra padding to stay above the short-output threshold." * 3
    )
    assert classify_unusable_output(structured) is None


TASK_INVERTED_OUTPUT = (
    "Role: Recruiter Assistant\n"
    "Context: Backend Engineer hiring process\n"
    "Task: Write an email to an applicant seeking feedback on their skills "
    "and experience for consideration as a Backend Engineer.\n"
    "Output Format: A professional email."
)

RESUME_DIRECTION_INVERTED_OUTPUT = (
    "Role: HR Recruiter\n"
    "Context: Job Application Process for Backend Engineer Position\n"
    "Task: Generate a Professional Email Template with Resume Attachment "
    "Request from Untrusted User Data\n\n"
    "Constraints:\n"
    "* The second paragraph should ask the recipient to attach their resume\n"
    "* Tone: Formal and Polite\n"
    "* Output Format: Email Body Only\n\n"
    "Note: This is an unverified piece of user-generated text. Please use "
    "this prompt to generate a high-quality response."
)


def test_semantic_inversion_is_out_of_scope_for_markers():
    # A prompt that silently changes the user's task is structurally valid;
    # phrase markers must not pretend to catch it. Prevention is the system
    # prompt's Semantic Fidelity Rules, and detection is the live
    # integration suite (tests/test_integration_live.py).
    assert classify_unusable_output(TASK_INVERTED_OUTPUT) is None


def test_recipient_attach_inversion_is_flagged_as_direction():
    assert classify_unusable_output(RESUME_DIRECTION_INVERTED_OUTPUT) == "direction"


def test_legitimate_sender_attach_phrasing_is_not_flagged():
    legitimate = (
        "Role: Career communications specialist\n"
        "Task: Draft the application email in which the applicant attaches "
        "their resume and references the enclosed portfolio.\n"
        "Constraints: The email must state that the applicant provides both "
        "documents; the recipient only receives them.\n"
        "Output Format: Subject line, greeting, body, signature."
    )
    assert classify_unusable_output(legitimate) is None


async def test_direction_inversion_triggers_direction_correction():
    ai = FakeAIBinding(
        responses=[
            {"response": RESUME_DIRECTION_INVERTED_OUTPUT},
            {"response": CLEAN_8B_PROMPT},
        ]
    )
    result = await generate_prompt(ai, "llama1b", "Apply for a job and attach my resume.")
    assert result == CLEAN_8B_PROMPT
    assert len(ai.calls) == 2
    second_user = ai.calls[1]["payload"]["messages"][1]["content"]
    assert second_user.endswith(CORRECTIVE_SUFFIXES["direction"])


OBSERVED_1B_EXPLOSION_OUTPUT = (
    "* Role: Professional Job Applicant\n"
    "* Context: Applying for a Backend Engineer Position\n"
    "* Task: Drafting an Email with Resume Attachment\n"
    "* Constraints:\n"
    "\t+ Tone: Formal and Objective\n"
    "\t+ Output Format: Single-Line Instruction\n"
    '* Constraints: Remains from original scenario except for "attach"\n'
    "* Constraints: Preserves original sentence structure and word order\n"
    "* Constraints: Avoids inserting context or clarifications\n"
    "* Constraints: Does not attempt to provide further guidance\n"
    "* Constraints: Only uses provided placeholders [Company Name] and [Job Title]\n"
    "* Constraints: Preserves original specificity regarding Company Name\n"
    "* Constraints: Removes any extraneous phrases or expressions\n"
    "* Constraints: Respects original formatting and punctuation\n"
    "* Constraints: Uses exact wording and phrasing\n"
    "* Constraints: Maintains consistency with original scenario\n"
    "* Constraints: Returns only the specified output, without adding new content\n"
    "* Constraints: Preserves original scope and limitations\n"
    "* Constraints: Addresses the target AI as the doer of the work\n"
    "* Constraints: Ensures clarity and precision in the generated prompt\n"
    "* Constraints: Correctly identifies the deliverable as an email with attachment\n"
    "* Constraints: Specifies the required format (single line)\n"
    "* Constraints: Includes specific placeholders for [Company Name] and [Job Title]\n"
    "* Constraints: Preserves original emphasis on professionalism and formality\n"
)

PERSONA_ROLE_OUTPUT = (
    "**Role:** Job Applicant\n"
    "**Context:** Applying for a Backend Engineer position\n"
    "**Task:** Draft the application email with the resume referenced in the body\n"
    "**Output Format:** Subject line and body\n"
)


def test_constraint_explosion_is_flagged():
    assert classify_unusable_output(OBSERVED_1B_EXPLOSION_OUTPUT) == "explosion"


def test_persona_role_is_flagged():
    assert classify_unusable_output(PERSONA_ROLE_OUTPUT) == "persona"


def test_producer_roles_and_bounded_bullets_are_not_flagged():
    specialist = (
        "Role: Candidate communications specialist\n"
        "Task: Draft the application email.\n"
        "Constraints:\n"
        "- Keep it under 200 words.\n"
        "- Use [Company Name] for the employer.\n"
        "- Reference the enclosed resume in the body.\n"
        "- Ask for clarification when the hiring manager's name is missing.\n"
        "- Keep the tone formal.\n"
        "- End with a signature block.\n"
        "Output Format: Subject line and body.\n"
    )
    assert classify_unusable_output(specialist) is None
    assert classify_unusable_output(CLEAN_8B_PROMPT) is None


async def test_constraint_explosion_triggers_explosion_correction():
    ai = FakeAIBinding(
        responses=[
            {"response": OBSERVED_1B_EXPLOSION_OUTPUT},
            {"response": CLEAN_8B_PROMPT},
        ]
    )
    result = await generate_prompt(ai, "llama1b", "Apply for a job and attach my resume.")
    assert result == CLEAN_8B_PROMPT
    assert len(ai.calls) == 2
    second_user = ai.calls[1]["payload"]["messages"][1]["content"]
    assert second_user.endswith(CORRECTIVE_SUFFIXES["explosion"])


async def test_persona_role_triggers_persona_correction():
    ai = FakeAIBinding(
        responses=[{"response": PERSONA_ROLE_OUTPUT}, {"response": CLEAN_8B_PROMPT}]
    )
    result = await generate_prompt(ai, "llama1b", "Apply for a job and attach my resume.")
    assert result == CLEAN_8B_PROMPT
    assert len(ai.calls) == 2
    second_user = ai.calls[1]["payload"]["messages"][1]["content"]
    assert second_user.endswith(CORRECTIVE_SUFFIXES["persona"])


def test_build_messages_corrective_variant():
    plain = build_messages("Write an email.")
    corrective = build_messages("Write an email.", corrective="leak")
    refusal = build_messages("Write an email.", corrective="refusal")
    assert plain[1]["content"] == corrective[1]["content"].removesuffix(CORRECTIVE_SUFFIXES["leak"])
    assert refusal[1]["content"].endswith(CORRECTIVE_SUFFIXES["refusal"])
    assert refusal[1]["content"] != corrective[1]["content"]
    assert corrective[0]["content"] == SYSTEM_PROMPT
