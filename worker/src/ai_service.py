"""Workers AI integration: model resolution, invocation, response
validation, and error classification.

The AI binding is injected (dependency injection), so this module is fully
testable in CPython with a fake binding. It never imports `js`/`workers` at
module level; the optional Pyodide conversion in `_to_js` degrades to an
identity function outside the Workers runtime.

Integration verified against:
- Python Workers FFI docs (https://developers.cloudflare.com/workers/languages/python/ffi/)
- Official example: python-workers-examples/workers-ai
  (`await self.env.AI.run(model_id, {...})`)
- Workers AI error table (https://developers.cloudflare.com/workers-ai/platform/errors/)
"""

from prompt_engine import (
    DEFAULT_GENERATION_PARAMS,
    build_messages,
)
MODEL_ALLOWLIST = {
    # Friendly id -> verified Workers AI model id.
    # `@cf/meta/llama-3.1-8b-instruct` is no longer listed in the Workers AI
    # model catalog; the FP8-quantized variant is its current supported
    # listing (documented substitution, see README).
    "llama8b": "@cf/meta/llama-3.1-8b-instruct-fp8",
    "llama1b": "@cf/meta/llama-3.2-1b-instruct",
}

MAX_ATTEMPTS = 2  # one bounded retry, only for transient failures
MAX_LEAK_REGENERATIONS = 1  # one corrective retry when the model leaked the answer

# Markers indicating the model wrote the deliverable itself (or added
# commentary) instead of returning only the instruction prompt. Observed
# failure mode of small models (e.g. Llama 3.2 1B). Kept deliberately
# conservative to avoid false positives on legitimate prompts.
LEAK_MARKERS = (
    "here is the transformed prompt",
    "here is the generated prompt",
    "here is your prompt",
    "here is a prompt",
    "here's the",
    "i am writing to",
    "i hope this email finds you well",
    "dear [",
    "dear customer",
    "dear sir",
    "dear madam",
    "sincerely,\n[",
    "regards,\n[",
)

QUOTA_MESSAGE = (
    "The AI usage limit has been reached. "
    "Please try again after the usage allowance resets."
)
TEMPORARY_MESSAGE = "AI generation is temporarily unavailable. Please try again later."
MALFORMED_RESPONSE_MESSAGE = (
    "The AI service returned an unusable response. Please try again."
)
PROVIDER_ERROR_MESSAGE = (
    "The AI service could not process the request. Please try again later."
)

QUOTA_MARKERS = (
    "daily free allocation",
    "10,000 neurons",
    "used up your daily",
)
TRANSIENT_MARKERS = (
    "capacity temporarily exceeded",
    "rate limit",
    "too many requests",
    "timeout",
    "aborted",
    "temporarily unavailable",
    "try again",
)


class AiServiceError(Exception):
    """An AI failure already classified into an HTTP status and safe message."""

    def __init__(self, code: str, status: int, message: str):
        super().__init__(message)
        self.code = code
        self.status = status
        self.message = message


class UnsupportedModelError(Exception):
    """Raised when the requested friendly model id is not in the allowlist."""

    def __init__(self, model_key):
        super().__init__(f"Unsupported model: {model_key!r}")
        self.model_key = model_key


def resolve_model_id(model_key: str) -> str:
    """Map a friendly model identifier to a verified Workers AI model id."""
    if not isinstance(model_key, str):
        raise UnsupportedModelError(model_key)
    try:
        return MODEL_ALLOWLIST[model_key]
    except KeyError:
        raise UnsupportedModelError(model_key) from None


def classify_ai_exception(exc: BaseException) -> str:
    """Classify an exception raised by the AI binding.

    Returns one of: "quota", "transient", "provider".

    Classification is based on the documented Workers AI error table
    (messages such as "You have used up your daily free allocation of
    10,000 neurons" and "Capacity temporarily exceeded"). Quota markers are
    checked first because the quota message also contains "try again".
    """
    text = str(exc).lower()
    js_message = getattr(exc, "message", None)
    if js_message is not None:
        text += " " + str(js_message).lower()

    if any(marker in text for marker in QUOTA_MARKERS):
        return "quota"
    if any(marker in text for marker in TRANSIENT_MARKERS):
        return "transient"
    return "provider"


def extract_prompt(ai_response) -> str:
    """Validate the AI response structure and return the normalized prompt.

    Accepts both a plain dict (unit tests) and the JS proxy object returned
    by the AI binding in the Workers runtime (`response.response`).
    """
    if isinstance(ai_response, dict):
        raw = ai_response.get("response")
    elif isinstance(ai_response, (str, bytes)) or ai_response is None:
        raw = None
    else:
        raw = getattr(ai_response, "response", None)

    if raw is None:
        raise AiServiceError(
            "INVALID_AI_RESPONSE", 502, MALFORMED_RESPONSE_MESSAGE
        )

    if not isinstance(raw, str):
        # JS strings arrive as Python str through the FFI; anything else
        # (stream object, number, ...) is unusable.
        raise AiServiceError(
            "INVALID_AI_RESPONSE", 502, MALFORMED_RESPONSE_MESSAGE
        )

    prompt = raw.strip()
    if not prompt:
        raise AiServiceError(
            "INVALID_AI_RESPONSE", 502, MALFORMED_RESPONSE_MESSAGE
        )
    return prompt


def looks_like_leaked_answer(text: str) -> bool:
    """Detect model output that executed the scenario instead of prompting.

    A generated prompt must contain instructions for another AI. Outputs
    that greet a recipient, draft the email, or preface themselves with
    'Here is the ...' violate the core contract and are rejected.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in LEAK_MARKERS)


def _to_js(payload: dict):
    """Convert nested Python structures to plain JS objects in the runtime.

    Falls back to returning the dict unchanged in CPython (unit tests), so
    this module stays importable and testable outside Pyodide.
    """
    try:
        from js import Object
        from pyodide.ffi import to_js

        return to_js(payload, dict_converter=Object.fromEntries)
    except ImportError:
        return payload


async def generate_prompt(ai_binding, model_key: str, scenario: str) -> str:
    """Resolve the model, invoke Workers AI, and return the generated prompt.

    Failure budget (all bounded):
    - transient provider failures: at most one retry;
    - leaked answers (model wrote the deliverable itself): at most one
      corrective regeneration with an explicit correction instruction;
    - quota exhaustion and provider errors: never retried.

    A persistent leak is rejected as an unusable AI response (502) so a
    task answer is never shown to the user as a "generated prompt".
    """
    model_id = resolve_model_id(model_key)

    transient_attempts = 0
    leak_regenerations = 0
    corrective = False

    while True:
        payload = {
            "messages": build_messages(scenario, corrective=corrective),
            **DEFAULT_GENERATION_PARAMS,
        }
        try:
            ai_response = await ai_binding.run(model_id, _to_js(payload))
            prompt = extract_prompt(ai_response)
        except (AiServiceError, UnsupportedModelError):
            raise
        except Exception as exc:  # noqa: BLE001 - classify every binding failure
            kind = classify_ai_exception(exc)
            if kind == "transient" and transient_attempts < MAX_ATTEMPTS - 1:
                transient_attempts += 1
                continue
            if kind == "quota":
                raise AiServiceError(
                    "AI_QUOTA_EXCEEDED", 429, QUOTA_MESSAGE
                ) from exc
            if kind == "transient":
                raise AiServiceError(
                    "AI_UNAVAILABLE", 503, TEMPORARY_MESSAGE
                ) from exc
            raise AiServiceError(
                "AI_PROVIDER_ERROR", 502, PROVIDER_ERROR_MESSAGE
            ) from exc

        if looks_like_leaked_answer(prompt):
            if leak_regenerations < MAX_LEAK_REGENERATIONS:
                leak_regenerations += 1
                corrective = True
                continue
            raise AiServiceError(
                "INVALID_AI_RESPONSE", 502, MALFORMED_RESPONSE_MESSAGE
            )
        return prompt
