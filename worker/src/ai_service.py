
"""Workers AI integration and error handling."""

import re

from prompt_engine import DEFAULT_GENERATION_PARAMS, build_messages


MODEL_ALLOWLIST = {
    "llama8b": "@cf/meta/llama-3.1-8b-instruct-fp8",
    "llama1b": "@cf/meta/llama-3.2-1b-instruct",
}

MAX_ATTEMPTS = 2
MAX_UNUSABLE_REGENERATIONS = 1

QUOTA_MESSAGE = (
    "The AI usage limit has been reached. "
    "Please try again after the usage allowance resets."
)
TEMPORARY_MESSAGE = (
    "AI generation is temporarily unavailable. Please try again later."
)
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
)

REFUSAL_PATTERN = re.compile(
    r"^\s*(?:i cannot|i can't|i am unable|i'm unable|"
    r"as an ai(?: language model)?)[\s,.:]",
    re.IGNORECASE,
)

COMMENTARY_PATTERN = re.compile(
    r"^\s*here(?:'s| is)\s+(?:the\s+|your\s+|a\s+)?"
    r"(?:generated\s+)?(?:prompt|response|result)\s*[:.]?\s*\n+",
    re.IGNORECASE,
)


class AiServiceError(Exception):
    """AI error with an HTTP status and a safe client-facing message."""

    def __init__(self, code: str, status: int, message: str):
        super().__init__(message)
        self.code = code
        self.status = status
        self.message = message


class UnsupportedModelError(Exception):
    """Raised when the requested model key is not supported."""

    def __init__(self, model_key):
        super().__init__(f"Unsupported model: {model_key!r}")
        self.model_key = model_key


def resolve_model_id(model_key: str) -> str:
    """Resolve a supported model key to its Workers AI model ID."""
    if not isinstance(model_key, str):
        raise UnsupportedModelError(model_key)

    try:
        return MODEL_ALLOWLIST[model_key]
    except KeyError:
        raise UnsupportedModelError(model_key) from None


def classify_ai_exception(exc: BaseException) -> str:
    """Classify an AI exception as quota, transient, or provider failure."""
    message = str(exc).lower()

    js_message = getattr(exc, "message", None)
    if js_message is not None:
        message += " " + str(js_message).lower()

    if any(marker in message for marker in QUOTA_MARKERS):
        return "quota"

    if any(marker in message for marker in TRANSIENT_MARKERS):
        return "transient"

    return "provider"


def _log_rejection(reason: str, text: str) -> None:
    """Log a short diagnostic when model output is rejected."""
    try:
        from js import console

        snippet = text.strip().replace("\n", " ")[:160]
        console.error(
            f"[prompt-generator] rejected output ({reason}): {snippet}"
        )
    except ImportError:
        pass


def extract_prompt(ai_response) -> str:
    """Extract a non-empty response from Workers AI."""
    if isinstance(ai_response, dict):
        raw = ai_response.get("response")
    elif isinstance(ai_response, (str, bytes)) or ai_response is None:
        raw = None
    else:
        raw = getattr(ai_response, "response", None)

    if not isinstance(raw, str) or not raw.strip():
        _log_rejection("empty or malformed response", str(raw))
        raise AiServiceError(
            "INVALID_AI_RESPONSE",
            502,
            MALFORMED_RESPONSE_MESSAGE,
        )

    return raw.strip()


def strip_leading_commentary(text: str) -> str:
    """Remove a simple introductory sentence if the model adds one."""
    previous = None

    while text != previous:
        previous = text
        text = COMMENTARY_PATTERN.sub("", text, count=1)

    return text.strip()


def classify_unusable_output(text: str) -> str | None:
    """Detect a few obvious output failures without brittle semantic rules."""
    stripped = text.strip()

    if not stripped:
        return "empty"

    # Detect conversational refusals, not refusal language inside a prompt.
    if REFUSAL_PATTERN.match(stripped):
        return "refusal"

    # Detect an exact sentence repeated excessively.
    counts = {}
    for line in stripped.splitlines():
        line = line.strip()
        if len(line) >= 30:
            counts[line] = counts.get(line, 0) + 1

    if any(count >= 3 for count in counts.values()):
        return "repetition"

    return None


def _to_js(payload: dict):
    """Convert Python dictionaries to JavaScript objects in Pyodide."""
    try:
        from js import Object
        from pyodide.ffi import to_js

        return to_js(payload, dict_converter=Object.fromEntries)
    except ImportError:
        return payload


async def generate_prompt(
    ai_binding,
    model_key: str,
    scenario: str,
) -> str:
    """Generate a prompt, with bounded retries for transient failures."""
    model_id = resolve_model_id(model_key)

    transient_attempts = 0
    unusable_regenerations = 0
    corrective = None

    while True:
        payload = {
            "messages": build_messages(scenario, corrective=corrective),
            **DEFAULT_GENERATION_PARAMS,
        }

        try:
            response = await ai_binding.run(
                model_id,
                _to_js(payload),
            )
            prompt = strip_leading_commentary(extract_prompt(response))

        except (AiServiceError, UnsupportedModelError):
            raise

        except Exception as exc:
            kind = classify_ai_exception(exc)

            if kind == "quota":
                raise AiServiceError(
                    "AI_QUOTA_EXCEEDED",
                    429,
                    QUOTA_MESSAGE,
                ) from exc

            if (
                kind == "transient"
                and transient_attempts < MAX_ATTEMPTS - 1
            ):
                transient_attempts += 1
                continue

            if kind == "transient":
                raise AiServiceError(
                    "AI_UNAVAILABLE",
                    503,
                    TEMPORARY_MESSAGE,
                ) from exc

            raise AiServiceError(
                "AI_PROVIDER_ERROR",
                502,
                PROVIDER_ERROR_MESSAGE,
            ) from exc

        unusable_kind = classify_unusable_output(prompt)

        if unusable_kind:
            _log_rejection(unusable_kind, prompt)

            if unusable_regenerations < MAX_UNUSABLE_REGENERATIONS:
                unusable_regenerations += 1
                corrective = (
                    unusable_kind
                    if unusable_kind in ("refusal", "repetition")
                    else "leak"
                )
                continue

            raise AiServiceError(
                "INVALID_AI_RESPONSE",
                502,
                MALFORMED_RESPONSE_MESSAGE,
            )

        return prompt
