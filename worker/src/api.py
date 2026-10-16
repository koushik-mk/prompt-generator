
"""HTTP routing, request validation, and error mapping."""

import json

from ai_service import (
    MODEL_ALLOWLIST,
    AiServiceError,
    generate_prompt,
)
from prompt_engine import MAX_REQUEST_BYTES, MAX_SCENARIO_LENGTH


GENERATE_PATH = "/generate-prompt"

REQUEST_TOO_LARGE_MESSAGE = (
    "The request body is too large. Please shorten your scenario and try again."
)

INVALID_SCENARIO_MESSAGE = (
    "Please provide a scenario describing the task you need a prompt for."
)


class ValidationFailure(Exception):
    """An invalid request with its corresponding HTTP response details."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def error_payload(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def validate_request(body_text: str) -> tuple[str, str]:
    """Validate the JSON request and return (model_key, scenario)."""
    if len(body_text.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise ValidationFailure(
            413,
            "REQUEST_TOO_LARGE",
            REQUEST_TOO_LARGE_MESSAGE,
        )

    try:
        body = json.loads(body_text)
    except (json.JSONDecodeError, ValueError):
        raise ValidationFailure(
            400,
            "MALFORMED_JSON",
            "The request body is not valid JSON.",
        ) from None

    if not isinstance(body, dict):
        raise ValidationFailure(
            400,
            "MALFORMED_JSON",
            "The request body must be a JSON object.",
        )

    model_key = body.get("model")

    if not isinstance(model_key, str) or not model_key.strip():
        raise ValidationFailure(
            400,
            "INVALID_MODEL",
            "Please select a supported model: "
            + ", ".join(sorted(MODEL_ALLOWLIST))
            + ".",
        )

    model_key = model_key.strip()

    if model_key not in MODEL_ALLOWLIST:
        raise ValidationFailure(
            400,
            "INVALID_MODEL",
            "The selected model is not supported. Choose: "
            + ", ".join(sorted(MODEL_ALLOWLIST))
            + ".",
        )

    scenario = body.get("scenario")

    if not isinstance(scenario, str):
        raise ValidationFailure(
            400,
            "INVALID_SCENARIO",
            "The scenario must be a string.",
        )

    scenario = scenario.strip()

    if not scenario:
        raise ValidationFailure(
            400,
            "INVALID_SCENARIO",
            INVALID_SCENARIO_MESSAGE,
        )

    if len(scenario) > MAX_SCENARIO_LENGTH:
        raise ValidationFailure(
            400,
            "INVALID_SCENARIO",
            f"The scenario must not exceed {MAX_SCENARIO_LENGTH} characters.",
        )

    return model_key, scenario


async def handle_generate(get_ai_binding, body_text: str) -> tuple:
    """Handle POST /generate-prompt and return (status, JSON payload)."""
    try:
        model_key, scenario = validate_request(body_text)
    except ValidationFailure as exc:
        return exc.status, error_payload(exc.code, exc.message)

    ai_binding = get_ai_binding()

    if ai_binding is None:
        return 503, error_payload(
            "AI_UNAVAILABLE",
            "AI generation is temporarily unavailable. Please try again later.",
        )

    try:
        generated = await generate_prompt(
            ai_binding,
            model_key,
            scenario,
        )
    except AiServiceError as exc:
        return exc.status, error_payload(exc.code, exc.message)

    return 200, {
        "model": model_key,
        "generatedPrompt": generated,
    }


async def handle_request(
    method: str,
    path: str,
    body_text: str,
    get_ai_binding,
):
    """Route API requests; return None for paths handled by static assets."""
    if path == GENERATE_PATH:
        if method == "POST":
            return await handle_generate(get_ai_binding, body_text)

        if method == "OPTIONS":
            return 204, None

        return 405, error_payload(
            "METHOD_NOT_ALLOWED",
            f"Method {method} is not allowed on {GENERATE_PATH}. Use POST.",
        )

    if method in ("GET", "HEAD"):
        return None

    return 404, error_payload(
        "NOT_FOUND",
        f"No endpoint {method} {path}.",
    )
