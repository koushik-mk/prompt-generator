"""HTTP request-handling logic for the Prompt Generator API.

Pure Python (no js/workers imports): `entry.py` is a thin runtime wrapper
around `handle_request`, which makes routing, validation, and error mapping
directly unit-testable in CPython with a mocked AI binding.
"""

import json

from ai_service import (
    MODEL_ALLOWLIST,
    AiServiceError,
    UnsupportedModelError,
    generate_prompt,
)
from prompt_engine import MAX_REQUEST_BYTES, MAX_SCENARIO_LENGTH

GENERATE_PATH = "/generate-prompt"


class ValidationFailure(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def error_payload(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def validate_request(body_text: str) -> tuple:
    """Parse and validate the request body. Returns (model_key, scenario)."""
    if len(body_text.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise ValidationFailure(
            413,
            "REQUEST_TOO_LARGE",
            "The request body is too large. Please shorten your scenario and try again.",
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
            "Please select a model. Supported models: "
            + ", ".join(sorted(MODEL_ALLOWLIST))
            + ".",
        )
    model_key = model_key.strip()
    if model_key not in MODEL_ALLOWLIST:
        # Enforced up-front so arbitrary model ids are rejected before any
        # other processing. ai_service.resolve_model_id re-checks as
        # defense-in-depth.
        raise ValidationFailure(
            400,
            "INVALID_MODEL",
            "The selected model is not supported. Supported models: "
            + ", ".join(sorted(MODEL_ALLOWLIST))
            + ".",
        )

    scenario = body.get("scenario")
    if scenario is None:
        raise ValidationFailure(
            400,
            "INVALID_SCENARIO",
            "Please provide a scenario describing the task you need a prompt for.",
        )
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
            "Please provide a scenario describing the task you need a prompt for.",
        )
    if len(scenario) > MAX_SCENARIO_LENGTH:
        raise ValidationFailure(
            400,
            "INVALID_SCENARIO",
            f"Please enter a scenario containing no more than {MAX_SCENARIO_LENGTH} characters.",
        )

    return model_key, scenario


async def handle_generate(get_ai_binding, body_text: str) -> tuple:
    """Handle POST /generate-prompt. Returns (status_code, payload).

    `get_ai_binding` is a zero-argument callable so the AI binding is only
    resolved after the request passes validation (and so a missing binding
    produces a clean 503 instead of an internal error).
    """
    try:
        model_key, scenario = validate_request(body_text)
    except ValidationFailure as failure:
        return failure.status, error_payload(failure.code, failure.message)

    ai_binding = get_ai_binding()
    if ai_binding is None:
        return 503, error_payload(
            "AI_UNAVAILABLE",
            "AI generation is temporarily unavailable. Please try again later.",
        )

    try:
        generated = await generate_prompt(ai_binding, model_key, scenario)
    except UnsupportedModelError:
        return 400, error_payload(
            "INVALID_MODEL",
            "The selected model is not supported. Supported models: "
            + ", ".join(sorted(MODEL_ALLOWLIST))
            + ".",
        )
    except AiServiceError as exc:
        return exc.status, error_payload(exc.code, exc.message)

    return 200, {"model": model_key, "generatedPrompt": generated}


async def handle_request(method: str, path: str, body_text: str, get_ai_binding) -> tuple:
    """Route a request. Returns (status_code, payload).

    A payload of None with status 204 means "respond with no body" (CORS
    preflight). A return of None means "not an API route" and the caller
    decides (static assets or 404).
    """
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
        # Static asset candidate; the caller serves it via the ASSETS binding.
        return None

    return 404, error_payload(
        "NOT_FOUND",
        f"No endpoint {method} {path}. The only API endpoint is POST {GENERATE_PATH}.",
    )
