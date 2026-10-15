"""Cloudflare Python Worker entrypoint.

Thin runtime glue only: request/response conversion, CORS, static asset
delegation, and logging. All routing, validation, and AI logic lives in
pure-Python modules (api.py, ai_service.py, prompt_engine.py) that are
covered by CPython unit tests.

Runtime integration facts (verified against official docs/examples):
- Python Workers expose the Workers AI binding via the Pyodide FFI:
  `await self.env.AI.run(model_id, payload)` works from Python
  (python-workers-examples/workers-ai).
- `workers.Response.from_json(data, status=..., headers=dict)` builds JSON
  responses with custom status/headers (workers-runtime-sdk source).
"""

from urllib.parse import urlparse

from workers import Response, WorkerEntrypoint

from api import GENERATE_PATH, error_payload, handle_request


def _log_error(message: str) -> None:
    """Server-side diagnostic logging. Never logs scenario content or secrets."""
    try:
        from js import console

        console.error(message)
    except ImportError:
        pass


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        try:
            return await self._handle(request)
        except Exception as exc:  # noqa: BLE001 - last-resort handler
            _log_error(
                f"[prompt-generator] unexpected error: {type(exc).__name__}: {exc}"
            )
            return Response.from_json(
                error_payload(
                    "INTERNAL_ERROR",
                    "An unexpected error occurred. Please try again later.",
                ),
                status=500,
                headers={"content-type": "application/json"},
            )

    async def _handle(self, request):
        method = str(request.method).upper()
        path = urlparse(str(request.url)).path
        origin = self._request_origin(request)
        cors_headers = self._cors_headers(origin)

        # CORS preflight (the only method the API accepts besides POST)
        if method == "OPTIONS":
            return Response(status=204, headers=cors_headers)

        if path == GENERATE_PATH:
            body_text = await request.text()
            status, payload = await handle_request(
                method, path, body_text, self._ai_binding
            )
            headers = {"content-type": "application/json", **cors_headers}
            if payload is None:
                return Response(status=status, headers=cors_headers)
            return Response.from_json(payload, status=status, headers=headers)

        result = await handle_request(method, path, "", None)
        if result is None:
            # GET/HEAD: serve the built frontend through the assets binding.
            # The JS Response from ASSETS.fetch is returned as-is; the Workers
            # runtime accepts it directly (workers.Response refuses to wrap a
            # js.Response instance).
            assets = getattr(self.env, "ASSETS", None)
            if assets is not None:
                return await assets.fetch(request)
            return Response.from_json(
                error_payload("NOT_FOUND", f"No endpoint GET {path}."),
                status=404,
                headers={"content-type": "application/json"},
            )

        status, payload = result
        return Response.from_json(
            payload,
            status=status,
            headers={"content-type": "application/json", **cors_headers},
        )

    def _ai_binding(self):
        """Lazily resolve the Workers AI binding. Returns None if not configured."""
        return getattr(self.env, "AI", None)

    def _request_origin(self, request) -> str:
        try:
            origin = request.headers.get("Origin")
            return str(origin) if origin else ""
        except Exception:  # noqa: BLE001
            return ""

    def _cors_headers(self, origin: str) -> dict:
        """Reflect only explicitly allowed origins. No wildcard origins.

        Same-origin production traffic (frontend served from this worker's
        assets) needs no CORS headers at all; ALLOWED_ORIGINS exists for
        cross-origin setups such as the local Vite dev server.
        """
        if not origin:
            return {}
        allowed_var = getattr(self.env, "ALLOWED_ORIGINS", None)
        allowed = [o.strip() for o in str(allowed_var or "").split(",") if o.strip()]
        if origin not in allowed:
            return {}
        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Vary": "Origin",
        }
