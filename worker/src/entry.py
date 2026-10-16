
"""Cloudflare Python Worker entry point."""

from urllib.parse import urlparse

from workers import Response, WorkerEntrypoint

from api import (
    GENERATE_PATH,
    REQUEST_TOO_LARGE_MESSAGE,
    error_payload,
    handle_request,
)
from prompt_engine import MAX_REQUEST_BYTES


def _log_error(message: str) -> None:
    """Log unexpected server errors."""
    try:
        from js import console

        console.error(message)
    except ImportError:
        pass


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        try:
            return await self._handle(request)
        except Exception as exc:
            _log_error(
                f"[prompt-generator] unexpected error: "
                f"{type(exc).__name__}: {exc}"
            )
            return Response.from_json(
                error_payload(
                    "INTERNAL_ERROR",
                    "An unexpected error occurred. Please try again later.",
                ),
                status=500,
                headers={"content-type": "application/json"},
            )

    def _json(self, payload, status: int, cors_headers: dict):
        return Response.from_json(
            payload,
            status=status,
            headers={
                "content-type": "application/json",
                **cors_headers,
            },
        )

    async def _read_body_capped(self, request) -> tuple[str, bool]:
        """Read at most MAX_REQUEST_BYTES plus one byte of request data."""
        length = request.headers.get("Content-Length")

        if length:
            try:
                if int(length) > MAX_REQUEST_BYTES:
                    return "", True
            except (TypeError, ValueError):
                pass

        body = request.body

        if body is None:
            return "", False

        reader = body.getReader()
        chunks = []
        total = 0

        try:
            while True:
                part = await reader.read()

                if part.done:
                    break

                chunk = bytes(part.value)
                total += len(chunk)

                if total > MAX_REQUEST_BYTES:
                    await reader.cancel()
                    return "", True

                chunks.append(chunk)

        finally:
            # The reader is either exhausted or cancelled on overflow.
            pass

        try:
            return b"".join(chunks).decode("utf-8"), False
        except UnicodeDecodeError:
            return "", False

    async def _handle(self, request):
        method = str(request.method).upper()
        path = urlparse(str(request.url)).path

        origin = self._request_origin(request)
        cors_headers = self._cors_headers(origin)

        if method == "OPTIONS":
            return Response(status=204, headers=cors_headers)

        if path == GENERATE_PATH:
            body_text = ""

            if method == "POST":
                body_text, oversize = await self._read_body_capped(request)

                if oversize:
                    return self._json(
                        error_payload(
                            "REQUEST_TOO_LARGE",
                            REQUEST_TOO_LARGE_MESSAGE,
                        ),
                        413,
                        cors_headers,
                    )

            status, payload = await handle_request(
                method,
                path,
                body_text,
                self._ai_binding,
            )

            if payload is None:
                return Response(status=status, headers=cors_headers)

            return self._json(payload, status, cors_headers)

        result = await handle_request(method, path, "", None)

        if result is None:
            assets = getattr(self.env, "ASSETS", None)

            if assets is not None:
                return await assets.fetch(request)

            return self._json(
                error_payload(
                    "NOT_FOUND",
                    f"No endpoint {method} {path}.",
                ),
                404,
                cors_headers,
            )

        status, payload = result
        return self._json(payload, status, cors_headers)

    def _ai_binding(self):
        return getattr(self.env, "AI", None)

    def _request_origin(self, request) -> str:
        try:
            origin = request.headers.get("Origin")
            return str(origin) if origin else ""
        except Exception:
            return ""

    def _cors_headers(self, origin: str) -> dict:
        """Allow only explicitly configured frontend origins."""
        if not origin:
            return {}

        allowed_var = getattr(self.env, "ALLOWED_ORIGINS", None)
        allowed = [
            item.strip()
            for item in str(allowed_var or "").split(",")
            if item.strip()
        ]

        if origin not in allowed:
            return {}

        return {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Vary": "Origin",
        }
