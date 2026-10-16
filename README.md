# Prompt Generator

A web application that transforms user-provided scenarios into
structured, reusable prompts for AI models. Rather than completing the
requested task, the application generates clear instructions that
another AI model can follow.

**Live Demo:**
https://prompt-generator.koushiknaidukoushiknaidu.workers.dev

## Features

-   **AI-powered prompt generation:** Converts natural-language
    scenarios into structured prompts with relevant roles, context,
    tasks, constraints, and output formats.
-   **Model selection:** Supports two text-generation models through
    Cloudflare Workers AI.
-   **Input validation:** Validates scenarios and request payloads
    before inference.
-   **Output validation:** Detects selected unusable responses and
    attempts corrective generation when appropriate.
-   **Error handling:** Handles invalid inputs, AI provider failures,
    service availability issues, and quota exhaustion.
-   **Responsive interface:** Includes loading indicators, clear error
    messages, and copy-to-clipboard functionality.
-   **Resource management:** Limits request sizes and generated output
    length and uses bounded retries.

## Architecture

The application uses a React frontend and a Python backend running on
Cloudflare Workers. The Worker serves the built frontend as static
assets and exposes the prompt-generation API.

### Request flow

``` text
React frontend
      |
      | POST /generate-prompt
      v
entry.py
Routing, CORS, and static assets
      |
      v
api.py
Request validation and error mapping
      |
      v
prompt_engine.py
System prompt and message construction
      |
      v
ai_service.py
Model selection, inference, and output validation
      |
      v
Cloudflare Workers AI binding
```

The backend is divided into modules with distinct responsibilities:

-   `entry.py` handles Cloudflare Worker runtime integration, routing,
    CORS, and static assets.
-   `api.py` validates requests and maps errors to HTTP responses.
-   `prompt_engine.py` defines the system prompt and constructs model
    messages.
-   `ai_service.py` manages model selection, inference, and output
    checks.

Most application logic can be tested with standard Python tooling.
Runtime-specific integration is handled separately because the Worker
entry point depends on Cloudflare's Python runtime.

## Technology Stack

| Component         | Technology                          |
|-------------------|-------------------------------------|
| Frontend          | React 18, Vite 6, CSS               |
| Backend           | Python on Cloudflare Workers        |
| AI inference      | Cloudflare Workers AI               |
| Backend testing   | pytest, pytest-asyncio              |
| Frontend testing  | Vitest, Testing Library             |
| Deployment        | Cloudflare Workers and static assets|

## Getting Started

### Prerequisites

-   Node.js 18 or later
-   `uv` 0.12.3 or later
-   A Cloudflare account with access to Workers AI
-   Wrangler authentication for local development and deployment

Authenticate with Cloudflare if you have not already done so:

``` bash
npx wrangler login
```

### Run the frontend

``` bash
cd frontend
npm install
```

On Windows PowerShell, create the development environment file:

``` powershell
Copy-Item .env.example .env.development.local
```

Start the Vite development server:

``` bash
npm run dev
```

The frontend runs at `http://localhost:5173`.

### Run the backend

In a separate terminal:

``` bash
cd worker
uv sync
uv run pywrangler dev
```

The Worker runs at `http://localhost:8787`.

`pywrangler` is the Python Workers launcher for the Wrangler CLI: it
runs the same local `wrangler dev` development server with the Python
runtime enabled.

For local development, configure the frontend API base URL and allowed
origin as required by the project configuration.

### Run the application with the built frontend

To test the same-origin setup used in production:

``` bash
cd frontend
npm run build

cd ../worker
uv run pywrangler dev
```

Open `http://localhost:8787`.

The Worker configuration references the built frontend assets in
`frontend/dist`.

## Project Structure

``` text
prompt-generator/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── PromptForm
│   │   │   ├── GeneratedPrompt
│   │   │   ├── ErrorMessage
│   │   │   └── LoadingIndicator
│   │   ├── tests/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── models.js
│   │   └── validation.js
│   ├── .env.example
│   └── vite.config.js
├── worker/
│   ├── src/
│   │   ├── entry.py
│   │   ├── api.py
│   │   ├── prompt_engine.py
│   │   └── ai_service.py
│   ├── tests/
│   ├── pyproject.toml
│   └── wrangler.jsonc
├── .gitignore
└── README.md
```

## API Reference

### `POST /generate-prompt`

Generates a prompt from a scenario using the selected model.

**Request body**

``` json
{
  "model": "llama8b",
  "scenario": "I need an email to apologize for a delayed product shipment."
}
```

**Successful response**

``` json
{
  "model": "llama8b",
  "generatedPrompt": "Role: Act as an experienced customer support specialist..."
}
```

**Error response**

``` json
{
  "error": {
    "code": "INVALID_SCENARIO",
    "message": "The scenario must not exceed 5000 characters."
  }
}
```

### HTTP status codes

| Status | Error code | Description |
|--------|------------|-------------|
| 400  | `MALFORMED_JSON`                            | Invalid JSON request body                  |
| 400  | `INVALID_MODEL`                             | Missing or unsupported model               |
| 400  | `INVALID_SCENARIO`                          | Missing, blank, invalid, or oversized scenario |
| 404  | `NOT_FOUND`                                 | Unknown endpoint                           |
| 405  | `METHOD_NOT_ALLOWED`                        | Unsupported HTTP method                    |
| 413  | `REQUEST_TOO_LARGE`                         | Request body exceeds 64 KiB                |
| 429  | `AI_QUOTA_EXCEEDED`                         | AI provider quota has been exhausted       |
| 502  | `INVALID_AI_RESPONSE` / `AI_PROVIDER_ERROR` | Unusable model response or provider error  |
| 503  | `AI_UNAVAILABLE`                            | AI service or capacity is unavailable      |
| 500  | `INTERNAL_ERROR`                            | Unexpected server error                    |

The backend enforces validation rules. Frontend validation provides
immediate feedback but is not treated as a security boundary.

## Supported Models

  Application ID   Cloudflare Workers AI model
  ---------------- --------------------------------------
  `llama8b`        `@cf/meta/llama-3.1-8b-instruct-fp8`
  `llama1b`        `@cf/meta/llama-3.2-1b-instruct`

The implementation uses `@cf/meta/llama-3.1-8b-instruct-fp8` for the 8B
option. Model identifiers are managed through an allowlist. Requests
using unsupported identifiers are rejected before inference, and
selecting a model in the frontend determines which configured model
receives the request.

## Prompt Engineering Approach

The prompt-generation logic is centralized in
`worker/src/prompt_engine.py`.

The system prompt guides the model to:

-   Preserve the user's original intent and requested deliverable.
-   Generate instructions for another AI model rather than complete the
    scenario itself.
-   Include relevant role, context, task, constraints, and output-format
    details.
-   Avoid introducing unsupported facts or unnecessary requirements.
-   Use placeholders where important information is missing.
-   Keep generated prompts clear and actionable.

The application separates the scenario from the system instructions and
identifies scenario content as untrusted input. This helps reduce the
risk of user-provided text overriding the intended behavior.

This structure was chosen for small instruction-tuned models. The rules
are numbered and strongly worded because 1B and 8B models follow
explicit prohibitions more reliably than long prose. The most important
rules are negative ones: the model must never perform the task itself
and must never draft the deliverable, because the most common failure
for models of this size is to answer the scenario directly instead of
writing a prompt. The scenario is wrapped in `USER SCENARIO START` and
`USER SCENARIO END` markers and treated as data so that embedded
instructions cannot override the system rules. Output validation acts as
a final safety net.

Output checks handle selected failure cases, such as deliverable
leakage, conversational refusals, or repetitive text. When appropriate,
the service makes one bounded corrective generation attempt that appends
a targeted correction to the user message. These checks address known
response-quality issues but do not guarantee that every generated prompt
will be correct.

Smaller models such as the 1B option can still invent plausible details
that the scenario did not provide; the 8B model follows the constraints
more reliably.

## Error Handling and Resource Usage

The application validates requests before invoking the AI service and
uses bounded retry behavior for selected transient failures and unusable
outputs.

Provider errors are mapped to application-level responses so the
frontend can display understandable messages. Quota exhaustion is
handled separately from temporary availability issues.

Additional measures include:

-   Disabling the Generate button while a request is in progress.
-   Preventing duplicate submissions during an active request.
-   Avoiding inference on every keystroke.
-   Limiting input size and generated output length.
-   Avoiding retries for quota-exhaustion errors.

The application does not display an estimated remaining neuron balance
because it does not retrieve that value from the provider.

## Testing

### Backend

``` bash
cd worker
uv sync
uv run pytest -q
```

The backend test suite is designed to run against a mocked AI binding so
application logic is tested without consuming inference quota. It targets
model selection, request validation, error classification,
prompt-injection handling, and output validation. The full suite runs
offline and is expected to pass without network access.

### Frontend

``` bash
cd frontend
npm install
npm run test
```

The frontend tests cover rendering, validation, loading and error
states, duplicate-submission prevention, model selection, and clipboard
functionality.

Runtime-specific behavior should be tested separately from unit tests,
including Worker startup, API routing, CORS behavior, and inference
through the configured Workers AI binding.

## Deployment

Build the frontend and deploy the Worker:

``` bash
cd frontend
npm run build

cd ../worker
uv run pywrangler deploy
```

The frontend must be rebuilt before deployment so that the latest files
are included in the Worker assets.

The frontend uses `VITE_API_BASE_URL` to configure the API endpoint. An
empty value allows requests to use the same origin. If the frontend is
hosted separately, configure the API base URL and allow the frontend
origin in the Worker configuration.
