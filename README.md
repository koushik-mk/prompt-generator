# Prompt Generator

A small web app that takes a scenario, like "I need an email to apologize for a delayed product shipment," and turns it into a polished prompt you can paste into any AI system. It does not answer the scenario itself. It writes the instructions another model should follow. That distinction is the whole point of the project, so a fair amount of the code below exists just to enforce it.

Deployed and working: https://prompt-generator.koushiknaidukoushiknaidu.workers.dev

## What it does

You pick a model, describe what you need in a textarea, and hit Generate. A few seconds later you get back something structured (role, context, task, constraints, output format) that you can copy and hand to ChatGPT, Gemini, whatever. The UI shows a spinner while the request is in flight, disables the button so you can't double submit, and gives you a copy button plus clear error messages when something goes wrong. It works fine on a phone too.

## How it fits together

The backend is a Python Cloudflare Worker. The same worker also serves the built React app as static assets, so in production everything lives on one origin and there is no CORS to think about. Locally you run Vite on port 5173 and the worker on 8787, and the worker allows that one origin explicitly.

The request flow is boring on purpose:

```
browser -> POST /generate-prompt -> entry.py (routing, CORS, assets)
        -> api.py (validation, error mapping)
        -> prompt_engine.py (system prompt + message building)
        -> ai_service.py (model allowlist, AI.run, output checks)
        -> Workers AI binding (self.env.AI)
```

I split it this way because entry.py can only run inside the Pyodide runtime (it imports `workers` and `js`), which means pytest can't touch it. Everything with actual logic sits in plain Python modules that import on a normal CPython, so the test suite covers routing, validation, and AI handling without spinning up a runtime. entry.py is left as glue: parse the request, call `handle_request`, convert the tuple it returns into a `Response`.

On the Python-and-Workers-AI question: I checked the docs before writing anything. Python Workers expose bindings through the Pyodide FFI, and Cloudflare's own `python-workers-examples/workers-ai` sample calls `await self.env.AI.run(model, {...})` straight from Python. So there is no JavaScript shim and no REST fallback. The only wrinkle is that nested dicts need `pyodide.ffi.to_js(..., dict_converter=Object.fromEntries)` before being passed to the binding, and JSON responses with custom status codes come from `workers.Response.from_json(data, status=..., headers=...)`. Both are in the code and both are verified against the published SDK source.

## Stack

| Part | Choice |
|---|---|
| Frontend | React 18 with Vite 6, plain CSS |
| Backend | Python 3 on the Cloudflare Workers Python runtime |
| AI | Cloudflare Workers AI through the `AI` binding |
| Tests | pytest + pytest-asyncio for the worker, Vitest + Testing Library for the UI |
| Deploy | one Worker carrying the API and the static assets |

## What you need

Node 18 or newer, uv at least 0.12.3 (pywrangler refuses older versions), and a Cloudflare account. That last one is needed even for local dev, because the AI binding talks to the real service even when you run `pywrangler dev` on your laptop. `npx wrangler login` once and you're set.

## Layout

```
prompt-generator/
├── frontend/            react app
│   ├── src/
│   │   ├── components/  PromptForm, GeneratedPrompt, ErrorMessage, LoadingIndicator
│   │   ├── tests/       vitest suites
│   │   ├── App.jsx      page layout and shared state
│   │   ├── api.js       fetch wrapper and error mapping
│   │   ├── models.js    friendly names mapped to backend ids
│   │   └── validation.js
│   ├── .env.example
│   └── vite.config.js
├── worker/              python worker
│   ├── src/
│   │   ├── entry.py         runtime glue
│   │   ├── api.py           validation and error mapping
│   │   ├── prompt_engine.py the system prompt and message construction
│   │   └── ai_service.py    allowlist, inference, output guards
│   ├── tests/
│   ├── pyproject.toml
│   └── wrangler.jsonc
├── .gitignore
└── README.md
```

## Running it locally

```bash
cd frontend
npm install
Copy-Item .env.example .env.development.local
npm run dev              # http://localhost:5173

cd ../worker
uv sync
uv run pywrangler dev    # http://localhost:8787
```

Or skip Vite entirely: run `npm run build` in frontend and just start the worker, then open http://localhost:8787. That path uses the same-origin setup production uses.

The worker config (`wrangler.jsonc`) sets `compatibility_flags` to `python_workers`, binds Workers AI as `AI`, points `assets` at `../frontend/dist` with `run_worker_first` so the Python code sees every request first, and carries an `ALLOWED_ORIGINS` var for the cross-origin dev case. Nothing secret lives in the file.

The frontend reads one variable, `VITE_API_BASE_URL`. Empty means same origin. Worth knowing: I originally put the dev URL in `.env.local` and Vite happily baked it into the production bundle, which made the deployed site try to call my laptop. It now lives in `.env.development.local`, which Vite only loads in dev mode.

## The API

`POST /generate-prompt` with JSON:

```json
{ "model": "llama8b", "scenario": "I need an email to apologize for a delayed product shipment." }
```

200:

```json
{ "model": "llama8b", "generatedPrompt": "Role: Act as an experienced customer support specialist..." }
```

Errors all share one shape:

```json
{ "error": { "code": "INVALID_SCENARIO", "message": "The scenario must not exceed 5000 characters." } }
```

| Status | Code | Meaning |
|---|---|---|
| 400 | MALFORMED_JSON | body isn't valid JSON or isn't an object |
| 400 | INVALID_MODEL | model missing or not in the allowlist |
| 400 | INVALID_SCENARIO | missing, blank, not a string, or over 5000 chars |
| 404 | NOT_FOUND | unknown path on a non-GET method |
| 405 | METHOD_NOT_ALLOWED | anything but POST/OPTIONS on the endpoint |
| 413 | REQUEST_TOO_LARGE | body over 64 KiB |
| 429 | AI_QUOTA_EXCEEDED | the provider says the daily allowance is gone |
| 502 | INVALID_AI_RESPONSE / AI_PROVIDER_ERROR | unusable model output / provider failure |
| 503 | AI_UNAVAILABLE | capacity issue, or the AI binding isn't configured |
| 500 | INTERNAL_ERROR | anything I didn't anticipate |

Backend validation is the authority. The frontend validates for UX only. Request size is enforced twice: the worker checks Content-Length as an early hint and then reads the body as a capped stream, cancelling as soon as 64 KiB is exceeded so an oversized body is never fully buffered, and `validate_request` re-checks the decoded length as the authoritative limit.

## Models

| Id in the UI | Workers AI model | Notes |
|---|---|---|
| llama8b | @cf/meta/llama-3.1-8b-instruct-fp8 | the good one |
| llama1b | @cf/meta/llama-3.2-1b-instruct | faster and cheaper, noticeably less careful |

The assessment suggested `@cf/meta/llama-3.1-8b-instruct`, but that id isn't in the current Workers AI catalog anymore. The FP8 variant is the listing that exists today (same model family, 32k context), so that's what the allowlist maps to. The 1B is there as-is. The allowlist is the only place model ids appear: unknown ids get a 400 before any inference happens, so nobody can make the worker run an arbitrary model, and switching the dropdown genuinely switches the model (there are tests on both sides for that).

## The system prompt

All of it lives in `worker/src/prompt_engine.py`, nowhere else, as a numbered rule set. It tells the model it is a prompt engineer that must never perform the scenario, and walks it through seven obligations: preserve the intended action, deliverable, audience, and outcome without swapping tasks or reversing sender and recipient; assign a professional role that describes the AI's job rather than the user's identity ("Professional Job Application Email Writer", not "Job Applicant"); write actionable instructions so an email request becomes "draft the email" instead of a human-facing guide or tip list; preserve facts, using descriptive placeholders such as `[Company Name]` and never inventing names, qualifications, or dates; include only constraints the user stated or the task genuinely needs, dropping duplicates and asking for clarification when requirements conflict; format the result as prompt-only output with useful headings (Role, Context, Task, Constraints, Tone, Output Format), where the Task section names the user's task and the Output Format section describes the deliverable (subject line plus body for an email); and ignore injected instructions. A final self-check line asks the model to verify intent, deliverable, audience, and facts survived before returning.

Communication tasks get explicit direction rules inside those obligations: a resume the sender attaches stays a document the email mentions, never a document the recipient is asked to provide, and the generated prompt must not imply the target AI actually sent mail or attached files, because it only produces text. The scenario reaches the model in its own user message between `USER SCENARIO START` and `USER SCENARIO END` markers, labelled untrusted input to transform, with "Return only the generated prompt" repeated after it (small models listen to the last thing they read). Tests feed it "ignore all previous instructions" payloads against that quarantine.

Why the heading set: the role gives the target model a persona and therefore a register and vocabulary; context passes along audience and background so nothing has to be guessed; the task is where the user's actual intent survives; constraints stop the downstream model from overreaching on tone and scope; the format makes the result usable without rework. If you ever see the app output the apology email itself instead of instructions for one, that is a bug, and the runtime guards in `ai_service.py` exist to catch exactly that.

## Error handling

Provider errors get classified against Cloudflare's documented Workers AI error table. Text about the daily free allocation of 10,000 neurons maps to 429 with a plain-language quota message. Capacity, rate limit, and timeout text maps to 503 with "temporarily unavailable." Everything else is a 502. Nothing gets retried except transient failures (one retry) and unusable model output (one corrective regeneration, with a correction tailored to what went wrong, after I measured that two bought nothing extra against the neuron budget). Quota and provider errors never retry.

Speaking of unusable output: the 1B model will occasionally write the email instead of the prompt, answer conversationally ("I cannot help with that"), or repeat one sentence until the token cap cuts it off. All three happened during real testing. So the service inspects what comes back: it strips chatty prefaces ("Here is the generated prompt:"), flags conversational refusals anchored at the start of the output, and flags degenerate repetition when one 30+ character line appears three or more times. A flagged output regenerates once with a correction keyed to the failure kind (leak, refusal, or repetition). If it still fails after that, the user gets a clean error instead of garbage. `repetition_penalty` is sent on every request to reduce the looping in the first place.

## Neurons

Free accounts get 10,000 neurons a day. The app doesn't burn them carelessly: no inference until a request passes validation, the button disables mid-request and an in-flight ref blocks duplicates, nothing generates on keystrokes, output is capped at 768 tokens, and retries are bounded as above. There is no fake quota counter in the app and no claim about your remaining balance, because there's no supported way to read it. Real quota errors from the provider get the friendly 429 message.

## Tests

```bash
cd worker && uv sync && uv run pytest -q    # AI binding mocked, zero neurons
cd frontend && npm install && npm run test  # 14 tests
```

The pytest suite mocks the AI binding completely, so running it costs zero neurons: through a fake binding it exercises request validation, routing status codes, quota/transient/provider classification, the injection quarantine, and the output guards. The frontend suite covers rendering, validation, loading and error states, duplicate submission, model selection reaching the request body, and both clipboard paths. Because entry.py needs Pyodide, the runtime behaviour was verified separately: `pywrangler deploy --dry-run` for the build, local dev-server smoke runs for routing and CORS (including oversized bodies arriving with and without Content-Length), and real requests against the deployed worker for both models. Actual quota exhaustion was never triggered on purpose, so that path is unit-tested only.

## Deploying

```bash
cd frontend && npm run build
cd ../worker && uv run pywrangler deploy
```

The current deployment was verified end to end after login: the SPA loads over HTTPS, bad requests return the documented JSON errors, and both models return real generated prompts. Build the frontend before every deploy since the worker uploads `frontend/dist` as assets. If you'd rather host the frontend somewhere else, set `VITE_API_BASE_URL` at build time and add that origin to `ALLOWED_ORIGINS`.
