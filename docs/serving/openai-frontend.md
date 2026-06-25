(openai-frontend)=
# OpenAI-compatible frontend

`set_openai_frontend()` configures a serving function with OpenAI-compatible REST endpoints in a single call. It registers the path templates, input body mappings (extracting fields from the request), and output body mappings (filtering/reshaping the response) for each supported operation group — letting the official OpenAI Python SDK invoke an MLRun serving function as if it were OpenAI.

Built on top of the {ref}`API handler<api-handler>` — `set_openai_frontend()` produces an `APIHandlerConfig` under the hood. You can also configure additional endpoints alongside the OpenAI ones.

## Supported endpoint groups

Selected via the `OpenAIEndpoint` enum:

| Enum value | OpenAI operation group | Paths registered                                                                                                                                                                                          |
|---|---|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OpenAIEndpoint.CHAT_COMPLETIONS` | `/chat/completions` | POST `/chat/completions`, GET `/chat/completions`, GET / POST / DELETE `/chat/completions/{completion_id}`, GET `/chat/completions/{completion_id}/messages`                                              |
| `OpenAIEndpoint.RESPONSES` | `/responses` | POST `/responses`, GET / DELETE `/responses/{response_id}`, GET `/responses/{response_id}/input_items`, POST `/responses/input_tokens`, POST `/responses/{response_id}/cancel`, POST `/responses/compact` |

Each group ships pre-built input and output body mappings that:
- Extract the documented OpenAI request fields from the request body as keyword arguments (e.g. `model`, `messages`, `input`, `instructions`, …).
- Filter the handler's response down to the OpenAI response contract so the SDK's typed deserializers (`ChatCompletion`, `Response`, …) accept it.

Mandatory fields (per the OpenAI spec) are enforced via `mandatory=True` on the relevant mappings — missing fields fail the request with HTTP 422 (Unprocessable Entity).

## Quick start

```python
import mlrun
from mlrun.serving.openai_mappings import OpenAIEndpoint

fn = mlrun.code_to_function(
    name="openai-frontend",
    kind="serving",
    filename="openai_handler.py",
    image="mlrun/mlrun",
)

# Register every supported endpoint group, no prefix
fn.set_openai_frontend()

# Or: only the responses group, behind the standard /v1 prefix
fn.set_openai_frontend([OpenAIEndpoint.RESPONSES], prefix="/v1")
```

## Path prefix

The `prefix=` argument prepends a path segment to every registered endpoint. Most OpenAI clients send requests under `/v1/` — use `prefix="/v1"` to match:

```python
fn.set_openai_frontend(prefix="/v1")
# Registers /v1/chat/completions, /v1/responses, /v1/responses/{response_id}, …
```

The prefix is optional; if provided, it must start with `/` (e.g. `/v1`, `/v2`) — a missing leading slash raises `MLRunInvalidArgumentError`.

## Dispatcher handler (`include_url_info=True`)

A flow graph terminates in a single handler, so to serve multiple endpoint groups (or multiple methods on the same path template — e.g. `GET` vs `DELETE` on `/responses/{response_id}`) you add a small dispatcher step that routes by request path and HTTP method. Enable `include_url_info=True` on the `APIHandlerConfig` so `mlrun_request_path` and `mlrun_request_method` are injected into the handler:

```python
from mlrun.serving.endpoint_mapping import APIHandlerConfig

fn.set_api_handler_config(APIHandlerConfig(include_url_info=True))
fn.set_openai_frontend()  # merged into the existing config

graph = fn.set_topology("flow", engine="sync")
graph.to(name="router", handler="openai_router").respond()
```

The router selects the right handler per path + method:

```python
# openai_handler.py
def openai_router(body, mlrun_request_path, mlrun_request_method, **kwargs):
    if mlrun_request_method == "POST" and mlrun_request_path == "/chat/completions":
        return chat_completion_handler(body, **kwargs)
    if mlrun_request_method == "POST" and mlrun_request_path == "/responses":
        return response_handler(body, **kwargs)
    raise RuntimeError(f"no handler for {mlrun_request_method} {mlrun_request_path}")
```

See {ref}`URL info <api-handler>` for the full `include_url_info` contract.

## Invoking from the OpenAI Python SDK

Point the SDK at the deployed function's URL — no other changes needed:

```python
import httpx
import openai

import mlrun

client = openai.OpenAI(
    base_url=fn.get_url(),
    api_key="dummy",  # MLRun doesn't enforce a key; pass anything non-empty
    http_client=httpx.Client(verify=mlrun.mlconf.httpdb.http.verify),
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
)
```

The SDK deserializes the response into a typed `ChatCompletion` object — the output body mappings filter extra fields and enforce mandatory ones to match what the SDK expects.

## Custom endpoints alongside

`set_openai_frontend()` is additive — it merges its endpoints into the existing `APIHandlerConfig`. You can add custom endpoints (e.g. health checks, admin) on top:

```python
from http import HTTPMethod
from mlrun.common.schemas.serving import APIHandlerAction

fn.set_openai_frontend(prefix="/v1")
config = APIHandlerConfig.from_dict(fn.spec.api_handler_config)
config.add_endpoint_handler("/health", HTTPMethod.GET, APIHandlerAction.ALLOW)
fn.set_api_handler_config(config)
```

## Returning errors

Raise an `mlrun.errors.MLRunHTTPStatusError` subclass to surface a precise HTTP status code (e.g. `MLRunNotFoundError` → 404). To return an OpenAI-shaped error envelope (so the SDK deserializes it as a typed error), return `Response(body={"error": {...}}, status_code=4xx, content_type="application/json")` — the output body mapping is skipped on non-2xx, so the body and status code pass through to the caller intact. See {ref}`Returning a custom HTTP status code <api-handler>` for details.
