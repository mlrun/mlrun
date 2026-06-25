(api-handler)=
# API handler

```{admonition} Note
This feature is in TechPreview status; there will be changes to the SDK in a future release. 
```

An API handler is a graph step that is automatically prepended to a serving graph when configured. It validates incoming HTTP requests against a set of user-defined endpoints, extracts parameters from path templates, query strings, and the request body, and passes them to the next step in the graph.

Use an API handler to:
- Implement industry-defined REST API schemas on your serving graph (for example, the OpenAI chat-completion interface for LLMs).
- Gate access to specific paths using ALLOW and FORBID rules.

API handlers are active only for HTTP-triggered invocations. When an event arrives through a non-HTTP trigger such as a stream, the API handler is bypassed (the path is always `/` in that case).

**Supported runtimes**: Serving functions with an HTTP trigger, and the mock server (local testing).

## Overview

When the `GraphServer` receives an HTTP event and an API handler is configured, it runs the handler step before forwarding the event to the graph. The handler:

1. Matches the request's HTTP method and URL path against the configured endpoint list.
2. If a match is found:
   - Extracts path template parameters and query string parameters.
   - Optionally applies JSONPath transformations on the request body (`body_map`).
   - Optionally injects the normalized request path into the event (`include_url_info`).
   - Passes the enriched event to the graph root.
3. If no match is found, the handler fails the request with an appropriate HTTP error (404 for unknown paths, 405 for method not allowed, 403 for FORBID action).

## Configuration

### APIHandlerConfig

`APIHandlerConfig` holds the full configuration for the API handler. Create one, add endpoint rules and optional body mappings, then attach it to the serving function.

```python
from http import HTTPMethod

from mlrun.common.schemas.serving import APIHandlerAction
from mlrun.runtimes.nuclio.serving import APIHandlerConfig

config = APIHandlerConfig()

# Allow GET /v1/models
config.add_endpoint_handler("/v1/models", HTTPMethod.GET, APIHandlerAction.ALLOW)

# Allow POST /v1/models/{model_name}/predict  (path template)
config.add_endpoint_handler(
    "/v1/models/{model_name}/predict",
    HTTPMethod.POST,
    APIHandlerAction.ALLOW,
    description="Run inference on a named model",
)

# Forbid access to admin endpoints
config.add_endpoint_handler("/admin/*", HTTPMethod.GET, APIHandlerAction.FORBID)

# Attach configuration to the serving function
serving_fn.set_api_handler_config(config)
```

The configuration is serialized into `serving_fn.spec.api_handler_config` and is picked up at deployment time.

### `add_endpoint_handler` signature

```python
config.add_endpoint_handler(
    path,  # URL path, e.g. "/v1/chat" or "/v1/chat/*"
    http_method=HTTPMethod.POST,  # HTTPMethod enum or string ("GET", "POST", ...)
    action=APIHandlerAction.ALLOW,  # ALLOW or FORBID
    description=None,  # Optional human-readable description
    input_body_mappings=None,  # BodyMappings instance (see Body mapping section)
)
```

To remove an endpoint:

```python
config.remove_endpoint_handler("/v1/models", HTTPMethod.GET)
```

## Path matching rules

Endpoints are matched in the following priority order:

1. **Exact paths** — no wildcards or template parameters, e.g. `/v1/models`.
2. **Path templates** — contain `{param}` placeholders, e.g. `/v1/models/{model_name}/predict`. Matched with a pre-compiled regex; insertion order wins when multiple templates match the same path.
3. **Wildcard paths** — end with `*`, e.g. `/admin/*`. The `*` must be at the end and appear only once. Matched by prefix; the request path must contain at least one segment after the prefix. Insertion order wins.

**Examples:**

| Configured path           | Matches                             | Does not match   |
|---------------------------|-------------------------------------|------------------|
| `/v1/models`              | `/v1/models`                        | `/v1/models/gpt` |
| `/v1/models/{model_name}` | `/v1/models/gpt`, `/v1/models/bert` | `/v1/models`     |
| `/v1/*`                   | `/v1/chat`, `/v1/models/gpt`        | `/v1`            |

## Extracting request parameters

When the handler allows a request, it can extract parameters from three sources:

| Source                   | How to configure                 | Available as               |
|--------------------------|----------------------------------|----------------------------|
| Path template parameters | `{param}` in the path pattern    | keyword argument           |
| Query string parameters  | automatic                        | keyword argument           |
| Request body fields      | `BodyMappings` (JSONPath, see below) | keyword argument       |

The extracted parameters are passed to the next step as keyword arguments. If the same parameter name appears in more than one source, the request fails with an error (400). Conflicts between `body_map` and path template parameters are detected at setup time.

Parameters are always forwarded to the next step. When any parameters are extracted or `include_url_info` is enabled, they are collected into a dict and passed as keyword arguments. Otherwise, the original event body is forwarded unchanged.

### Body mapping (`BodyMappings`)

`BodyMappings` maps destination parameter names to [JSONPath](https://datatracker.ietf.org/doc/html/rfc9535) source expressions. Each endpoint has its own `BodyMappings` instance passed via `input_body_mappings`. The request body must be a JSON dict when body mappings are configured.

```python
from mlrun.runtimes.nuclio.serving import BodyMappings

# Build a body mapping for a specific endpoint
bm = BodyMappings()
bm.add_mapping(destination_path="model_name", source_json_path="$.model")
bm.add_mapping(destination_path="inputs", source_json_path="$.data.inputs")

# Multiple matches (e.g. all book titles) return a list
bm.add_mapping(
    destination_path="titles", source_json_path="$['store']['book'][*]['title']"
)

# Attach the mapping to the endpoint
config.add_endpoint_handler(
    "/v1/predict",
    HTTPMethod.POST,
    APIHandlerAction.ALLOW,
    input_body_mappings=bm,
)
```

`add_mapping` parameters:

| Parameter          | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `destination_path` | Name of the keyword argument passed to the next step.                       |
| `source_json_path` | JSONPath expression evaluated against the request body dict.                |
| `mandatory`        | If `True` (default `False`), a missing field fails the request with the error code HTTP 400 - bad request. |

Rules:
- A single JSONPath match → the value is returned as-is.
- Multiple matches → a list is returned.
- No match on a **mandatory** field → the request fails with HTTP 400 (Bad Request).
- No match on an optional field → the parameter is silently omitted.
- Non-dict body → body mappings are silently skipped.
- Calling `add_mapping` with a duplicate `destination_path` or `source_json_path` overwrites the existing entry and logs a warning.

To remove a mapping by destination path: `bm.remove_mapping("model_name")` — where `"model_name"` is the `destination_path`.

#### Hierarchical body map merging

When a request matches multiple endpoints (for example, a wildcard `/*` and a specific `/v1/predict`), their `input_body_mappings` are **merged** from least specific to most specific. The most specific endpoint wins on conflict:

- **Same destination** — the more specific endpoint's source overwrites the less specific one.
- **Same source, different destination** — the stale destination from the less specific endpoint is removed; only the more specific endpoint's destination is kept.

This allows a wildcard endpoint to define shared defaults while specific endpoints override individual mappings:

```python
# Wildcard: shared defaults for all POST endpoints under /v1/
star_bm = BodyMappings()
star_bm.add_mapping(
    destination_path="model", source_json_path="$.model", mandatory=True
)
star_bm.add_mapping(destination_path="stream", source_json_path="$.stream")
config.add_endpoint_handler(
    "/v1/*", HTTPMethod.POST, APIHandlerAction.ALLOW, input_body_mappings=star_bm
)

# Specific endpoint: inherits "stream" from wildcard, overrides "model" → "model_name"
predict_bm = BodyMappings()
predict_bm.add_mapping(destination_path="model_name", source_json_path="$.model")
predict_bm.add_mapping(destination_path="messages", source_json_path="$.messages")
config.add_endpoint_handler(
    "/v1/chat/completions",
    HTTPMethod.POST,
    APIHandlerAction.ALLOW,
    input_body_mappings=predict_bm,
)
# POST /v1/chat/completions effective mapping:
#   model_name ← $.model   (specific wins; "model" destination from wildcard is dropped)
#   stream     ← $.stream  (inherited from wildcard)
#   messages   ← $.messages (specific only)
```

### URL info (`include_url_info`)

When `include_url_info=True`, the handler injects two additional fields into the event:

- `mlrun_request_path` — the normalized, URL-decoded path (without the query string). Decoding matches Flask/FastAPI semantics: an encoded slash (`%2F`) in a segment becomes indistinguishable from a path separator.
- `mlrun_request_method` — the HTTP method as an uppercase string (e.g. `"GET"`, `"DELETE"`).

Both are passed together so a dispatcher handler can distinguish endpoints that share a path template but differ by method. Query string parameters are always extracted as keyword arguments regardless of this setting.

```python
config = APIHandlerConfig(include_url_info=True)
config.add_endpoint_handler(
    "/v1/chat/completions/{completion_id}", HTTPMethod.GET, APIHandlerAction.ALLOW
)
```

A `GET /v1/chat/completions/abc123?limit=10` request passes the following keyword arguments to the next step:

```python
{
    "completion_id": "abc123",  # from path template
    "limit": "10",  # from query string
    "mlrun_request_path": "/v1/chat/completions/abc123",  # from include_url_info
    "mlrun_request_method": "GET",  # from include_url_info
}
```

Dispatch by method on a shared path template:

```python
def responses_router(
    body, response_id, mlrun_request_path, mlrun_request_method, **kwargs
):
    if mlrun_request_method == "GET":
        return get_response(response_id)
    if mlrun_request_method == "DELETE":
        return delete_response(response_id)
    raise ValueError(f"unsupported method {mlrun_request_method}")
```

The handler signature must accept these names (explicitly or via `**kwargs`); otherwise Python raises `TypeError: unexpected keyword argument`.

## Returning a custom HTTP status code

A handler can return a `Response(body, status_code, ...)` wrapper to set a custom HTTP response. See [Returning custom HTTP responses](./serving-graph.md#returning-custom-http-responses) for the construction patterns — this section covers the **interaction with `output_body_mappings`**.

`output_body_mappings` describes the *success-shape* contract, so the mapping runs **only when `status_code < 300`**. Non-2xx responses pass through with their body and status code intact, so the caller sees the original error envelope instead of a synthetic 422 from a failed mandatory-field check.

If you simply return a `dict`, the runtime treats the response as `200 OK` and runs the output mapping as usual — no change from previous behavior.

## How downstream steps receive parameters

Extracted parameters are passed as keyword arguments to the handler function or `do()` method. For example, given the endpoint `/v1/chat/completions/{completion_id}` with `include_url_info=True`, a `GET /v1/chat/completions/abc123?limit=10` request calls the next step as:

```python
def step_handler(
    body,
    completion_id,
    limit,
    mlrun_request_path,
    mlrun_request_method,
    **kwargs,
):
    # body: original request body
    # completion_id="abc123"                            — from path template
    # limit="10"                                        — from query string
    # mlrun_request_path="/v1/chat/completions/abc123"  — from include_url_info
    # mlrun_request_method="GET"                        — from include_url_info
    ...


class MyStep:
    def do(
        self,
        body,
        completion_id,
        limit,
        mlrun_request_path,
        mlrun_request_method,
        **kwargs,
    ): ...
```

## Complete example

The following example configures a serving function with an API handler that supports an OpenAI-compatible `POST /v1/chat/completions` endpoint. It extracts the `model` field and `messages` array from the request body, and makes the request path available to downstream steps.

```python
import mlrun
from http import HTTPMethod

from mlrun.common.schemas.serving import APIHandlerAction
from mlrun.runtimes.nuclio.serving import APIHandlerConfig, BodyMappings

project = mlrun.get_or_create_project("chat-serving", context="./")

serving_fn = project.set_function(
    name="chat-completion",
    kind="serving",
    image="mlrun/mlrun",
)


# --- Define the graph step that processes chat completions ---
# model_name and messages are passed as keyword arguments by the API handler
def chat_handler(model_name, messages, mlrun_request_path, **kwargs):
    reply = f"Received {len(messages)} message(s) for model '{model_name}'"
    return {"reply": reply, "path": mlrun_request_path}


graph = serving_fn.set_topology("flow", engine="sync")
graph.to(name="chat", handler=chat_handler).respond()

# --- Configure the API handler ---
config = APIHandlerConfig(include_url_info=True)

# Extract model and messages from the request body using JSONPath
bm = BodyMappings()
bm.add_mapping(
    destination_path="model_name", source_json_path="$.model", mandatory=True
)
bm.add_mapping(
    destination_path="messages", source_json_path="$.messages", mandatory=True
)

# Allow the OpenAI-compatible chat completion endpoint
config.add_endpoint_handler(
    "/v1/chat/completions",
    HTTPMethod.POST,
    APIHandlerAction.ALLOW,
    description="OpenAI-compatible chat completion",
    input_body_mappings=bm,
)

# Block all admin paths
config.add_endpoint_handler("/admin/*", HTTPMethod.GET, APIHandlerAction.FORBID)
config.add_endpoint_handler("/admin/*", HTTPMethod.POST, APIHandlerAction.FORBID)

# Attach to the serving function
serving_fn.set_api_handler_config(config)

# --- Test locally with the mock server ---
server = serving_fn.to_mock_server()

# Allowed endpoint: body mappings extract model_name and messages; chat_handler receives them as kwargs
result = server.test(
    "/v1/chat/completions",
    method="POST",
    body={"model": "my-llm", "messages": [{"role": "user", "content": "Hello"}]},
)
# result: {"reply": "Received 1 message(s) for model 'my-llm'", "path": "/v1/chat/completions"}

# Forbidden endpoint: raises 403
try:
    server.test("/admin/settings", method="GET", body={})
except Exception as e:
    print(e)  # Access forbidden

# Unknown endpoint: raises 404
try:
    server.test("/unknown", method="GET", body={})
except Exception as e:
    print(e)  # Endpoint not found
```
