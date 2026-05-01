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
| Request body fields      | `body_map` (JSONPath, see below) | keyword argument           |

The extracted parameters are passed to the next step as keyword arguments. If the same parameter name appears in more than one source, the request fails with an error (400). Conflicts between `body_map` and path template parameters are detected at setup time.

Parameters are always forwarded to the next step. When any parameters are extracted or `include_url_info` is enabled, they are collected into a dict and passed as keyword arguments. Otherwise, the original event body is forwarded unchanged.

### Body mapping (`body_map`)

`body_map` maps parameter names to [JSONPath](https://datatracker.ietf.org/doc/html/rfc9535) expressions. It is **global** — the same mapping applies to every configured endpoint. The request body must be a JSON dict when `body_map` is configured.

```python
# Extract "model" field and nested "inputs" field from the request body
config.add_body_mapping("model_name", "$.model")
config.add_body_mapping("inputs", "$.data.inputs")

# Multiple matches (e.g. all book titles) return a list
config.add_body_mapping("titles", "$['store']['book'][*]['title']")
```

Rules:
- A single match → the value is returned as-is.
- Multiple matches → a list is returned.
- No match → the request fails with HTTP 422 (Unprocessable Entity).
- Non-JSON or non-dict body → the request fails with HTTP 422 (Unprocessable Entity).
- `body_map` applies to **all** ALLOW endpoints in the config. Because the mapping is global, a `body_map` that matches one endpoint's body format may cause 422 errors for other endpoints whose requests do not share that format.

```{note}
The strict 422 behavior for missing body mappings and non-JSON bodies is a known limitation in the current release and will be relaxed in a future release to allow more flexible body handling.
```

To remove a mapping: `config.remove_body_mapping("model_name")`

### URL info (`include_url_info`)

When `include_url_info=True`, the handler injects an additional field `mlrun_request_path` into the event. This field contains the normalized URL path (without the query string). Query string parameters are always extracted as keyword arguments regardless of this setting.

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
}
```

## How downstream steps receive parameters

Extracted parameters are passed as keyword arguments to the handler function or `do()` method. For example, given the endpoint `/v1/chat/completions/{completion_id}` with `include_url_info=True`, a `GET /v1/chat/completions/abc123?limit=10` request calls the next step as:

```python
def step_handler(body, completion_id, limit, mlrun_request_path, **kwargs):
    # body: original request body
    # completion_id="abc123"                            — from path template
    # limit="10"                                        — from query string
    # mlrun_request_path="/v1/chat/completions/abc123"  — from include_url_info
    ...


class MyStep:
    def do(self, body, completion_id, limit, mlrun_request_path, **kwargs): ...
```

## Complete example

The following example configures a serving function with an API handler that supports an OpenAI-compatible `POST /v1/chat/completions` endpoint. It extracts the `model` field and `messages` array from the request body, and makes the request path available to downstream steps.

```python
import mlrun
from http import HTTPMethod

from mlrun.common.schemas.serving import APIHandlerAction
from mlrun.runtimes.nuclio.serving import APIHandlerConfig

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

# Allow the OpenAI-compatible chat completion endpoint
config.add_endpoint_handler(
    "/v1/chat/completions",
    HTTPMethod.POST,
    APIHandlerAction.ALLOW,
    description="OpenAI-compatible chat completion",
)

# Block all admin paths
config.add_endpoint_handler("/admin/*", HTTPMethod.GET, APIHandlerAction.FORBID)
config.add_endpoint_handler("/admin/*", HTTPMethod.POST, APIHandlerAction.FORBID)

# Extract model and messages from the request body using JSONPath
config.add_body_mapping("model_name", "$.model")
config.add_body_mapping("messages", "$.messages")

# Attach to the serving function
serving_fn.set_api_handler_config(config)

# --- Test locally with the mock server ---
server = serving_fn.to_mock_server()

# Allowed endpoint: body_map extracts model_name and messages; chat_handler receives them as kwargs
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
