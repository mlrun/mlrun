# Copyright 2026 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""API Handler implementation for serving graphs"""

from http import HTTPMethod

from jsonpath_ng import parse as jsonpath_parse
from jsonpath_ng.exceptions import JsonPathLexerError, JsonPathParserError

import mlrun.common.schemas as schemas
import mlrun.errors
import mlrun.runtimes.nuclio.serving
import mlrun.serving.server
import mlrun.serving.states
import mlrun.serving.utils as serving_utils
import mlrun.utils
from mlrun.common.schemas.serving import _APIEndpointKeys
from mlrun.serving.utils import _MappedBody


class _APIHandlerStep(mlrun.serving.states.TaskStep):
    """Private API handler step for routing and validating serving requests"""

    kind = "api_handler"
    default_shape = "diamond"

    def __init__(
        self,
        config: mlrun.runtimes.nuclio.serving.APIHandlerConfig | dict | None = None,
        name: str | None = None,
        context: mlrun.serving.server.GraphContext | None = None,
        **kwargs,
    ):
        # Filter kwargs to only pass what BaseStep expects
        base_kwargs = {
            k: v for k, v in kwargs.items() if k in ["after", "shape", "max_iterations"]
        }
        super().__init__(name=name or "api-handler", **base_kwargs)

        if isinstance(config, dict):
            self.config = mlrun.runtimes.nuclio.serving.APIHandlerConfig.from_dict(
                config
            )
        elif isinstance(config, mlrun.runtimes.nuclio.serving.APIHandlerConfig):
            self.config = config
        else:
            self.config = mlrun.runtimes.nuclio.serving.APIHandlerConfig()
        self.context = context

        # Parse JSONPath expressions during initialization for performance and early error detection
        self._parsed_body_map = {}
        if self.config.body_map:
            for param_name, jsonpath_expr in self.config.body_map.items():
                try:
                    self._parsed_body_map[param_name] = jsonpath_parse(jsonpath_expr)
                except (JsonPathLexerError, JsonPathParserError) as exc:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"Invalid JSON path expression for parameter '{param_name}': "
                        f"'{jsonpath_expr}'. Error: {exc}"
                    ) from exc

        mlrun.utils.logger.debug("The context in API handler", context=self.context)

    def _apply_parsed_body_map(self, body: dict) -> "_MappedBody":
        """Apply pre-parsed JSONPath expressions to extract parameters from event body.

        :param body: The event body dict to extract parameters from.
        :return: A :class:`_MappedBody` with extracted parameters.
        :raises KeyError: If any JSONPath expression has no match in body.
        """
        result = {}
        for param_name, parsed_expr in self._parsed_body_map.items():
            matches = parsed_expr.find(body)
            if not matches:
                raise KeyError(
                    f"JSONPath expression for parameter '{param_name}' "
                    f"matched nothing in the event body"
                )
            # Single match: return value; multiple matches: return list
            if len(matches) == 1:
                result[param_name] = matches[0].value
            else:
                result[param_name] = [match.value for match in matches]
        return _MappedBody(result)

    def do(self, event):
        """Handle incoming request and validate against configured endpoints"""
        try:
            # In MLRun serving framework, the actual event metadata is available in the context
            # while the event parameter here is typically the body content
            method = None
            path = None

            # Check the event object directly
            if hasattr(event, "method"):
                method = event.method
            if hasattr(event, "path"):
                path = event.path

            # Fallback to context if available
            if (method is None or path is None) and self.context:
                if hasattr(self.context, "current_event"):
                    original_event = self.context.current_event
                    method = method or getattr(original_event, "method", None)
                    path = path or getattr(original_event, "path", None)

            # Validate that we have both method and path
            if method is None:
                raise mlrun.errors.MLRunBadRequestError(
                    "HTTP method not found in request context"
                )
            if path is None:
                raise mlrun.errors.MLRunBadRequestError(
                    "Request path not found in request context"
                )

            # Convert string method to HTTPMethod if needed
            if isinstance(method, str):
                try:
                    method = HTTPMethod(method.upper())
                except ValueError:
                    raise mlrun.errors.MLRunBadRequestError(
                        f"Unsupported HTTP method: {method}"
                    )

            mlrun.utils.logger.debug(
                "API handler processing request",
                method=method.value,
                path=path,
                endpoints_count=len(self.config._endpoints),
            )

            # Find matching endpoint
            matching_endpoint_key = self._match_endpoint(method, path)

            if not matching_endpoint_key:
                # No matching endpoint found
                mlrun.utils.logger.warning(
                    "No matching endpoint found",
                    method=method.value,
                    path=path,
                )
                raise mlrun.errors.MLRunNotFoundError(
                    f"Endpoint not found: {method.value} {path}"
                )

            # Get endpoint definition
            endpoint_def = self.config._endpoints[matching_endpoint_key]
            action = endpoint_def[_APIEndpointKeys.ACTION]

            # Parse the endpoint key for logging
            matched_method, matched_path = self.config._parse_endpoint_key(
                matching_endpoint_key
            )

            mlrun.utils.logger.debug(
                "Found matching endpoint",
                method=method.value,
                path=path,
                matched_path=matched_path,
                action=action,
            )

            # Handle the action
            if action == schemas.APIHandlerAction.ALLOW:
                # Apply body_map transformation if configured at the config level
                if self._parsed_body_map:
                    body = event.body if hasattr(event, "body") else event
                    if isinstance(body, dict):
                        mapped_body = self._apply_parsed_body_map(body)
                        mlrun.utils.logger.debug(
                            "Applied body_map transformation",
                            body_map=self.config.body_map,
                            mapped_params=list(mapped_body.keys()),
                        )
                        if hasattr(event, "body"):
                            event.body = mapped_body
                        else:
                            event = mapped_body
                    else:
                        mlrun.utils.logger.warning(
                            "body_map configured but event body is not a dict, "
                            "skipping body_map transformation",
                            body_type=type(body).__name__,
                        )
                # Pass the event to the next step in the graph
                return event
            elif action == schemas.APIHandlerAction.FORBID:
                # Reject the request
                raise mlrun.errors.MLRunBadRequestError(
                    f"Access forbidden to {method.value} {path}"
                )
            else:
                raise mlrun.errors.MLRunBadRequestError(f"Unknown action: {action}")

        except Exception as exc:
            # Log the error and re-raise
            mlrun.utils.logger.error(
                "API handler error",
                error=str(exc),
                method=getattr(event, "method", "unknown"),
                path=getattr(event, "path", "unknown"),
            )
            raise

    def _match_endpoint(self, method: HTTPMethod, path: str) -> str | None:
        """Find matching endpoint key for the given method and path"""
        # Only exact matches supported
        endpoint_key = serving_utils._combine_serving_endpoint_key(method, path)
        if endpoint_key in self.config._endpoints:
            return endpoint_key

        return None
