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

import re
from http import HTTPMethod
from re import Pattern
from typing import Any, Union
from urllib.parse import parse_qs, unquote, urlsplit

import nuclio_sdk
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
from mlrun.serving.utils import _RequestContext


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

        # Pre-compile patterns in a single pass for performance.
        # Template patterns: /api/{user_id}/items → regex with named groups.
        # Star patterns:     /api/v1/*           → plain prefix string.
        self._endpoint_patterns: list[tuple[HTTPMethod, Pattern, str, dict]]
        self._star_patterns: list[tuple[HTTPMethod, str, str, dict]]
        self._endpoint_patterns, self._star_patterns = self._compile_patterns()

        mlrun.utils.logger.debug("The context in API handler", context=self.context)

    def _compile_patterns(
        self,
    ) -> tuple[
        list[tuple[HTTPMethod, Pattern, str, dict]],
        list[tuple[HTTPMethod, str, str, dict]],
    ]:
        """Compile all non-exact endpoint patterns in a single pass.

        Exact endpoints (no ``{`` or ``*``) are handled by O(1) dict lookup at
        request time and do not need pre-compilation.

        Template patterns (``{param}``):
            /api/{user_id}/items/{item_id}
            ↓ becomes ↓
            ^/api/(?P<user_id>[^/]+)/items/(?P<item_id>[^/]+)$

            - ^ and $ anchor the full path
            - ``(?P<name>[^/]+)`` captures one non-slash path segment per parameter

        Star (wildcard) patterns (``*`` at end only):
            /api/v1/*  → prefix ``/api/v1/``
            Matches any path that starts with the prefix and has at least one
            additional character after it.

        :return: Tuple of (template_patterns, star_patterns) where

            * ``template_patterns`` is a list of
              ``(method, compiled_regex, endpoint_key, endpoint_config)``
            * ``star_patterns`` is a list of
              ``(method, prefix, endpoint_key, endpoint_config)``
        """
        template_patterns: list[tuple[HTTPMethod, Pattern, str, dict]] = []
        star_patterns: list[tuple[HTTPMethod, str, str, dict]] = []

        for endpoint_key, endpoint_config in self.config._endpoints.items():
            method, path_pattern = self.config._parse_endpoint_key(endpoint_key)

            if "*" in path_pattern:
                # --- Star (wildcard) pattern ---
                if not path_pattern.endswith("*"):
                    raise mlrun.errors.MLRunValueError(
                        f"Invalid endpoint path '{path_pattern}': "
                        f"wildcard '*' must be at the end of the path"
                    )
                if path_pattern.count("*") > 1:
                    raise mlrun.errors.MLRunValueError(
                        f"Invalid endpoint path '{path_pattern}': "
                        f"wildcard '*' must appear only once at the end of the path"
                    )
                # Strip trailing '*'; guarantee a trailing '/' for prefix matching.
                # Examples: /api/v1/* → /api/v1/   /* → /
                prefix = path_pattern.rstrip("*")
                if not prefix.endswith("/"):
                    prefix += "/"
                star_patterns.append((method, prefix, endpoint_key, endpoint_config))

            elif "{" in path_pattern:
                # --- Template pattern ---
                # Convert {param} placeholders to named regex capture groups.
                # Example: /api/{user_id}/data → ^/api/(?P<user_id>[^/]+)/data$
                regex_pattern = re.escape(path_pattern)
                regex_pattern = re.sub(
                    r"\\\{([^}]+)\\\}",  # Match escaped {param_name}
                    r"(?P<\1>[^/]+)",  # Replace with (?P<param_name>[^/]+)
                    regex_pattern,
                )
                regex_pattern = f"^{regex_pattern}$"
                try:
                    compiled = re.compile(regex_pattern)
                except re.error as exc:
                    raise mlrun.errors.MLRunValueError(
                        f"Failed to compile regex for endpoint pattern '{path_pattern}' "
                        f"(key: {endpoint_key}): {exc}"
                    ) from exc
                template_patterns.append(
                    (method, compiled, endpoint_key, endpoint_config)
                )
            # else: exact endpoint – handled by dict lookup, no compilation needed

        # Validate that body_map parameter names don't overlap with path template
        # parameter names. This is a static conflict that can be caught early,
        # before any request arrives.
        if self._parsed_body_map and template_patterns:
            body_map_names = set(self._parsed_body_map.keys())
            for _, compiled_pattern, _, _ in template_patterns:
                path_param_names = set(compiled_pattern.groupindex.keys())
                overlapping = body_map_names & path_param_names
                if overlapping:
                    raise mlrun.errors.MLRunValueError(
                        f"Configuration conflict: body_map parameter(s) "
                        f"{', '.join(sorted(overlapping))} overlap with path template "
                        f"parameter(s) in pattern '{compiled_pattern.pattern}'. "
                        f"Rename the body_map parameter(s) or the path template "
                        f"placeholder(s) to avoid ambiguity."
                    )

        return template_patterns, star_patterns

    def _apply_parsed_body_map(self, body: dict) -> dict:
        """Apply pre-parsed JSONPath expressions to extract parameters from event body.

        :param body: The event body dict to extract parameters from.
        :return: Dict of extracted parameters (missing JSONPath matches are silently skipped).
        """
        result = {}
        for param_name, parsed_expr in self._parsed_body_map.items():
            matches = parsed_expr.find(body)
            if not matches:
                continue
            # Single match: return value; multiple matches: return list
            if len(matches) == 1:
                result[param_name] = matches[0].value
            else:
                result[param_name] = [match.value for match in matches]
        return result

    @staticmethod
    def _parse_query_params(path_query: str) -> tuple[str, dict[str, str | list[str]]]:
        """Parse path and query parameters from a request path.

        Query values are normalized to strings. For repeated query keys, returns a list.

        :param path_query: Raw path, possibly with query string.
        :return: Tuple of normalized path (without query) and query-params dict.
                 Values are strings for single occurrences, lists for multiple.
        """
        parsed_url = urlsplit(path_query)
        normalized_path = parsed_url.path or "/"

        query_params: dict[str, str | list[str]] = {}
        parsed_query = parse_qs(parsed_url.query, keep_blank_values=True)
        for query_key, values in parsed_query.items():
            if values:
                # Single value: return as string; multiple values: return as list of strings
                query_params[query_key] = values[0] if len(values) == 1 else values

        return normalized_path, query_params

    def _extract_query_params(
        self,
        event: Union[nuclio_sdk.Event, "mlrun.serving.server.MockEvent"],
        path: str,
    ) -> tuple[str, dict[str, str | list[str]]]:
        """Extract query parameters from event or path.

        First attempts to parse from path (for mock/test events with ?query in path),
        then falls back to event.fields (for real Nuclio HTTP events).

        :param event: Event object (Nuclio event or MockEvent, may have fields attribute)
        :param path: Request path (may include query string)
        :return: Tuple of (normalized_path, query_params dict)
        """
        # Try to parse query params from path first (mock/test events)
        normalized_path, query_params = self._parse_query_params(path)

        # If no query params in path, try event.fields (real Nuclio HTTP events)
        if not query_params:
            nuclio_fields = getattr(event, "fields", None)
            if nuclio_fields:
                # Nuclio fields are dict[str, list[str]]
                # Convert to our format: single value → str, multiple values → list
                query_params = {}
                for key, values in nuclio_fields.items():
                    if isinstance(values, list):
                        query_params[key] = (
                            values if len(values) > 1 else values[0] if values else None
                        )
                    else:
                        query_params[key] = values

        return normalized_path, query_params

    def do(
        self, event: Union[nuclio_sdk.Event, "mlrun.serving.server.MockEvent"]
    ) -> Union[nuclio_sdk.Event, "mlrun.serving.server.MockEvent"]:
        """Handle incoming request and validate against configured endpoints

        :param event: Event object (Nuclio event or MockEvent)
        :return: Original event or RequestContext with extracted parameters
        """
        try:
            method = getattr(event, "method", None)
            path = getattr(event, "path", None)

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

            # Extract normalized path and query parameters (single parse)
            normalized_path, query_params = self._extract_query_params(event, path)

            mlrun.utils.logger.debug(
                "API handler processing request",
                method=method.value,
                path=normalized_path,
                query_params=query_params,
            )

            # Find matching endpoint
            matching_endpoint_key, path_params = self._match_endpoint(
                method, normalized_path
            )

            if not matching_endpoint_key:
                # Check if path exists with any method (for 405 vs 404 distinction)
                # Note: Only checking exact paths for performance; templated paths will return 404
                path_exists = False
                for key in self.config._endpoints.keys():
                    _, endpoint_path = self.config._parse_endpoint_key(key)
                    if endpoint_path == normalized_path:
                        path_exists = True
                        break

                if path_exists:
                    # Path exists but method not allowed (405)
                    mlrun.utils.logger.warning(
                        "Method not allowed for endpoint",
                        method=method.value,
                        path=normalized_path,
                    )
                    raise mlrun.errors.MLRunMethodNotAllowedError(
                        f"Method not allowed: {method.value} {normalized_path}"
                    )
                else:
                    # No matching endpoint found (404)
                    mlrun.utils.logger.warning(
                        "No matching endpoint found",
                        method=method.value,
                        path=normalized_path,
                    )
                    raise mlrun.errors.MLRunNotFoundError(
                        f"Endpoint not found: {method.value} {normalized_path}"
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
                path=normalized_path,
                matched_path=matched_path,
                action=action,
            )

            # Handle the action
            if action == schemas.APIHandlerAction.ALLOW:
                # Extract body_map parameters if configured
                body_params = {}
                if self._parsed_body_map:
                    body = event.body if hasattr(event, "body") else event
                    if isinstance(body, dict):
                        try:
                            body_params = self._apply_parsed_body_map(body)
                            mlrun.utils.logger.debug(
                                "Applied body_map transformation",
                                body_map=self.config.body_map,
                                extracted_params=list(body_params.keys()),
                            )
                        except Exception as exc:
                            raise mlrun.errors.MLRunBadRequestError(
                                f"Failed to process body_map transformation: {exc}"
                            ) from exc
                    # Non-dict body (e.g. None, string, bytes): body_map does not apply
                    # to this endpoint's format — silently skip, same as a JSONPath miss.

                # Build system-injected URL params when include_url_info is enabled.
                # mlrun_request_path holds the normalized path of the matched request.
                url_params: dict[str, Any] = {}
                if self.config.include_url_info:
                    url_params["mlrun_request_path"] = normalized_path

                # Build the event body for the next step.
                # When any params are present, always use _RequestContext so the
                # handler receives the original body as the first positional arg and all
                # extracted params (body_map, path, query, url) as keyword args.
                # When nothing was extracted, pass the original event body unchanged.
                if body_params or path_params or query_params or url_params:
                    mlrun.utils.logger.debug(
                        "Creating RequestContext",
                        body_params=body_params,
                        path_params=path_params,
                        query_params=query_params,
                        url_params=url_params,
                    )
                    original_body = event.body if hasattr(event, "body") else None
                    event.body = _RequestContext(
                        original_body=original_body,
                        body_params=body_params,
                        path_params=path_params,
                        query_params=query_params,
                        url_params=url_params,
                    )

                # Pass the event to the next step in the graph
                return event
            elif action == schemas.APIHandlerAction.FORBID:
                # Reject the request
                raise mlrun.errors.MLRunAccessDeniedError(
                    f"Access forbidden to {method.value} {normalized_path}"
                )
            else:
                raise mlrun.errors.MLRunInternalServerError(f"Unknown action: {action}")

        except Exception as exc:
            # Log the error and re-raise
            mlrun.utils.logger.error(
                "API handler error",
                error=str(exc),
                method=getattr(event, "method", "unknown"),
                path=getattr(event, "path", "unknown"),
            )
            raise

    def _match_endpoint(
        self, method: HTTPMethod, path: str
    ) -> tuple[str | None, dict[str, str]]:
        """Find matching endpoint key for the given method and path.

        Uses a three-phase search strategy with strict precedence:
        1. Fast exact match lookup (O(1) dict lookup)
        2. Pre-compiled regex pattern matching for path templates (O(n), insertion order)
        3. Star (wildcard) prefix matching (O(n), insertion order)

        :param method: HTTP method to match
        :param path: Request path to match
        :return: Tuple of (endpoint_key, extracted_path_params) or (None, {}) if no match.
                 Path params are always strings (extracted from URL segments).
        """
        # Phase 1: Fast path for exact matches (no path parameters)
        endpoint_key = serving_utils._combine_serving_endpoint_key(method, path)
        if endpoint_key in self.config._endpoints:
            return endpoint_key, {}

        # Phase 2: Try pre-compiled regex patterns for path templates
        for (
            pattern_method,
            compiled_pattern,
            pattern_endpoint_key,
            _,
        ) in self._endpoint_patterns:
            if pattern_method != method:
                continue

            match = compiled_pattern.match(path)
            if match:
                # Extract path parameters from named groups
                # Note: URL-decode path segments to handle encoded characters
                path_params = {
                    name: unquote(value) for name, value in match.groupdict().items()
                }
                return pattern_endpoint_key, path_params

        # Phase 3: Try star (wildcard) patterns for prefix matching
        # Ensure path ends with / for comparison with prefix
        # Using trailing slash ensures /apiv2/users doesn't match /api/* prefix
        # Path must be strictly "under" the prefix, not equal to it
        path_with_slash = path if path.endswith("/") else path + "/"
        for (
            star_method,
            prefix,
            star_endpoint_key,
            _,
        ) in self._star_patterns:
            if star_method != method:
                continue

            # Path must start with prefix AND be longer than prefix (at least one more char)
            # This ensures /api/ doesn't match /api/* (only /api/something does)
            if path_with_slash.startswith(prefix) and len(path_with_slash) > len(
                prefix
            ):
                # Star patterns don't extract parameters
                return star_endpoint_key, {}

        return None, {}
