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
from dataclasses import dataclass, field
from http import HTTPMethod
from re import Pattern
from typing import Any, Union
from urllib.parse import parse_qs, unquote, urlsplit

import jsonpath_ng
import jsonpath_ng.exceptions
import nuclio_sdk

import mlrun.common.schemas as schemas
import mlrun.errors
import mlrun.runtimes.nuclio.serving
import mlrun.serving.server
import mlrun.serving.states
import mlrun.serving.utils as serving_utils
import mlrun.utils
from mlrun.serving.utils import (
    _RequestContext,
    check_body_and_path_parameters_overlapping,
)


@dataclass
class EndpointMatch:
    """A single matched endpoint with its extracted path parameters."""

    endpoint: "mlrun.runtimes.nuclio.serving.EndpointConfig"
    path_params: dict[str, str] = field(default_factory=dict)


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

        # Pre-compile patterns and body maps in a single pass for performance.
        # Template patterns: /api/{user_id}/items → regex with named groups.
        # Star patterns:     /api/v1/*           → plain prefix string.
        # Body map cache:    endpoint key → {destination_path: (compiled_expr, mandatory)}
        self._endpoint_patterns: list[
            tuple[HTTPMethod, Pattern, mlrun.runtimes.nuclio.serving.EndpointConfig]
        ]
        self._star_patterns: list[
            tuple[HTTPMethod, str, mlrun.runtimes.nuclio.serving.EndpointConfig]
        ]
        self._parsed_body_map: dict[str, dict[str, tuple[Any, bool]]]
        (
            self._endpoint_patterns,
            self._star_patterns,
            self._parsed_body_map,
        ) = self._compile_patterns()

        mlrun.utils.logger.debug("The context in API handler", context=self.context)

    def _compile_patterns(
        self,
    ) -> tuple[
        list[
            tuple[HTTPMethod, Pattern, "mlrun.runtimes.nuclio.serving.EndpointConfig"]
        ],
        list[tuple[HTTPMethod, str, "mlrun.runtimes.nuclio.serving.EndpointConfig"]],
        dict[str, dict[str, tuple[Any, bool]]],
    ]:
        """Compile all non-exact endpoint patterns and input body maps in a single pass.

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

        :return: Tuple of (template_patterns, star_patterns, parsed_body_map) where

            * ``template_patterns`` is a list of ``(method, compiled_regex, EndpointConfig)``
            * ``star_patterns`` is a list of ``(method, prefix, EndpointConfig)``
            * ``parsed_body_map`` maps endpoint key → ``{destination_path: (compiled_expr, mandatory)}``
        """
        template_patterns: list[
            tuple[HTTPMethod, Pattern, mlrun.runtimes.nuclio.serving.EndpointConfig]
        ] = []
        star_patterns: list[
            tuple[HTTPMethod, str, mlrun.runtimes.nuclio.serving.EndpointConfig]
        ] = []
        parsed_body_map: dict[str, dict[str, tuple[Any, bool]]] = {}

        # Tracks normalized template shapes per method to detect overlapping templates.
        # e.g. /a/{key} and /a/{user_id} both normalize to /a/{*} → conflict.
        seen_template_shapes: dict[tuple[HTTPMethod, str], str] = {}

        for ep in self.config.endpoints.values():
            method = ep.http_method
            path_pattern = ep.path

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
                star_patterns.append((method, prefix, ep))

            elif "{" in path_pattern:
                # --- Template pattern ---
                # Detect overlapping templates: /a/{key} and /a/{user_id} are ambiguous.
                # Normalize by replacing all {param} placeholders with {*} and check for
                # duplicates per HTTP method.
                shape = re.sub(r"\{[^}]*\}", "{*}", path_pattern)
                shape_key = (method, shape)
                if shape_key in seen_template_shapes:
                    raise mlrun.errors.MLRunValueError(
                        f"Overlapping template endpoints for {method.value}: "
                        f"'{path_pattern}' and '{seen_template_shapes[shape_key]}' "
                        f"match the same set of paths"
                    )
                seen_template_shapes[shape_key] = path_pattern

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
                        f"(key: {ep.get_endpoint_key()}): {exc}"
                    ) from exc
                template_patterns.append((method, compiled, ep))
            # else: exact endpoint – handled by dict lookup, no compilation needed

            # Compile input_body_mappings for this endpoint (any pattern type)
            if ep.input_body_mappings:
                compiled_map: dict[str, tuple[Any, bool]] = {}
                for mapping in ep.input_body_mappings.mappings:
                    try:
                        compiled_expr = jsonpath_ng.parse(mapping["source_json_path"])
                    except (
                        jsonpath_ng.exceptions.JsonPathLexerError,
                        jsonpath_ng.exceptions.JsonPathParserError,
                    ) as e:
                        raise mlrun.errors.MLRunValueError(
                            f"Invalid JSONPath expression '{mapping['source_json_path']}' "
                            f"in endpoint '{ep.get_endpoint_key()}': {e}"
                        ) from e
                    compiled_map[mapping["destination_path"]] = (
                        compiled_expr,
                        mapping["mandatory"],
                    )
                parsed_body_map[ep.get_endpoint_key()] = compiled_map

        # Sort star patterns by prefix length descending — longer prefix = more specific = higher priority
        star_patterns.sort(key=lambda x: len(x[1]), reverse=True)
        check_body_and_path_parameters_overlapping(template_patterns, star_patterns)
        return template_patterns, star_patterns, parsed_body_map

    def _apply_body_map(
        self,
        body: dict,
        effective_map: dict[str, tuple[Any, bool]],
    ) -> dict:
        """Apply a compiled body map to extract parameters from the event body.

        :param body: The event body dict to extract parameters from.
        :param effective_map: Merged map of ``{destination_path: (compiled_expr, mandatory)}``.
        :return: Dict of extracted parameters.
        :raises mlrun.errors.MLRunBadRequestError: If a mandatory field is missing.
        """
        result = {}
        for dest_path, (compiled_expr, mandatory) in effective_map.items():
            matches = compiled_expr.find(body)
            if not matches:
                if mandatory:
                    raise mlrun.errors.MLRunBadRequestError(
                        f"Mandatory field '{dest_path}' not found in request body"
                    )
                continue
            if len(matches) == 1:
                result[dest_path] = matches[0].value
            else:
                result[dest_path] = [match.value for match in matches]
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

    def _merge_body_maps(
        self,
        matches: list[EndpointMatch],
    ) -> dict[str, tuple[Any, bool]]:
        """Merge input body maps from all matched endpoints, lowest priority first.

        Most specific endpoint wins on conflict:
        - Same destination → higher-priority source overwrites (dict key collision).
        - Same source, different destination → stale destination is removed so the
          value is not passed to two destinations at once.

        :param matches: Ordered list of :class:`EndpointMatch` from
                        :meth:`_collect_endpoint_matches` (index 0 = highest priority).
        :return: Merged map of ``{destination_path: (compiled_expr, mandatory)}``.
        """
        effective_map: dict[str, tuple[Any, bool]] = {}
        src_to_dest: dict[str, str] = {}  # str(expr) → current destination

        for match in reversed(matches):
            ep_key = match.endpoint.get_endpoint_key()
            if ep_key not in self._parsed_body_map:
                continue
            for dest, (expr, mandatory) in self._parsed_body_map[ep_key].items():
                src = str(expr)
                if src in src_to_dest:
                    effective_map.pop(src_to_dest[src])
                effective_map[dest] = (expr, mandatory)
                src_to_dest[src] = dest
        return effective_map

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

            # Find all matching endpoints (highest priority first)
            matches = self._collect_endpoint_matches(method, normalized_path)
            if not matches:
                self._raise_not_found_endpoint(method, normalized_path)

            first_match = matches[0]
            ep = first_match.endpoint
            path_params = first_match.path_params

            mlrun.utils.logger.debug(
                "Found matching endpoint",
                method=method.value,
                path=normalized_path,
                matched_path=ep.path,
                action=ep.action,
            )

            # Handle the action
            if ep.action == schemas.APIHandlerAction.ALLOW:
                effective_map = self._merge_body_maps(matches)

                body_params = {}
                if effective_map:
                    body = event.body if hasattr(event, "body") else event
                    if isinstance(body, dict):
                        try:
                            body_params = self._apply_body_map(body, effective_map)
                            mlrun.utils.logger.debug(
                                "Applied input body mapping",
                                extracted_params=list(body_params.keys()),
                            )
                        except Exception as exc:
                            raise mlrun.errors.MLRunBadRequestError(
                                f"Failed to process body mapping: {exc}"
                            ) from exc
                    # Non-dict body (e.g. None, string, bytes): body mappings do not apply — silently skip.

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
            elif ep.action == schemas.APIHandlerAction.FORBID:
                # Reject the request
                raise mlrun.errors.MLRunAccessDeniedError(
                    f"Access forbidden to {method.value} {normalized_path}"
                )
            else:
                raise mlrun.errors.MLRunInternalServerError(
                    f"Unknown action: {ep.action}"
                )

        except Exception as exc:
            # Log the error and re-raise
            mlrun.utils.logger.error(
                "API handler error",
                error=str(exc),
                method=getattr(event, "method", "unknown"),
                path=getattr(event, "path", "unknown"),
            )
            raise

    def _raise_not_found_endpoint(self, method: str, normalized_path: str) -> None:
        # Check if path matches any registered endpoint regardless of method (for 405 vs 404 distinction).
        # String comparison is insufficient — template paths like /users/{id} won't match /users/123.
        path_exists = any(
            self._collect_endpoint_matches(m, normalized_path)
            for m in HTTPMethod
            if m != method
        )
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

    def _collect_endpoint_matches(
        self, method: HTTPMethod, path: str
    ) -> list[EndpointMatch]:
        """Collect all matching endpoints for the given method and path, ordered by priority.

        Priority (highest first):
        1. Exact match
        2. Template match  (/api/{id})  — skipped when an exact match is found, because
           templates are siblings of exact paths (same depth), not parents.  Including
           them when an exact match exists would inject spurious path parameters and
           unintended body-map inheritance.
        3. Star match      (/api/*) — always collected even when an exact match exists,
           because stars are true parent scopes.  Ordered by prefix length descending,
           so /a/b/c/* has higher priority than /a/b/* which has higher priority than /a/*.

        :param method: HTTP method to match
        :param path: Request path to match
        :return: List of :class:`EndpointMatch`, highest priority first.
        """
        matches: list[EndpointMatch] = []

        # Phase 1: Exact match
        endpoint_key = serving_utils.combine_serving_endpoint_key(method, path)
        exact_found = endpoint_key in self.config.endpoints
        if exact_found:
            matches.append(EndpointMatch(self.config.endpoints[endpoint_key]))

        # Phase 2: Template matches — skipped when an exact match was found
        if not exact_found:
            for pattern_method, compiled_pattern, ep in self._endpoint_patterns:
                if pattern_method != method:
                    continue
                match = compiled_pattern.match(path)
                if match:
                    path_params = {
                        name: unquote(value)
                        for name, value in match.groupdict().items()
                    }
                    matches.append(EndpointMatch(ep, path_params))

        # Phase 3: Star matches — always collected (true parent scopes)
        path_with_slash = path if path.endswith("/") else path + "/"
        for star_method, prefix, ep in self._star_patterns:
            if star_method != method:
                continue
            if path_with_slash.startswith(prefix) and len(path_with_slash) > len(
                prefix
            ):
                matches.append(EndpointMatch(ep))

        return matches
