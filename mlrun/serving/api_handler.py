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
from re import Pattern
from typing import Any, Union
from urllib.parse import parse_qs, urlsplit

import nuclio_sdk

import mlrun.common.schemas as schemas
import mlrun.errors
import mlrun.serving.endpoint_mapping as endpoint_mapping
import mlrun.serving.server
import mlrun.serving.states
import mlrun.utils
from mlrun.serving.utils import _RequestContext


class _APIHandlerStep(mlrun.serving.states.TaskStep):
    """Private API handler step for routing and validating serving requests"""

    kind = "api_handler"
    default_shape = "diamond"

    def __init__(
        self,
        config: endpoint_mapping.APIHandlerConfig | dict | None = None,
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
            self.config = endpoint_mapping.APIHandlerConfig.from_dict(config)
        elif isinstance(config, endpoint_mapping.APIHandlerConfig):
            self.config = config
        else:
            self.config = endpoint_mapping.APIHandlerConfig()
        self.context = context

        # Pre-compile patterns and body maps in a single pass for performance.
        # Template patterns: /api/{user_id}/items → regex with named groups.
        # Star patterns:     /api/v1/*           → plain prefix string.
        # Body map cache:    endpoint key → {destination_path: (compiled_expr, mandatory)}
        self._endpoint_patterns: list[
            tuple[HTTPMethod, Pattern, endpoint_mapping.EndpointConfig]
        ]
        self._star_patterns: list[
            tuple[HTTPMethod, str, endpoint_mapping.EndpointConfig]
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
        list[tuple[HTTPMethod, Pattern, "endpoint_mapping.EndpointConfig"]],
        list[tuple[HTTPMethod, str, "endpoint_mapping.EndpointConfig"]],
        dict[str, dict[str, tuple[Any, bool]]],
    ]:
        """Compile path patterns and input body maps.

        :return: Tuple of (template_patterns, star_patterns, parsed_input_body_map).
        """
        template_patterns, star_patterns = (
            endpoint_mapping.compile_dynamic_path_patterns(self.config.endpoints)
        )
        endpoint_mapping.check_body_and_path_parameters_overlapping(
            template_patterns, star_patterns
        )

        parsed_body_map: dict[str, dict[str, tuple[Any, bool]]] = {}
        for ep in self.config.endpoints.values():
            if ep.input_body_mappings:
                parsed_body_map[ep.get_endpoint_key()] = (
                    endpoint_mapping.compile_body_map(
                        ep.input_body_mappings, ep.get_endpoint_key()
                    )
                )

        return template_patterns, star_patterns, parsed_body_map

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
        :raises mlrun.errors.MLRunBadRequestError: Missing method/path, or unsupported HTTP method.
        :raises mlrun.errors.MLRunNotFoundError: No configured endpoint matches the request path (404).
        :raises mlrun.errors.MLRunMethodNotAllowedError: Path matches an endpoint but the method is not allowed (405).
        :raises mlrun.errors.MLRunUnprocessableEntityError: Body mapping failed or mandatory mappings on non-dict body.
        :raises mlrun.errors.MLRunAccessDeniedError: Matched endpoint is configured with action=FORBID.
        :raises mlrun.errors.MLRunInternalServerError: Matched endpoint has an unknown action.
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
                effective_map = endpoint_mapping.merge_body_maps(
                    matches, self._parsed_body_map
                )

                body_params = {}
                if effective_map:
                    body = event.body if hasattr(event, "body") else event
                    body_params = endpoint_mapping.apply_body_map_with_dict_check(
                        body,
                        effective_map,
                    )
                    if body_params is not None:
                        mlrun.utils.logger.debug(
                            "Applied input body mapping",
                            extracted_params=list(body_params.keys()),
                        )
                    else:
                        body_params = {}

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
    ) -> list[endpoint_mapping.EndpointMatch]:
        return endpoint_mapping.collect_endpoint_matches(
            method,
            path,
            self.config.endpoints,
            self._endpoint_patterns,
            self._star_patterns,
        )
