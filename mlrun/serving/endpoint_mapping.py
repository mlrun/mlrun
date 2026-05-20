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

"""API handler config classes and shared body mapping utilities."""

import re
from dataclasses import dataclass, field
from http import HTTPMethod
from re import Pattern
from typing import Any
from urllib.parse import unquote

import jsonpath_ng
import jsonpath_ng.exceptions

import mlrun.common.schemas as schemas
import mlrun.errors
import mlrun.model
import mlrun.utils

# ---------------------------------------------------------------------------
# Shared HTTP method validator (used by both EndpointConfig and APIHandlerConfig)
# ---------------------------------------------------------------------------


def _validate_http_method(http_method: HTTPMethod | str) -> HTTPMethod:
    """Validate and normalize an HTTP method.

    :param http_method: HTTPMethod enum or string (e.g. ``"GET"``, ``"POST"``).
    :return: Normalized :class:`HTTPMethod` value.
    :raises mlrun.errors.MLRunInvalidArgumentError: If the value is not a valid HTTP method.
    """
    if isinstance(http_method, HTTPMethod):
        return http_method
    if isinstance(http_method, str):
        try:
            return HTTPMethod(http_method.upper())
        except ValueError:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Invalid HTTP method string '{http_method}'. "
                f"Valid values are: {', '.join(m.value for m in HTTPMethod)}"
            ) from None
    raise mlrun.errors.MLRunInvalidArgumentError(
        f"http_method must be an HTTPMethod enum or string, got {type(http_method).__name__} "
        f"with value '{http_method}'. Valid values are: {', '.join(m.value for m in HTTPMethod)}"
    )


def combine_serving_endpoint_key(method: HTTPMethod, path: str) -> str:
    """Combine method and path to create a unique endpoint key."""
    return f"{method.value}:{path}"


# ---------------------------------------------------------------------------
# Config classes
# ---------------------------------------------------------------------------


class BodyMappings(mlrun.model.ModelObj):
    """Directional parameter mappings for a single phase — input (REST → graph) or output (graph → REST).

    The direction is determined by which parameter of ``add_endpoint_handler`` the instance
    is passed to (``body_mappings`` for input, ``output_body_mappings`` for output), not by
    the class itself.

    Usage::

        # Input: extract fields from the incoming REST request body
        input_bm = BodyMappings()
        input_bm.add_mapping("$.model", destination_path="model", mandatory=True)
        input_bm.add_mapping("$.messages", destination_path="messages", mandatory=True)
        input_bm.add_mapping("$.temperature", destination_path="temperature")

        # Output: reshape the graph response before returning to the caller
        output_bm = BodyMappings()
        output_bm.add_mapping(
            "message.content", destination_path="content", mandatory=True
        )
        output_bm.add_mapping("finish_reason", destination_path="finish_reason")

        config = APIHandlerConfig()
        config.add_endpoint_handler(
            "/v1/chat/completions",
            HTTPMethod.POST,
            body_mappings=input_bm,
            output_body_mappings=output_bm,
        )
    """

    _dict_fields = ["mappings"]

    def __init__(self) -> None:
        # _by_src is the primary store: src → {"destination_path", "mandatory"}.
        # _by_dest_index is a reverse lookup: dest → src, for O(1) duplicate detection.
        self._by_src: dict[str, dict] = {}  # src → {"destination_path", "mandatory"}
        self._by_dest_index: dict[str, str] = {}  # dest → src (reverse index only)

    @property
    def mappings(self) -> list[dict]:
        return [{"source_path": src, **data} for src, data in self._by_src.items()]

    @mappings.setter
    def mappings(self, value: list[dict]) -> None:
        self._by_src = {}
        self._by_dest_index = {}
        for m in value or []:
            if "source_path" not in m:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Each body mapping must include 'source_path'"
                )
            if "destination_path" not in m:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Each body mapping must include 'destination_path'"
                )
            src = m["source_path"]
            dest = m["destination_path"]
            self._by_src[src] = {
                "destination_path": dest,
                "mandatory": m.get("mandatory", False),
            }
            self._by_dest_index[dest] = src

    def add_mapping(
        self,
        source_path: str,
        destination_path: str,
        mandatory: bool = False,
    ) -> None:
        """Add a single field mapping.

        :param source_path: JSONPath expression to extract the value from the source.
                                 For input — extracts from the REST request body (e.g. ``"$.model"``).
                                 For output — extracts from the graph response (e.g. ``"message.content"``).
        :param destination_path: Where to place the extracted value.
                                 For input — the parameter name passed into the graph.
                                 For output — the field name in the REST response returned to the caller.
        :param mandatory: If ``True``, a missing field raises an error at request time.
                          If ``False``, a missing field is included as ``None`` in the output
                          (output) or silently skipped (input).
        :raises mlrun.errors.MLRunInvalidArgumentError: If ``destination_path`` is empty.
        """
        if not source_path:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "source_path must be a non-empty string"
            )
        if not destination_path:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "destination_path must be a non-empty string"
            )
        try:
            jsonpath_ng.parse(source_path)
        except (
            jsonpath_ng.exceptions.JsonPathLexerError,
            jsonpath_ng.exceptions.JsonPathParserError,
        ) as exc:
            raise mlrun.errors.MLRunValueError(
                f"Invalid JSON path expression for parameter '{destination_path}': "
                f"'{source_path}'. Error: {exc}"
            ) from exc

        entry = {"destination_path": destination_path, "mandatory": mandatory}

        # Duplicate source — overwrite existing entry.
        if source_path in self._by_src:
            old_dest = self._by_src[source_path].get("destination_path")
            mlrun.utils.logger.warning(
                "Overriding existing body mapping: duplicate source path",
                source_path=source_path,
                old_destination=old_dest,
                new_destination=destination_path,
            )
            self._by_dest_index.pop(old_dest, None)
            self._by_src[source_path] = entry
            self._by_dest_index[destination_path] = source_path
            return

        # Duplicate destination — overwrite existing entry.
        if destination_path in self._by_dest_index:
            old_src = self._by_dest_index[destination_path]
            mlrun.utils.logger.warning(
                "Overriding existing body mapping: duplicate destination path",
                old_source_path=old_src,
                new_source_path=source_path,
                destination_path=destination_path,
            )
            self._by_src.pop(old_src, None)
            self._by_src[source_path] = entry
            self._by_dest_index[destination_path] = source_path
            return

        self._by_src[source_path] = entry
        self._by_dest_index[destination_path] = source_path

    def remove_mapping(self, destination_path: str) -> None:
        """Remove the mapping with the given destination_path. No-op if not found.

        :param destination_path: The destination key to remove.
        """
        if destination_path not in self._by_dest_index:
            return
        src = self._by_dest_index.pop(destination_path)
        self._by_src.pop(src, None)

    def __repr__(self) -> str:
        return f"BodyMappings(mappings={self.mappings!r})"


class EndpointConfig(mlrun.model.ModelObj):
    """Configuration for a single API endpoint — routing and input mapping in one object."""

    _dict_fields = [
        "path",
        "http_method",
        "action",
        "description",
        "input_body_mappings",
        "output_body_mappings",
    ]

    def __init__(
        self,
        path: str = "",
        http_method: HTTPMethod | str = HTTPMethod.POST,
        action: schemas.APIHandlerAction = schemas.APIHandlerAction.ALLOW,
        description: str | None = None,
        input_body_mappings: BodyMappings | None = None,
        output_body_mappings: BodyMappings | None = None,
    ) -> None:
        self.path = self._normalize_path(path)
        self._validate_path(self.path)
        self.http_method = _validate_http_method(http_method)
        self.action = action
        self.description = description
        self.input_body_mappings = input_body_mappings
        self.output_body_mappings = output_body_mappings

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize path to ensure it starts with a forward slash."""
        if not path.startswith("/"):
            return f"/{path}"
        return path

    @staticmethod
    def _validate_path(path: str) -> None:
        """Validate an endpoint path for structural correctness.

        Currently enforces wildcard ``*`` rules:

        * ``*`` may only appear once.
        * ``*`` must be the last character in the path.

        :param path: Normalized path (with leading ``/``) to validate.
        :raises mlrun.errors.MLRunValueError: If the path contains an invalid ``*`` pattern.
        """
        star_count = path.count("*")
        if star_count == 0:
            return
        if path[-1] != "*":
            raise mlrun.errors.MLRunValueError(
                f"Invalid endpoint path '{path}': "
                f"wildcard '*' must be at the end of the path"
            )
        if star_count > 1:
            raise mlrun.errors.MLRunValueError(
                f"Invalid endpoint path '{path}': "
                f"wildcard '*' must appear only once at the end of the path"
            )

    def get_endpoint_key(self) -> str:
        """Return the endpoint key in the format 'METHOD:path', e.g. 'POST:/v1/chat/completions'."""
        return combine_serving_endpoint_key(self.http_method, self.path)

    def __repr__(self) -> str:
        return (
            f"EndpointConfig(path={self.path!r}, http_method={self.http_method!r}, "
            f"action={self.action!r}, input_body_mappings={self.input_body_mappings!r}, "
            f"output_body_mappings={self.output_body_mappings!r})"
        )


class APIHandlerConfig(mlrun.model.ModelObj):
    """Configuration for API handler in serving graph."""

    _dict_fields = ["enabled", "endpoints", "include_url_info"]

    def __init__(
        self,
        enabled: bool = True,
        endpoints: dict[str, dict | EndpointConfig] | None = None,
        include_url_info: bool = False,
    ):
        self.enabled = enabled
        self._endpoints: dict[str, EndpointConfig] = {}
        self.endpoints = endpoints or {}
        self.include_url_info = include_url_info

    @property
    def endpoints(self) -> dict[str, EndpointConfig]:
        """Get the endpoints as a dict keyed by endpoint key (``"METHOD:path"``)."""
        return self._endpoints

    @endpoints.setter
    def endpoints(self, endpoints: dict[str, dict | EndpointConfig]) -> None:
        """Set the endpoints from a dict of raw dicts (deserialization) or EndpointConfig objects."""
        self._endpoints = {}
        for endpoint_key, ep in endpoints.items():
            if isinstance(ep, EndpointConfig):
                self._endpoints[endpoint_key] = ep
            else:
                if "path" not in ep or "http_method" not in ep:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"Endpoint '{endpoint_key}' is using the old APIHandlerConfig format. "
                        f"The API has changed — each endpoint must include 'path' and 'http_method'. "
                        f"Please update your stored config to the new format."
                    )
                body_mappings_dict = ep.get("input_body_mappings")
                input_body_mappings = (
                    BodyMappings.from_dict(body_mappings_dict)
                    if body_mappings_dict
                    else None
                )
                output_body_mappings_dict = ep.get("output_body_mappings")
                output_body_mappings = (
                    BodyMappings.from_dict(output_body_mappings_dict)
                    if output_body_mappings_dict
                    else None
                )
                self._endpoints[endpoint_key] = EndpointConfig(
                    path=ep.get("path", ""),
                    http_method=ep.get("http_method", HTTPMethod.POST),
                    action=ep.get("action", schemas.APIHandlerAction.ALLOW),
                    description=ep.get("description"),
                    input_body_mappings=input_body_mappings,
                    output_body_mappings=output_body_mappings,
                )

    @staticmethod
    def _validate_http_method(http_method: HTTPMethod | str) -> HTTPMethod:
        return _validate_http_method(http_method)

    def get_endpoint_config(
        self, method: HTTPMethod | str, path: str
    ) -> "EndpointConfig | None":
        """Get endpoint configuration for a specific method and path."""
        method = _validate_http_method(method)
        path = EndpointConfig._normalize_path(path)
        endpoint_key = combine_serving_endpoint_key(method, path)
        return self._endpoints.get(endpoint_key)

    def add_endpoint_handler(
        self,
        path: str,
        http_method: HTTPMethod | str = HTTPMethod.POST,
        action: schemas.APIHandlerAction = schemas.APIHandlerAction.ALLOW,
        description: str | None = None,
        input_body_mappings: "BodyMappings | None" = None,
        output_body_mappings: "BodyMappings | None" = None,
    ) -> None:
        """Add an endpoint handler configuration.

        :param path: URL path for the endpoint (e.g., ``/v1/models`` or ``/api/v1/*``)
        :param http_method: HTTP method for the endpoint (``HTTPMethod`` enum or string like ``"GET"``, ``"POST"``)
        :param action: Action to take for this endpoint (:py:class:`~mlrun.common.schemas.APIHandlerAction`)
        :param description: Optional description of the endpoint
        :param input_body_mappings: Optional input :class:`BodyMappings` for this endpoint (REST → graph).
            If ``None``, the request body is passed through as-is.
        :param output_body_mappings: Optional output :class:`BodyMappings` for this endpoint (graph → REST).
            If ``None``, the response is returned as-is.
        :raises mlrun.errors.MLRunValueError: If the path contains an invalid wildcard ``*`` pattern
        """
        ep = EndpointConfig(
            path=path,
            http_method=http_method,
            action=action,
            description=description,
            input_body_mappings=input_body_mappings,
            output_body_mappings=output_body_mappings,
        )
        endpoint_key = ep.get_endpoint_key()

        if endpoint_key in self._endpoints:
            mlrun.utils.logger.warning(
                "Overriding existing endpoint handler configuration",
                method=ep.http_method.value,
                path=ep.path,
                old_action=self._endpoints[endpoint_key].action,
                new_action=str(action),
            )

        self._endpoints[endpoint_key] = ep

    def remove_endpoint_handler(
        self,
        path: str,
        http_method: HTTPMethod | str = HTTPMethod.POST,
    ) -> None:
        """Remove an endpoint handler configuration.

        :param path: URL path for the endpoint to remove
        :param http_method: HTTP method for the endpoint to remove (``HTTPMethod`` enum or string)
        """
        http_method = _validate_http_method(http_method)
        path = EndpointConfig._normalize_path(path)
        endpoint_key = combine_serving_endpoint_key(http_method, path)
        self._endpoints.pop(endpoint_key, None)

    def to_dict(self, fields=None, exclude=None, strip=False):
        d = super().to_dict(fields=fields, exclude=exclude, strip=strip)
        if d.get("endpoints"):
            d["endpoints"] = {
                k: v.to_dict(strip=strip) for k, v in self._endpoints.items()
            }
        return d


# ---------------------------------------------------------------------------
# Shared matching and body map utilities
# ---------------------------------------------------------------------------


@dataclass
class EndpointMatch:
    """A single matched endpoint with its extracted path parameters."""

    endpoint: EndpointConfig
    path_params: dict[str, str] = field(default_factory=dict)


def compile_dynamic_path_patterns(
    endpoints: dict[str, EndpointConfig],
) -> tuple[
    list[tuple[HTTPMethod, Pattern, EndpointConfig]],
    list[tuple[HTTPMethod, str, EndpointConfig]],
]:
    """Compile dynamic endpoint path patterns into matchable structures.

    Handles two dynamic pattern types (exact paths need no compilation):

    - **Path parameters** (``{param}``): e.g. ``/api/{user_id}/items`` →
      compiled regex ``^/api/(?P<user_id>[^/]+)/items$`` with named capture groups.
    - **Wildcard** (``*`` at end): e.g. ``/api/v1/*`` → prefix ``/api/v1/``
      matched against the start of the request path.

    :param endpoints: Dict of endpoint key → :class:`EndpointConfig`.
    :return: Tuple of (template_patterns, star_patterns).
    """
    template_patterns: list[tuple[HTTPMethod, Pattern, EndpointConfig]] = []
    star_patterns: list[tuple[HTTPMethod, str, EndpointConfig]] = []

    # Tracks normalized template shapes per method to detect overlapping templates.
    # e.g. /a/{key} and /a/{user_id} both normalize to /a/{*} → conflict.
    seen_template_shapes: dict[tuple[HTTPMethod, str], str] = {}

    for ep in endpoints.values():
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

    # Sort star patterns by prefix length descending — longer prefix = more specific = higher priority
    star_patterns.sort(key=lambda x: len(x[1]), reverse=True)
    return template_patterns, star_patterns


def check_body_and_path_parameters_overlapping(
    template_patterns: list[tuple[HTTPMethod, Pattern, EndpointConfig]],
    star_patterns: list[tuple[HTTPMethod, str, EndpointConfig]],
) -> None:
    """Check that input_body_mappings destination_path names don't conflict with path
    template parameter names that would be extracted on the same request.

    Two sources of conflict for each template endpoint:
    1. Same endpoint — the template endpoint itself has input_body_mappings with a conflicting name.
    2. Star endpoint — a star endpoint whose prefix covers the template's path has
       input_body_mappings with a conflicting name (its mappings apply to all requests under
       its prefix, including requests that also match the template).

    :raises mlrun.errors.MLRunValueError: On config-time conflict detection.
    """
    for template_method, compiled_pattern, template_ep in template_patterns:
        path_param_names = set(compiled_pattern.groupindex.keys())

        # Source 1: same endpoint has input_body_mappings with conflicting destination_path
        candidates = [(template_ep, "same endpoint")]

        # Source 2: star endpoints whose prefix covers this template's path
        for star_method, prefix, star_ep in star_patterns:
            if star_method != template_method:
                continue
            if template_ep.path.startswith(prefix):
                candidates.append(
                    (star_ep, f"star endpoint '{star_ep.get_endpoint_key()}'")
                )

        for candidate_ep, source_desc in candidates:
            if not candidate_ep.input_body_mappings:
                continue
            dest_names = {
                m["destination_path"]
                for m in candidate_ep.input_body_mappings.mappings
                if m.get("destination_path")
            }
            overlapping = dest_names & path_param_names
            if overlapping:
                raise mlrun.errors.MLRunValueError(
                    f"Configuration conflict: input_body_mappings destination_path(s) "
                    f"{', '.join(sorted(overlapping))} from {source_desc} "
                    f"overlap with path template parameter(s) in pattern "
                    f"'{compiled_pattern.pattern}' "
                    f"(endpoint '{template_ep.get_endpoint_key()}'). "
                    f"Rename the destination_path(s) or the path template "
                    f"placeholder(s) to avoid ambiguity."
                )


def compile_body_map(
    body_mappings: BodyMappings,
    endpoint_key: str,
) -> dict[str, tuple[Any, bool]]:
    """Compile a BodyMappings object into a map of {destination_path: (compiled_expr, mandatory)}.

    :param body_mappings: The :class:`BodyMappings` to compile.
    :param endpoint_key: Endpoint key used in error messages.
    :return: Compiled map ready for use with :func:`apply_body_map`.
    :raises mlrun.errors.MLRunValueError: If a JSONPath expression is invalid.
    """
    compiled_map: dict[str, tuple[Any, bool]] = {}
    for mapping in body_mappings.mappings:
        try:
            compiled_expr = jsonpath_ng.parse(mapping["source_path"])
        except (
            jsonpath_ng.exceptions.JsonPathLexerError,
            jsonpath_ng.exceptions.JsonPathParserError,
        ) as e:
            raise mlrun.errors.MLRunValueError(
                f"Invalid JSONPath expression '{mapping['source_path']}' "
                f"in endpoint '{endpoint_key}': {e}"
            ) from e
        compiled_map[mapping["destination_path"]] = (
            compiled_expr,
            mapping["mandatory"],
        )
    return compiled_map


def collect_endpoint_matches(
    method: HTTPMethod,
    path: str,
    endpoints: dict[str, EndpointConfig],
    endpoint_patterns: list[tuple[HTTPMethod, Pattern, EndpointConfig]],
    star_patterns: list[tuple[HTTPMethod, str, EndpointConfig]],
) -> list[EndpointMatch]:
    """Collect all matching endpoints for the given method and path, ordered by priority.

    Priority (highest first):
    1. Exact match
    2. Template match  (/api/{id})  — skipped when an exact match is found, because
       templates are siblings of exact paths (same depth), not parents.
    3. Star match      (/api/*) — always collected even when an exact match exists,
       because stars are true parent scopes.  Ordered by prefix length descending,
       so /a/b/c/* has higher priority than /a/b/* which has higher priority than /a/*.

    :param method: HTTP method to match.
    :param path: Request path to match.
    :param endpoints: Dict of exact endpoint key → :class:`EndpointConfig`.
    :param endpoint_patterns: Compiled path-parameter patterns.
    :param star_patterns: Compiled wildcard patterns.
    :return: List of :class:`EndpointMatch`, highest priority first.
    """
    matches: list[EndpointMatch] = []

    # Phase 1: Exact match
    endpoint_key = combine_serving_endpoint_key(method, path)
    exact_found = endpoint_key in endpoints
    if exact_found:
        matches.append(EndpointMatch(endpoints[endpoint_key]))

    # Phase 2: Template matches — skipped when an exact match was found
    if not exact_found:
        for pattern_method, compiled_pattern, ep in endpoint_patterns:
            if pattern_method != method:
                continue
            match = compiled_pattern.match(path)
            if match:
                path_params = {
                    name: unquote(value) for name, value in match.groupdict().items()
                }
                matches.append(EndpointMatch(ep, path_params))

    # Phase 3: Star matches — always collected (true parent scopes)
    path_with_slash = path if path.endswith("/") else path + "/"
    for star_method, prefix, ep in star_patterns:
        if star_method != method:
            continue
        if path_with_slash.startswith(prefix) and len(path_with_slash) > len(prefix):
            matches.append(EndpointMatch(ep))

    return matches


def apply_body_map(
    body: dict,
    effective_map: dict[str, tuple[Any, bool]],
    fill_missing_with_none: bool = False,
) -> dict:
    """Apply a compiled body map to extract parameters from a body dict.

    :param body: The body dict to extract parameters from.
    :param effective_map: Merged map of ``{destination_path: (compiled_expr, mandatory)}``.
    :param fill_missing_with_none: If True, missing non-mandatory fields are included as None
        instead of being skipped. Use for output mapping where callers expect a full structure.
    :return: Dict of extracted parameters.
    :raises mlrun.errors.MLRunBadRequestError: If a mandatory field is missing.
    """
    result = {}
    for dest_path, (compiled_expr, mandatory) in effective_map.items():
        matches = compiled_expr.find(body)
        if not matches:
            if mandatory:
                raise mlrun.errors.MLRunBadRequestError(
                    f"Mandatory field '{dest_path}' not found in body"
                )
            if fill_missing_with_none:
                result[dest_path] = None
            continue
        result[dest_path] = (
            matches[0].value if len(matches) == 1 else [m.value for m in matches]
        )
    return result


def merge_body_maps(
    matches: list[EndpointMatch],
    parsed_body_map: dict[str, dict[str, tuple[Any, bool]]],
) -> dict[str, tuple[Any, bool]]:
    """Merge body maps from all matched endpoints, lowest priority first.

    Most specific endpoint wins on conflict:
    - Same destination → higher-priority source overwrites (dict key collision).
    - Same source, different destination → stale destination is removed so the
      value is not passed to two destinations at once.

    :param matches: Ordered list of :class:`EndpointMatch`, index 0 = highest priority.
    :param parsed_body_map: Pre-compiled map of ``{endpoint_key: {dest: (expr, mandatory)}}``.
    :return: Merged map of ``{destination_path: (compiled_expr, mandatory)}``.
    """
    effective_map: dict[str, tuple[Any, bool]] = {}
    src_to_dest: dict[str, str] = {}  # str(expr) → current destination

    for match in reversed(matches):
        ep_key = match.endpoint.get_endpoint_key()
        if ep_key not in parsed_body_map:
            continue
        for dest, (expr, mandatory) in parsed_body_map[ep_key].items():
            src = str(expr)
            if src in src_to_dest:
                effective_map.pop(src_to_dest[src])
            effective_map[dest] = (expr, mandatory)
            src_to_dest[src] = dest
    return effective_map
