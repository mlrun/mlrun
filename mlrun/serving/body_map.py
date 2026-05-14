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

"""Shared body mapping utilities used by the API handler and result handler."""

import re
from dataclasses import dataclass, field
from http import HTTPMethod
from re import Pattern
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import jsonpath_ng
import jsonpath_ng.exceptions

import mlrun.errors
import mlrun.serving.utils as serving_utils

if TYPE_CHECKING:
    import mlrun.runtimes.nuclio.serving


@dataclass
class EndpointMatch:
    """A single matched endpoint with its extracted path parameters."""

    endpoint: "mlrun.runtimes.nuclio.serving.EndpointConfig"
    path_params: dict[str, str] = field(default_factory=dict)


def compile_dynamic_path_patterns(
    endpoints: "dict[str, mlrun.runtimes.nuclio.serving.EndpointConfig]",
) -> tuple[
    list[tuple[HTTPMethod, Pattern, "mlrun.runtimes.nuclio.serving.EndpointConfig"]],
    list[tuple[HTTPMethod, str, "mlrun.runtimes.nuclio.serving.EndpointConfig"]],
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
    template_patterns: list[
        tuple[HTTPMethod, Pattern, mlrun.runtimes.nuclio.serving.EndpointConfig]
    ] = []
    star_patterns: list[
        tuple[HTTPMethod, str, mlrun.runtimes.nuclio.serving.EndpointConfig]
    ] = []

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


def compile_body_map(
    body_mappings: "mlrun.runtimes.nuclio.serving.BodyMappings",
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
            compiled_expr = jsonpath_ng.parse(mapping["source_json_path"])
        except (
            jsonpath_ng.exceptions.JsonPathLexerError,
            jsonpath_ng.exceptions.JsonPathParserError,
        ) as e:
            raise mlrun.errors.MLRunValueError(
                f"Invalid JSONPath expression '{mapping['source_json_path']}' "
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
    endpoints: "dict[str, mlrun.runtimes.nuclio.serving.EndpointConfig]",
    endpoint_patterns: "list[tuple[HTTPMethod, Pattern, mlrun.runtimes.nuclio.serving.EndpointConfig]]",
    star_patterns: "list[tuple[HTTPMethod, str, mlrun.runtimes.nuclio.serving.EndpointConfig]]",
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
    endpoint_key = serving_utils.combine_serving_endpoint_key(method, path)
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
) -> dict:
    """Apply a compiled body map to extract parameters from a body dict.

    :param body: The body dict to extract parameters from.
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
                    f"Mandatory field '{dest_path}' not found in body"
                )
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
