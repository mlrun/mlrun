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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mlrun.errors

if TYPE_CHECKING:
    import mlrun.runtimes.nuclio.serving


@dataclass
class EndpointMatch:
    """A single matched endpoint with its extracted path parameters."""

    endpoint: "mlrun.runtimes.nuclio.serving.EndpointConfig"
    path_params: dict[str, str] = field(default_factory=dict)


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
        result[dest_path] = matches[0].value if len(matches) == 1 else [m.value for m in matches]
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