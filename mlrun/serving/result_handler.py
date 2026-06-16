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

"""Result handler — reshapes graph responses based on per-endpoint output_body_mappings."""

from http import HTTPMethod
from re import Pattern
from typing import Any

from mlrun.serving.endpoint_mapping import (
    APIHandlerConfig,
    EndpointConfig,
    EndpointMatch,
    apply_body_map_with_dict_check,
    collect_endpoint_matches,
    compile_body_map,
    compile_dynamic_path_patterns,
    merge_body_maps,
)


class ResultHandler:
    """Reshapes graph responses before returning them to the REST caller.

    Built once at server init from :class:`APIHandlerConfig`. Per request,
    looks up the effective merged ``output_body_mappings`` for ``(method, path)``
    and applies it to the graph response. If no mapping is configured for the
    matched endpoint, the response is returned as-is.
    """

    def __init__(
        self,
        config: APIHandlerConfig,
    ) -> None:
        self._endpoints = config.endpoints
        self._endpoint_patterns: list[tuple[HTTPMethod, Pattern, EndpointConfig]]
        self._star_patterns: list[tuple[HTTPMethod, str, EndpointConfig]]
        self._endpoint_patterns, self._star_patterns = compile_dynamic_path_patterns(
            config.endpoints
        )

        self._parsed_output_body_map: dict[str, dict[str, tuple[Any, bool]]] = {}
        for ep in config.endpoints.values():
            if ep.output_body_mappings:
                self._parsed_output_body_map[ep.get_endpoint_key()] = compile_body_map(
                    ep.output_body_mappings, ep.get_endpoint_key()
                )

    def apply(self, method: HTTPMethod | str, path: str, response: Any) -> Any:
        """Apply output body mappings to reshape the graph response.

        Mirrors the input-side mapping contract: when output mappings are
        configured for the matched endpoint, raise only when the mapping cannot
        be applied to a response it was expected to apply to. Non-dict response
        with only optional mappings silently passes through, mirroring the
        input handler's behavior for non-dict bodies with optional mappings.

        :param method: HTTP method of the request.
        :param path: Request path.
        :param response: Graph response to reshape.
        :return: Reshaped response, or the original response if no mapping applies.
        :raises mlrun.errors.MLRunUnprocessableEntityError: Non-dict response with mandatory mapping, or apply error.
        :raises mlrun.errors.MLRunBadRequestError: Other failure during mapping apply.
        """
        if not self._parsed_output_body_map:
            return response

        if isinstance(method, str):
            try:
                method = HTTPMethod(method.upper())
            except ValueError:
                return response

        matches: list[EndpointMatch] = collect_endpoint_matches(
            method,
            path,
            self._endpoints,
            self._endpoint_patterns,
            self._star_patterns,
        )
        if not matches:
            return response

        effective_map = merge_body_maps(matches, self._parsed_output_body_map)
        if not effective_map:
            return response

        reshaped = apply_body_map_with_dict_check(
            response,
            effective_map,
            input_body=False,
        )
        return reshaped if reshaped is not None else response
