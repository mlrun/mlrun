# Copyright 2023 Iguazio
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

import inspect
from http import HTTPMethod
from re import Pattern
from typing import Any

import mlrun.errors
from mlrun.utils import get_in, update_in

# headers keys with underscore are getting ignored by werkzeug https://github.com/pallets/werkzeug/pull/2622
# to avoid conflicts with WGSI which converts all header keys to uppercase with underscores.
# more info https://github.com/benoitc/gunicorn/issues/2799, this comment can be removed once old keys are removed
event_id_key = "MLRUN-EVENT-ID"
event_path_key = "MLRUN-EVENT-PATH"


def _extract_input_data(input_path, body):
    if input_path:
        if not hasattr(body, "__getitem__"):
            raise TypeError("input_path parameter supports only dict-like event bodies")
        return get_in(body, input_path)
    return body


def _update_result_body(result_path, event_body, result):
    if result_path and event_body:
        if not hasattr(event_body, "__getitem__"):
            raise TypeError(
                "result_path parameter supports only dict-like event bodies"
            )
        update_in(event_body, result_path, result)
    else:
        event_body = result
    return event_body


class _RequestContext(dict):
    """Unified request context passed to handlers after API handler processing.

    Merges parameters from body_map (JSONPath extraction), path templates, query
    string, and system-injected URL info into a single dict.  The original event
    body is preserved as :attr:`original_body`.

    When a downstream :class:`TaskStep` receives this object it calls the handler
    as ``fn(original_body, **params)`` so handlers can declare named parameters::

        def handler(body, model_name, version, **kwargs): ...

    Priority order (highest wins): path > query > body_map.
    Conflicts between path/query/body_map raise :exc:`MLRunBadRequestError`.
    System-injected ``url_params`` (``mlrun_`` prefix) are merged last without
    conflict checking.
    """

    def __init__(
        self,
        original_body: Any = None,
        path_params: dict[str, str] | None = None,
        query_params: dict[str, str | list[str]] | None = None,
        body_params: dict[str, Any] | None = None,
        url_params: dict[str, Any] | None = None,
    ):
        merged: dict[str, Any] = {}
        sources = [
            ("body_map", body_params or {}),
            ("query", query_params or {}),
            ("path", path_params or {}),
        ]

        param_sources: dict[str, list[str]] = {}
        for source_name, params in sources:
            for key, value in params.items():
                if key in merged:
                    param_sources.setdefault(key, []).append(source_name)
                else:
                    param_sources[key] = [source_name]
                merged[key] = value

        conflicts = {k: v for k, v in param_sources.items() if len(v) > 1}
        if conflicts:
            conflict_details = ", ".join(
                f"{k} (from {' + '.join(srcs)})" for k, srcs in conflicts.items()
            )
            raise mlrun.errors.MLRunBadRequestError(
                f"Parameter name conflict detected. Same parameter appears in multiple "
                f"request sources: {conflict_details}. Parameters must be unique across "
                f"path, query, and body_map."
            )

        if url_params:
            merged.update(url_params)

        super().__init__(merged)
        self.original_body = original_body


class StepToDict:
    """auto serialization of graph steps to a python dictionary"""

    meta_keys = [
        "context",
        "name",
        "input_path",
        "result_path",
        "full_event",
        "kwargs",
    ]

    def to_dict(
        self,
        fields: list | None = None,
        exclude: list | None = None,
        strip: bool = False,
    ):
        """convert the step object to a python dictionary"""
        fields = fields or getattr(self, "_dict_fields", None)
        if not fields:
            fields = list(inspect.signature(self.__init__).parameters.keys())
        if exclude:
            fields = [field for field in fields if field not in exclude]

        args = {
            key: getattr(self, key)
            for key in fields
            if getattr(self, key, None) is not None and key not in self.meta_keys
        }
        # add storey kwargs or extra kwargs
        if "kwargs" in fields and (hasattr(self, "kwargs") or hasattr(self, "_kwargs")):
            kwargs = getattr(self, "kwargs", {}) or getattr(self, "_kwargs", {})
            for key, value in kwargs.items():
                if key not in self.meta_keys:
                    args[key] = value

        mod_name = self.__class__.__module__
        class_path = self.__class__.__qualname__
        if mod_name not in ["__main__", "builtins"]:
            class_path = f"{mod_name}.{class_path}"
        struct = {
            "class_name": class_path,
            "name": self.name
            if hasattr(self, "name") and self.name
            else self.__class__.__name__,
            "class_args": args,
        }
        if hasattr(self, "_STEP_KIND"):
            struct["kind"] = self._STEP_KIND
        if hasattr(self, "_input_path") and self._input_path is not None:
            struct["input_path"] = self._input_path
        if hasattr(self, "_result_path") and self._result_path is not None:
            struct["result_path"] = self._result_path
        if hasattr(self, "_full_event") and self._full_event:
            struct["full_event"] = self._full_event
        return struct


class MonitoringApplicationToDict(StepToDict):
    _STEP_KIND = "monitoring_application"
    meta_keys = []


class RouterToDict(StepToDict):
    _STEP_KIND = "router"

    def to_dict(
        self,
        fields: list | None = None,
        exclude: list | None = None,
        strip: bool = False,
    ):
        return super().to_dict(exclude=["routes"], strip=strip)


def combine_serving_endpoint_key(method: HTTPMethod, path: str) -> str:
    """Combine method and path to create a unique endpoint key"""
    return f"{method.value}:{path}"


def check_body_and_path_parameters_overlapping(
    template_patterns: list[
        tuple[HTTPMethod, Pattern, "mlrun.runtimes.nuclio.serving.EndpointConfig"]
    ],
    star_patterns: list[
        tuple[HTTPMethod, str, "mlrun.runtimes.nuclio.serving.EndpointConfig"]
    ],
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
