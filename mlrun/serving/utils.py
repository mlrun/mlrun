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
#
import inspect
from typing import Optional

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
        fields: Optional[list] = None,
        exclude: Optional[list] = None,
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
        fields: Optional[list] = None,
        exclude: Optional[list] = None,
        strip: bool = False,
    ):
        return super().to_dict(exclude=["routes"], strip=strip)
