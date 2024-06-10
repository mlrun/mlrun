# Copyright 2024 Iguazio
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

import typing

import mlrun.common.types

from .base import ObjectFormat


class FunctionFormat(ObjectFormat, mlrun.common.types.StrEnum):
    minimal = "minimal"

    @staticmethod
    def format_method(_format: str) -> typing.Optional[typing.Callable]:
        return {
            FunctionFormat.full: None,
            FunctionFormat.minimal: FunctionFormat.filter_obj_method(
                [
                    "kind",
                    "metadata",
                    "status",
                    "spec.description",
                    "spec.command",
                    "spec.image",
                    "spec.default_handler",
                    "spec.default_class",
                    "spec.graph",
                    "spec.preemption_mode",
                    "spec.node_selector",
                    "spec.priority_class_name",
                ]
            ),
        }[_format]
