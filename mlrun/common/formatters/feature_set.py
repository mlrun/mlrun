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


class FeatureSetFormat(ObjectFormat, mlrun.common.types.StrEnum):
    full = "full"
    minimal = "minimal"

    @staticmethod
    def format_method(_format: str) -> typing.Optional[typing.Callable]:
        return {
            FeatureSetFormat.full: None,
            FeatureSetFormat.minimal: FeatureSetFormat.filter_obj_method(
                [
                    "metadata.name",
                    "metadata.project",
                    "metadata.tag",
                    "metadata.updated",
                    "metadata.uid",
                    "metadata.labels",
                    "spec.entities",
                    "spec.description",
                    "spec.targets",
                    "spec.engine",  # It's not needed by the UI, but we override it anyway to storey if empty
                    "status.state",
                ]
            ),
        }[_format]
