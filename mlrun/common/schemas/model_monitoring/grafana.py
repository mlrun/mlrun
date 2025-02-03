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

from typing import Optional, Union

from pydantic.v1 import BaseModel

import mlrun.common.types


class GrafanaColumnType(mlrun.common.types.StrEnum):
    NUMBER = "number"
    STRING = "string"


class GrafanaColumn(BaseModel):
    text: str
    type: str


class GrafanaNumberColumn(GrafanaColumn):
    type: str = GrafanaColumnType.NUMBER


class GrafanaStringColumn(GrafanaColumn):
    type: str = GrafanaColumnType.STRING


class GrafanaTable(BaseModel):
    columns: list[GrafanaColumn]
    rows: list[list[Optional[Union[float, int, str]]]] = []
    type: str = "table"

    def add_row(self, *args):
        self.rows.append(list(args))


class GrafanaModelEndpointsTable(GrafanaTable):
    def __init__(self):
        columns = self._init_columns()
        super().__init__(columns=columns)

    @staticmethod
    def _init_columns():
        return [
            GrafanaColumn(text="endpoint_id", type=GrafanaColumnType.STRING),
            GrafanaColumn(text="endpoint_name", type=GrafanaColumnType.STRING),
            GrafanaColumn(text="endpoint_function", type=GrafanaColumnType.STRING),
            GrafanaColumn(text="endpoint_model", type=GrafanaColumnType.STRING),
            GrafanaColumn(text="endpoint_model_class", type=GrafanaColumnType.STRING),
            GrafanaColumn(text="error_count", type=GrafanaColumnType.NUMBER),
            GrafanaColumn(text="drift_status", type=GrafanaColumnType.NUMBER),
            GrafanaColumn(text="sampling_percentage", type=GrafanaColumnType.NUMBER),
        ]
