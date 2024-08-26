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

from enum import Enum
from typing import Optional

from mlrun.genai.schemas.base import BaseWithVerMetadata


class ModelType(str, Enum):
    MODEL = "model"
    ADAPTER = "adapter"


class Model(BaseWithVerMetadata):
    _extra_fields = ["path", "producer", "deployment"]
    _top_level_fields = ["model_type", "task"]

    model_type: ModelType
    base_model: str
    project_id: str
    task: Optional[str]
    path: Optional[str]
    producer: Optional[str]
    deployment: Optional[str]
