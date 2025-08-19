# Copyright 2025 Iguazio
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

from typing import Annotated

import fastapi

from mlrun.common.schemas.model_monitoring.constants import (
    MODEL_ENDPOINT_ID_PATTERN,
    PROJECT_PATTERN,
)

ProjectAnnotation = Annotated[str, fastapi.Path(pattern=PROJECT_PATTERN)]
EndpointIDAnnotation = Annotated[str, fastapi.Path(pattern=MODEL_ENDPOINT_ID_PATTERN)]
