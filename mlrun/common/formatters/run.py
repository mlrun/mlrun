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


import mlrun.common.types
from mlrun.common.formatters.base import ObjectFormat


class RunFormat(ObjectFormat, mlrun.common.types.StrEnum):
    # No enrichment, data is pulled as-is from the database.
    standard = "standard"

    # Enrich run with full notifications since the notification params are subtracted from the run body.
    notifications = "notifications"

    # Performs run enrichment, including the run's artifacts. Only available for the `get` run API.
    full = "full"
