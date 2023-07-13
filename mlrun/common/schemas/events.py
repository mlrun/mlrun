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
import mlrun.common.types


class EventsModes(mlrun.common.types.StrEnum):
    enabled = "enabled"
    disabled = "disabled"


class EventClientKinds(mlrun.common.types.StrEnum):
    iguazio = "iguazio"
    nop = "nop"


class SecretEventActions(mlrun.common.types.StrEnum):
    created = "created"
    updated = "updated"
    deleted = "deleted"


class AuthSecretEventActions(mlrun.common.types.StrEnum):
    created = "created"
    updated = "updated"
