# Copyright 2018 Iguazio
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
from typing import Optional

from pydantic import BaseModel, Field

import mlrun.api.utils.helpers


class SecretProviderName(mlrun.api.utils.helpers.StrEnum):
    """Enum containing names of valid providers for secrets."""

    vault = "vault"
    kubernetes = "kubernetes"


class SecretsData(BaseModel):
    provider: SecretProviderName = Field(SecretProviderName.vault)
    secrets: Optional[dict]


class AuthSecretData(BaseModel):
    provider: SecretProviderName = Field(SecretProviderName.kubernetes)
    username: str
    access_key: str

    @staticmethod
    def get_field_secret_key(field: str):
        return {
            "username": "username",
            "access_key": "accessKey",
        }[field]


class SecretKeysData(BaseModel):
    provider: SecretProviderName = Field(SecretProviderName.vault)
    secret_keys: Optional[list]


class UserSecretCreationRequest(SecretsData):
    user: str
