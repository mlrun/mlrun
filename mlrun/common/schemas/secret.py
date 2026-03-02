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

from datetime import datetime

from pydantic.v1 import BaseModel, Field

import mlrun.common.types


class SecretProviderName(mlrun.common.types.StrEnum):
    """Enum containing names of valid providers for secrets."""

    vault = "vault"
    kubernetes = "kubernetes"


class SecretsData(BaseModel):
    provider: SecretProviderName = Field(SecretProviderName.vault)
    secrets: dict | None = {}


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
    secret_keys: list | None = []


class SecretToken(BaseModel):
    name: str
    token: str


class StoreSecretTokensResponse(BaseModel):
    created_tokens: list[str] = []
    updated_tokens: list[str] = []


class SecretTokenInfo(BaseModel):
    name: str
    expiration: datetime
    user_id: str


class ListSecretTokensResponse(BaseModel):
    secret_tokens: list[SecretTokenInfo]


class DeleteSecretTokenResponse(BaseModel):
    """Response for single token deletion."""

    deleted: bool = Field(
        default=True,
        description="True if token was deleted, False if token was not found",
    )


class DeleteSecretTokensResponse(BaseModel):
    """Response for bulk token deletion."""

    deleted_count: int = Field(
        default=0,
        description="Number of tokens successfully deleted",
    )
    failed_tokens: list[str] = Field(
        default_factory=list,
        description="List of token names that failed to delete",
    )
