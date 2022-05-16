from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SecretProviderName(str, Enum):
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
