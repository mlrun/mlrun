from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SecretProviderName(str, Enum):
    vault = "vault"


class SecretsData(BaseModel):
    # Currently only vault is supported. Once other providers are added, remove the const
    provider: SecretProviderName = Field(SecretProviderName.vault, const=True)
    secrets: Optional[dict]


class UserSecretCreationRequest(SecretsData):
    user: str
