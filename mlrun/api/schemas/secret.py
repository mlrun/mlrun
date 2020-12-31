from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class SecretProviderName(str, Enum):
    vault = "vault"


class SecretCreationRequest(BaseModel):
    # Currently only vault is supported. Once other providers are added, remove the const
    provider: SecretProviderName = Field(SecretProviderName.vault, const=True)
    secrets: Optional[dict]


class UserSecretCreationRequest(SecretCreationRequest):
    user: str

