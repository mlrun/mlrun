from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SecretProviderName(str, Enum):
    vault = "vault"
    kubernetes = "kubernetes"


class SecretsData(BaseModel):
    provider: SecretProviderName = Field(SecretProviderName.vault)
    secrets: Optional[dict]


class UserSecretCreationRequest(SecretsData):
    user: str
