from http import HTTPStatus

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from os import environ

import pytest
from mlrun import mlconf
from random import randrange

# Uncomment and set proper values for Vault test (at least one is required)
# For this test, you must also have a k8s cluster available (minikube is good enough).
# environ["MLRUN_VAULT_ROLE"] = "user:admin"
# environ["MLRUN_VAULT_TOKEN"] = <Your vault token here>


def _has_vault():
    return "MLRUN_VAULT_ROLE" in environ or "MLRUN_VAULT_TOKEN" in environ


def _set_vault_mlrun_configuration():
    mlconf.vault.url = "http://localhost:8200"
    mlconf.namespace = "default-tenant"


@pytest.mark.skipif(not _has_vault(), reason="no vault configuration")
def test_vault_secrets(db: Session, client: TestClient):
    _set_vault_mlrun_configuration()

    project_name = f"prj-{randrange(1000)}"

    data = {"provider": "vault", "secrets": {"secret1": "value1", "secret2": "value2"}}

    # Test project secrets
    response = client.post(f"/api/projects/{project_name}/secrets", json=data)
    assert response.status_code == HTTPStatus.CREATED.value
