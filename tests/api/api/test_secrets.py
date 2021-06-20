from http import HTTPStatus
from random import randrange

import deepdiff
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from mlrun import mlconf
from mlrun.api import schemas

# Set a valid Vault token to run this test.
# For this test, you must also have a k8s cluster available (minikube is good enough).
user_token = ""


def _set_vault_mlrun_configuration():
    mlconf.secret_stores.vault.url = "http://localhost:8200"
    mlconf.namespace = "default-tenant"
    mlconf.secret_stores.vault.user_token = user_token


@pytest.mark.skipif(user_token == "", reason="no vault configuration")
def test_vault_create_project_secrets(db: Session, client: TestClient):
    _set_vault_mlrun_configuration()

    project_name = f"prj-{randrange(1000)}"

    data = {"provider": "vault", "secrets": {"secret1": "value1", "secret2": "value2"}}

    # Test project secrets
    response = client.post(f"/api/projects/{project_name}/secrets", json=data)
    assert response.status_code == HTTPStatus.CREATED.value

    params = {"provider": schemas.SecretProviderName.vault.value, "secrets": None}
    headers = {schemas.HeaderNames.secret_store_token: user_token}

    response = client.get(
        f"/api/projects/{project_name}/secrets", headers=headers, params=params
    )
    secrets = response.json()["secrets"]
    assert secrets == data["secrets"]


# Set a namespace here to run the test_k8s_project_secrets test.
k8s_namespace = ""


# Need to be a fixture since the k8s_helper is initialized as part of te TestClient, so need
# to perform this configuration prior to that, to make k8s_helper use proper namespace.
@pytest.fixture(autouse=True)
def set_k8s_namespace():
    mlconf.namespace = k8s_namespace


@pytest.mark.skipif(k8s_namespace == "", reason="no k8s namespace defined")
def test_k8s_project_secrets(db: Session, client: TestClient):
    project_name = "project1"
    secrets = {"secret1": "value1", "secret2": "value2"}
    data = {"provider": "kubernetes", "secrets": secrets}
    expected_results = {
        "provider": "kubernetes",
        "secrets": {key: None for key in secrets},
    }

    # Clean up k8s secret if exists. Delete with no secret provider will cleanup all
    response = client.delete(
        f"/api/projects/{project_name}/secrets?provider=kubernetes"
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value
    response = client.get(f"/api/projects/{project_name}/secrets?provider=kubernetes")
    assert response.status_code == HTTPStatus.OK.value
    assert (
        deepdiff.DeepDiff(response.json(), {"provider": "kubernetes", "secrets": {}})
        == {}
    )

    response = client.post(f"/api/projects/{project_name}/secrets", json=data)
    assert response.status_code == HTTPStatus.CREATED.value

    response = client.get(f"/api/projects/{project_name}/secrets?provider=kubernetes")
    assert response.status_code == HTTPStatus.OK.value
    assert deepdiff.DeepDiff(response.json(), expected_results) == {}

    # Add a secret key
    add_secret_data = {"provider": "kubernetes", "secrets": {"secret3": "mySecret!!!"}}
    response = client.post(
        f"/api/projects/{project_name}/secrets", json=add_secret_data
    )
    assert response.status_code == HTTPStatus.CREATED.value

    expected_results["secrets"]["secret3"] = None
    response = client.get(f"/api/projects/{project_name}/secrets?provider=kubernetes")
    assert deepdiff.DeepDiff(response.json(), expected_results) == {}

    # Delete a single secret
    response = client.delete(
        f"/api/projects/{project_name}/secrets?provider=kubernetes&secret=secret1"
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value

    expected_results["secrets"].pop("secret1")
    response = client.get(f"/api/projects/{project_name}/secrets?provider=kubernetes")
    assert response.status_code == HTTPStatus.OK.value
    assert deepdiff.DeepDiff(response.json(), expected_results) == {}

    # Cleanup
    response = client.delete(
        f"/api/projects/{project_name}/secrets?provider=kubernetes"
    )
    assert response.status_code == HTTPStatus.NO_CONTENT.value
