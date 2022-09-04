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
from http import HTTPStatus
from random import randrange

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
    response = client.post(f"projects/{project_name}/secrets", json=data)
    assert response.status_code == HTTPStatus.CREATED.value

    params = {"provider": schemas.SecretProviderName.vault.value, "secrets": None}
    headers = {schemas.HeaderNames.secret_store_token: user_token}

    response = client.get(
        f"projects/{project_name}/secrets", headers=headers, params=params
    )
    secrets = response.json()["secrets"]
    assert secrets == data["secrets"]
