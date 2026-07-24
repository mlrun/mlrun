# Copyright 2026 Iguazio
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

# Tests for the ECR/ACR credential-minting functions run inside a Buildah build's init container
# (ML-12886), invoked via `python -m mlrun mint-registry-credentials`. GAR/GCR is not here: it has
# no mlrun installed to call into, and runs an inline script instead - see
# server/py/services/api/tests/unit/utils/test_registry_auth.py for that path.

import base64
import json
import unittest.mock

import boto3
import requests

from mlrun.utils.registry_auth import mint_acr_authfile, mint_ecr_authfile


def test_mint_ecr_authfile_writes_authfile(tmp_path, monkeypatch):
    fake_client = unittest.mock.MagicMock()
    fake_client.exceptions.RepositoryAlreadyExistsException = type(
        "RepositoryAlreadyExistsException", (Exception,), {}
    )
    fake_client.get_authorization_token.return_value = {
        "authorizationData": [{"authorizationToken": "QVdTOnRvcC1zZWNyZXQ="}]
    }
    monkeypatch.setattr(boto3, "client", unittest.mock.Mock(return_value=fake_client))

    authfile = tmp_path / "config.json"
    registry = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
    mint_ecr_authfile(registry, f"{registry}/myrepo:latest", str(authfile))

    boto3.client.assert_called_once_with("ecr", region_name="us-east-1")
    fake_client.create_repository.assert_called_once_with(repositoryName="myrepo")
    assert json.loads(authfile.read_text()) == {
        "auths": {registry: {"auth": "QVdTOnRvcC1zZWNyZXQ="}}
    }


def test_mint_ecr_authfile_repo_create_idempotent(tmp_path, monkeypatch):
    already_exists = type("RepositoryAlreadyExistsException", (Exception,), {})
    fake_client = unittest.mock.MagicMock()
    fake_client.exceptions.RepositoryAlreadyExistsException = already_exists
    fake_client.create_repository.side_effect = already_exists()
    fake_client.get_authorization_token.return_value = {
        "authorizationData": [{"authorizationToken": "token"}]
    }
    monkeypatch.setattr(boto3, "client", unittest.mock.Mock(return_value=fake_client))

    authfile = tmp_path / "config.json"
    registry = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
    # must not raise even though create_repository always errors "already exists"
    mint_ecr_authfile(registry, f"{registry}/myrepo:latest", str(authfile))

    assert json.loads(authfile.read_text())["auths"][registry]["auth"] == "token"


def test_mint_acr_authfile_writes_authfile(tmp_path, monkeypatch):
    fake_credential = unittest.mock.MagicMock()
    fake_credential.get_token.return_value = unittest.mock.Mock(token="aad-token")
    monkeypatch.setattr(
        "azure.identity.DefaultAzureCredential",
        unittest.mock.Mock(return_value=fake_credential),
    )
    fake_response = unittest.mock.MagicMock()
    fake_response.json.return_value = {"refresh_token": "my-refresh-token"}
    monkeypatch.setattr(
        requests, "post", unittest.mock.Mock(return_value=fake_response)
    )
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant-123")

    authfile = tmp_path / "config.json"
    registry = "myregistry.azurecr.io"
    mint_acr_authfile(registry, str(authfile))

    requests.post.assert_called_once()
    _, kwargs = requests.post.call_args
    assert kwargs["data"]["tenant"] == "tenant-123"
    assert kwargs["data"]["service"] == registry
    assert kwargs["data"]["access_token"] == "aad-token"
    fake_response.raise_for_status.assert_called_once()

    written = json.loads(authfile.read_text())
    decoded = base64.b64decode(written["auths"][registry]["auth"]).decode()
    assert decoded == "00000000-0000-0000-0000-000000000000:my-refresh-token"

    # ARM ("https://management.azure.com"), not the ACR-specific resource - the latter's resource
    # principal isn't provisioned in every tenant (confirmed AADSTS500011 on a live cluster,
    # ML-12886), while ARM is universal and ACR's /oauth2/exchange accepts ARM-scoped tokens too.
    fake_credential.get_token.assert_called_once_with(
        "https://management.azure.com/.default"
    )
