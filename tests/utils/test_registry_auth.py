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
# (ML-12886), invoked via `python -m mlrun mint-registry-credentials`. Also covers the merge-not-
# overwrite behaviour and optional `dest` (ML-12961) that let a push-side and a pull-side exchange
# share one authfile. GAR/GCR is not here: it has no mlrun installed to call into, and runs an
# inline script instead - see server/py/services/api/tests/unit/utils/test_registry_auth.py for
# that path.

import base64
import json
import unittest.mock

import boto3
import pytest
import requests

import mlrun.errors
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
    mint_ecr_authfile(registry, str(authfile), dest=f"{registry}/myrepo:latest")

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
    mint_ecr_authfile(registry, str(authfile), dest=f"{registry}/myrepo:latest")

    assert json.loads(authfile.read_text())["auths"][registry]["auth"] == "token"


def test_mint_ecr_authfile_skips_repo_creation_without_dest(tmp_path, monkeypatch):
    # a base-image (pull-only) exchange has no reason to create a repository - and may not even
    # have permission to on a registry it's only pulling from.
    fake_client = unittest.mock.MagicMock()
    fake_client.get_authorization_token.return_value = {
        "authorizationData": [{"authorizationToken": "token"}]
    }
    monkeypatch.setattr(boto3, "client", unittest.mock.Mock(return_value=fake_client))

    authfile = tmp_path / "config.json"
    registry = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
    mint_ecr_authfile(registry, str(authfile))

    fake_client.create_repository.assert_not_called()
    assert json.loads(authfile.read_text())["auths"][registry]["auth"] == "token"


def test_mint_ecr_authfile_merges_existing_entry(tmp_path, monkeypatch):
    # a push-side exchange (different registry) may have already written to this authfile - the
    # pull-side exchange must add its entry, not clobber the existing one.
    authfile = tmp_path / "config.json"
    other_registry = "myregistry.azurecr.io"
    authfile.write_text(json.dumps({"auths": {other_registry: {"auth": "existing"}}}))

    fake_client = unittest.mock.MagicMock()
    fake_client.get_authorization_token.return_value = {
        "authorizationData": [{"authorizationToken": "token"}]
    }
    monkeypatch.setattr(boto3, "client", unittest.mock.Mock(return_value=fake_client))

    registry = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
    mint_ecr_authfile(registry, str(authfile))

    auths = json.loads(authfile.read_text())["auths"]
    assert auths[other_registry]["auth"] == "existing"
    assert auths[registry]["auth"] == "token"


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
    assert kwargs["timeout"] == 10
    fake_response.raise_for_status.assert_called_once()

    written = json.loads(authfile.read_text())
    decoded = base64.b64decode(written["auths"][registry]["auth"]).decode()
    assert decoded == "00000000-0000-0000-0000-000000000000:my-refresh-token"

    # ARM scope, not the ACR-specific resource - see _ACR_TOKEN_SCOPE in registry_auth.py for why.
    fake_credential.get_token.assert_called_once_with(
        "https://management.azure.com/.default"
    )


def test_mint_acr_authfile_requires_azure_tenant_id(tmp_path, monkeypatch):
    monkeypatch.delenv("AZURE_TENANT_ID", raising=False)

    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        mint_acr_authfile("myregistry.azurecr.io", str(tmp_path / "config.json"))


def test_mint_acr_authfile_merges_existing_entry(tmp_path, monkeypatch):
    # a push-side exchange (different registry) may have already written to this authfile - the
    # pull-side exchange must add its entry, not clobber the existing one.
    authfile = tmp_path / "config.json"
    other_registry = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
    authfile.write_text(json.dumps({"auths": {other_registry: {"auth": "existing"}}}))

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

    registry = "myregistry.azurecr.io"
    mint_acr_authfile(registry, str(authfile))

    auths = json.loads(authfile.read_text())["auths"]
    assert auths[other_registry]["auth"] == "existing"
    decoded = base64.b64decode(auths[registry]["auth"]).decode()
    assert decoded == "00000000-0000-0000-0000-000000000000:my-refresh-token"
