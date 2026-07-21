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

# Tests for the Buildah cloud-registry credential exchange (ML-12886): the provider classifier and
# the generated ECR/ACR/GAR scripts. The scripts run in a *different* container (an init container,
# or - for GAR - later in the same container's push script) than mlrun's own process, so these tests
# `exec()` the generated source directly, with the underlying SDK calls (boto3 / azure-identity /
# requests / urllib) monkeypatched - this exercises the actual credential-exchange logic, not just
# the script's shape.

import base64
import json
import unittest.mock

import boto3
import pytest
import requests

from services.api.utils.builder import registry_auth


@pytest.mark.parametrize(
    "target,expected",
    [
        ("123456789012.dkr.ecr.us-east-1.amazonaws.com", "ecr"),
        ("myregistry.azurecr.io", "acr"),
        ("us-docker.pkg.dev", "gar"),
        ("gcr.io", "gar"),
        ("us.gcr.io", "gar"),
        ("index.docker.io", None),
        ("myregistry.internal.example.com", None),
        ("", None),
    ],
)
def test_classify_cloud_registry(target, expected):
    assert registry_auth.classify_cloud_registry(target) == expected


def test_ecr_credential_exchange_writes_authfile(tmp_path, monkeypatch):
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
    script = registry_auth._ecr_credential_exchange_script(
        registry, f"{registry}/myrepo:latest", str(authfile)
    )
    exec(script, {})  # noqa: S102 - exercising the generated init-container source, not user input

    boto3.client.assert_called_once_with("ecr", region_name="us-east-1")
    fake_client.create_repository.assert_any_call(repositoryName="myrepo")
    fake_client.create_repository.assert_any_call(repositoryName="myrepo/cache")
    assert json.loads(authfile.read_text()) == {
        "auths": {registry: {"auth": "QVdTOnRvcC1zZWNyZXQ="}}
    }


def test_ecr_credential_exchange_repo_create_idempotent(tmp_path, monkeypatch):
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
    script = registry_auth._ecr_credential_exchange_script(
        registry, f"{registry}/myrepo:latest", str(authfile)
    )
    # must not raise even though create_repository always errors "already exists"
    exec(script, {})  # noqa: S102

    assert json.loads(authfile.read_text())["auths"][registry]["auth"] == "token"


def test_acr_credential_exchange_writes_authfile(tmp_path, monkeypatch):
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
    script = registry_auth._acr_credential_exchange_script(registry, str(authfile))
    exec(script, {})  # noqa: S102

    requests.post.assert_called_once()
    _, kwargs = requests.post.call_args
    assert kwargs["data"]["tenant"] == "tenant-123"
    assert kwargs["data"]["service"] == registry
    assert kwargs["data"]["access_token"] == "aad-token"
    fake_response.raise_for_status.assert_called_once()

    written = json.loads(authfile.read_text())
    decoded = base64.b64decode(written["auths"][registry]["auth"]).decode()
    assert decoded == "00000000-0000-0000-0000-000000000000:my-refresh-token"


def test_gar_credential_exchange_script_uses_metadata_server_only(
    tmp_path, monkeypatch
):
    fake_response = unittest.mock.MagicMock()
    monkeypatch.setattr(
        "urllib.request.urlopen", unittest.mock.Mock(return_value=fake_response)
    )
    monkeypatch.setattr(
        "json.load", unittest.mock.Mock(return_value={"access_token": "gcp-token"})
    )

    authfile = tmp_path / "config.json"
    registry = "us-docker.pkg.dev"
    script = registry_auth.gar_credential_exchange_script(registry, str(authfile))
    # regression guard: only the stdlib the stock buildah image ships - no cloud SDK import, since
    # this runs in the Buildah main container, not an MLRun-image init container.
    assert "import google" not in script
    assert "Metadata-Flavor" in script
    exec(script, {})  # noqa: S102

    written = json.loads(authfile.read_text())
    decoded = base64.b64decode(written["auths"][registry]["auth"]).decode()
    assert decoded == "oauth2accesstoken:gcp-token"
