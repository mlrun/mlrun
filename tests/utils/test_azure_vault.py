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

from urllib.parse import urlparse

import pytest

import mlrun.errors
from mlrun.config import config as mlconf
from mlrun.utils.azure_vault import AzureVaultStore, build_azure_vault_url


@pytest.mark.parametrize(
    "vault_name",
    [
        "abc",
        "saar-key-vault",
        "ab1",
        "a" * 24,
        "MyVault-123",
    ],
)
def test_build_azure_vault_url_accepts_valid_names(vault_name):
    url = build_azure_vault_url(vault_name)
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.hostname == f"{vault_name.lower()}.vault.azure.net"


@pytest.mark.parametrize(
    "vault_name",
    [
        "",
        "ab",
        "1abc",
        "abc-",
        "a" * 25,
        "evil.com",
        "evil.com/x",
        "evil.com/x.vault.azure.net",
        "vault.azure.net",
        "name with space",
        "name_underscore",
        "name@user",
        "name:8080",
        "../escape",
        "a/b",
        "a\\b",
    ],
)
def test_build_azure_vault_url_rejects_invalid_names(vault_name):
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        build_azure_vault_url(vault_name)


def test_azure_vault_store_rejects_host_rewrite_before_sdk_initialization():
    with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
        AzureVaultStore("evil.com/x")


def test_rejected_name_would_escape_azure_hostname():
    # documents the SSRF shape CodeQL flagged: an unvalidated name can move the host
    raw = "https://{name}.vault.azure.net".format(name="evil.com/x")
    assert urlparse(raw).hostname == "evil.com"


def test_build_azure_vault_url_supports_custom_cloud_template(monkeypatch):
    monkeypatch.setattr(
        mlconf.secret_stores.azure_vault,
        "url",
        "https://{name}.vault.usgovcloudapi.net",
    )
    url = build_azure_vault_url("gov-vault")
    assert urlparse(url).hostname == "gov-vault.vault.usgovcloudapi.net"
