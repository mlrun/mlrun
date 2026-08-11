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

# Tests for the Buildah cloud-registry credential exchange (ML-12886): the provider classifier, the
# ECR/ACR init-container wiring (they invoke `python -m mlrun mint-registry-credentials` - see
# tests/utils/test_registry_auth.py for the actual credential-exchange logic those calls run), and
# the generated GAR/GCR script (no mlrun installed where it runs, so it stays a generated script
# `exec()`'d directly here with the underlying stdlib call mocked). Also covers the optional `dest`
# and `container_name` (ML-12961) that let a pull-side (base-image) exchange run alongside a
# push-side one in the same pod - see test_builder_backend.py for the make_buildah_pod-level wiring.

import base64
import json
import unittest.mock

import pytest

from mlrun.config import config

import framework.utils.singletons.k8s
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


def _init_container(pod) -> object:
    assert len(pod.init_containers) == 1
    return pod.init_containers[0]


def test_append_ecr_credential_exchange_init_container():
    pod = framework.utils.singletons.k8s.BasePod(task_name="t", image="img")
    registry = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
    dest = f"{registry}/myrepo:latest"
    registry_auth.append_ecr_credential_exchange_init_container(
        pod, registry, "/auth/config.json", dest=dest
    )

    container = _init_container(pod)
    assert container.name == "registry-credential-exchange"
    # same python -m mlrun <subcommand> convention Kaniko's source-fetch init container uses -
    # mlrun is installed on the init container's image, so there's no need to inline a script.
    assert container.command == ["python"]
    assert container.args == [
        "-m",
        "mlrun",
        "mint-registry-credentials",
        "--provider",
        "ecr",
        "--registry",
        registry,
        "--dest",
        dest,
        "--authfile",
        "/auth/config.json",
    ]


def test_append_ecr_credential_exchange_init_container_pull_only_omits_dest():
    # a base-image (pull-only) exchange has no dest - no reason to create a repository, and the
    # container name must differ so it can coexist with a push-side exchange in the same pod.
    pod = framework.utils.singletons.k8s.BasePod(task_name="t", image="img")
    registry = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
    registry_auth.append_ecr_credential_exchange_init_container(
        pod,
        registry,
        "/auth/config.json",
        container_name=registry_auth.PULL_CREDENTIAL_EXCHANGE_INIT_CONTAINER_NAME,
    )

    container = _init_container(pod)
    assert container.name == "registry-credential-exchange-pull"
    assert "--dest" not in container.args
    assert container.args == [
        "-m",
        "mlrun",
        "mint-registry-credentials",
        "--provider",
        "ecr",
        "--registry",
        registry,
        "--authfile",
        "/auth/config.json",
    ]


def test_append_acr_credential_exchange_init_container():
    pod = framework.utils.singletons.k8s.BasePod(task_name="t", image="img")
    registry = "myregistry.azurecr.io"
    registry_auth.append_acr_credential_exchange_init_container(
        pod, registry, "/auth/config.json"
    )

    container = _init_container(pod)
    assert container.name == "registry-credential-exchange"
    assert container.command == ["python"]
    assert container.args == [
        "-m",
        "mlrun",
        "mint-registry-credentials",
        "--provider",
        "acr",
        "--registry",
        registry,
        "--authfile",
        "/auth/config.json",
    ]


def test_append_acr_credential_exchange_init_container_custom_container_name():
    # a pull-side (base-image) exchange running alongside a push-side one in the same pod needs a
    # distinct container name - k8s requires unique container names within a pod.
    pod = framework.utils.singletons.k8s.BasePod(task_name="t", image="img")
    registry = "myregistry.azurecr.io"
    registry_auth.append_acr_credential_exchange_init_container(
        pod,
        registry,
        "/auth/config.json",
        container_name=registry_auth.PULL_CREDENTIAL_EXCHANGE_INIT_CONTAINER_NAME,
    )

    assert _init_container(pod).name == "registry-credential-exchange-pull"


def test_credential_exchange_image_defaults_to_default_base_image(monkeypatch):
    monkeypatch.setattr(config.httpdb.builder, "registry_credential_exchange_image", "")
    monkeypatch.setattr(config, "default_base_image", "myorg/custom-mlrun:v9")

    assert registry_auth._credential_exchange_image() == "myorg/custom-mlrun:v9"


def test_credential_exchange_image_explicit_override(monkeypatch):
    monkeypatch.setattr(
        config.httpdb.builder, "registry_credential_exchange_image", "myorg/other:v1"
    )

    assert registry_auth._credential_exchange_image() == "myorg/other:v1"


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
    # regression guard: only the stdlib the stock buildah image ships - no mlrun/cloud SDK import,
    # since this runs in the Buildah main container, which has neither installed.
    assert "import mlrun" not in script
    assert "import google" not in script
    assert "Metadata-Flavor" in script
    exec(script, {})  # noqa: S102 - exercising the generated script, not user input

    written = json.loads(authfile.read_text())
    decoded = base64.b64decode(written["auths"][registry]["auth"]).decode()
    assert decoded == "oauth2accesstoken:gcp-token"
