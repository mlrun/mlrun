# Copyright 2025 Iguazio
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
import base64
import logging
import os
import pathlib
import typing

import pytest
import yaml

import framework.utils.singletons.k8s

if typing.TYPE_CHECKING:
    import testcontainers.k3s


@pytest.fixture(scope="session")
def k3s():
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("testcontainers").setLevel(logging.DEBUG)

    # Ensure that the testcontainers library can find the Docker socket when running inside a Docker container.
    # This env var is parsed by testcontainers at import time.
    os.environ["TESTCONTAINERS_HOST_OVERRIDE"] = "host.docker.internal"
    import testcontainers.k3s

    container = testcontainers.k3s.K3SContainer().with_kwargs(
        privileged=True,
    )
    with container:
        yield container


@pytest.fixture
def raw_kubeconfig(
    k3s: "testcontainers.k3s.K3SContainer",
) -> dict:
    return yaml.safe_load(k3s.config_yaml())


@pytest.fixture
def valid_kubeconfig_path(
    tmp_path: pathlib.Path,
    raw_kubeconfig: dict,
) -> str:
    path = tmp_path / "kubeconfig.yaml"
    yaml.safe_dump(raw_kubeconfig, path.open("w"))
    return str(path)


@pytest.fixture
def bad_ca_kubeconfig_path(
    tmp_path: pathlib.Path,
    raw_kubeconfig: dict,
) -> str:
    bad = raw_kubeconfig.copy()
    bad["clusters"][0]["cluster"]["certificate-authority-data"] = base64.b64encode(
        b"not-a-ca"
    )
    path = tmp_path / "kubeconfig-badca.yaml"
    yaml.safe_dump(bad, path.open("w"))
    return str(path)


def _k8s_helper_from_config(
    cfg_path: str,
) -> framework.utils.singletons.k8s.K8sHelper:
    return framework.utils.singletons.k8s.K8sHelper(
        kube_config_path=cfg_path,
        silent=False,
        log=False,
    )


@pytest.fixture
def invalid_ssl_ca_k8s_helper(
    bad_ca_kubeconfig_path: str,
) -> framework.utils.singletons.k8s.K8sHelper:
    return _k8s_helper_from_config(bad_ca_kubeconfig_path)


@pytest.fixture
def valid_k8s_helper(valid_kubeconfig_path):
    return _k8s_helper_from_config(valid_kubeconfig_path)
