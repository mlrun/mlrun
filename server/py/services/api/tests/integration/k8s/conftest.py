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
import tempfile
from collections.abc import Generator

import pytest
import yaml
from testcontainers.k3s import K3SContainer

import framework.utils.singletons.k8s


@pytest.fixture(autouse=True)
def force_testcontainers_host(monkeypatch):
    # this makes testcontainers skip gateway_ip() and use localhost instead
    monkeypatch.setenv("TESTCONTAINERS_HOST_OVERRIDE", "localhost")


def _server_ca_user(k3s: K3SContainer) -> tuple[str, str, dict]:
    """Return (api-server URL, base64-encoded CA bundle, user-auth dict)."""
    cfg = yaml.safe_load(k3s.config_yaml())
    cluster = cfg["clusters"][0]["cluster"]
    user = cfg["users"][0]["user"]
    return cluster["server"], cluster["certificate-authority-data"], user


def _write_kubeconfig(server: str, user: dict, ca_b64: str, path: str) -> None:
    """Write a minimal kube-config containing exactly one user/cluster/context."""
    kube_cfg = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {
                "name": "pytest",
                "cluster": {
                    "server": server,
                    **({"certificate-authority-data": ca_b64} if ca_b64 else {}),
                },
            }
        ],
        "users": [{"name": "pytest", "user": user}],
        "contexts": [
            {"name": "pytest", "context": {"cluster": "pytest", "user": "pytest"}}
        ],
        "current-context": "pytest",
    }
    with open(path, "w", encoding="utf-8") as fp:
        yaml.safe_dump(kube_cfg, fp)


def _k8shelper_from_config(cfg_path: str) -> framework.utils.singletons.k8s.K8sHelper:
    """Instantiate K8sHelper pointing at the given kube-config."""
    return framework.utils.singletons.k8s.K8sHelper(
        kube_config_path=cfg_path,
        silent=False,
        log=False,
    )


@pytest.fixture(scope="session")
def k3s() -> Generator[K3SContainer]:
    """Session-wide disposable K3s control-plane."""
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    with K3SContainer() as cluster:
        yield cluster


@pytest.fixture(scope="session")
def valid_kubeconfig_path(k3s: K3SContainer) -> str:
    server, ca_b64, user = _server_ca_user(k3s)
    fd, cfg = tempfile.mkstemp(text=True)
    os.close(fd)
    _write_kubeconfig(server, user, ca_b64, cfg)
    return cfg


@pytest.fixture(scope="session")
def bad_ca_kubeconfig_path(k3s: K3SContainer) -> str:
    server, _, user = _server_ca_user(k3s)
    fd, cfg = tempfile.mkstemp(text=True)
    os.close(fd)
    bad_ca = base64.b64encode(b"not-ca").decode()
    _write_kubeconfig(server, user, bad_ca, cfg)
    return cfg
