# Copyright 2023 Iguazio
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

import unittest.mock

import pytest

import mlrun.common.constants as mlrun_constants
import mlrun.common.runtimes
import mlrun.common.schemas
import mlrun.runtimes
import server.api.runtime_handlers.mpijob
import server.api.utils.singletons.k8s


@pytest.fixture
def k8s_helper():
    k8s_helper = server.api.utils.singletons.k8s.K8sHelper(
        "test-namespace", silent=True
    )
    k8s_helper.v1api = unittest.mock.MagicMock()
    return k8s_helper


@pytest.mark.parametrize(
    "run_type,mpi_version,extra_selector",
    [
        ("job", "", ""),
        ("spark", "", "spark-role=driver"),
        (
            "mpijob",
            "v1",
            f"{mlrun_constants.MLRunInternalLabels.mpi_job_role}=launcher",
        ),
        (
            "mpijob",
            "v1alpha1",
            f"{mlrun_constants.MLRunInternalLabels.mpi_role_type}=launcher",
        ),
    ],
)
def test_get_logger_pods_label_selector(
    k8s_helper, monkeypatch, run_type, mpi_version, extra_selector
):
    monkeypatch.setattr(
        server.api.runtime_handlers.mpijob,
        "cached_mpijob_crd_version",
        mpi_version or mlrun.common.runtimes.constants.MPIJobCRDVersions.default(),
    )
    uid = "test-uid"
    project = "test-project"
    selector = (
        f"{mlrun_constants.MLRunInternalLabels.mlrun_class},"
        f"{mlrun_constants.MLRunInternalLabels.project}={project},"
        f"{mlrun_constants.MLRunInternalLabels.uid}={uid}"
    )
    if extra_selector:
        selector += f",{extra_selector}"

    k8s_helper.list_pods = unittest.mock.MagicMock()

    k8s_helper.get_logger_pods(project, uid, run_type)
    k8s_helper.list_pods.assert_called_once_with(
        k8s_helper.namespace, selector=selector
    )


@pytest.mark.parametrize(
    "k8s_secret_data,secrets_data",
    [
        ({"key1": "value1", "key2": "value2"}, []),
        ({"key1": "value1", "key2": "value2"}, None),
        (None, ["key1"]),
    ],
)
def test_delete_secrets_no_changes(k8s_helper, k8s_secret_data, secrets_data):
    k8s_helper.v1api.read_namespaced_secret.return_value = unittest.mock.MagicMock(
        data=k8s_secret_data
    )

    result = k8s_helper.delete_secrets("my-secret", secrets_data)

    assert result is None
    k8s_helper.v1api.read_namespaced_secret.assert_called_once_with(
        "my-secret", k8s_helper.namespace
    )
    k8s_helper.v1api.replace_namespaced_secret.assert_not_called()
    k8s_helper.v1api.delete_namespaced_secret.assert_not_called()


def test_delete_secrets_secret_found_with_changes(k8s_helper):
    secret_data = {"key1": "value1", "key2": "value2"}
    k8s_secret_mock = unittest.mock.MagicMock(data=secret_data)
    k8s_helper.v1api.read_namespaced_secret.return_value = k8s_secret_mock

    result = k8s_helper.delete_secrets("my-secret", ["key1"])

    assert result == mlrun.common.schemas.SecretEventActions.updated
    k8s_helper.v1api.read_namespaced_secret.assert_called_once_with(
        "my-secret", k8s_helper.namespace
    )
    k8s_secret_mock.data = {"key2": "value2"}
    k8s_helper.v1api.replace_namespaced_secret.assert_called_once_with(
        "my-secret", k8s_helper.namespace, k8s_secret_mock
    )
    k8s_helper.v1api.delete_namespaced_secret.assert_not_called()


def test_delete_secrets_delete_secret(k8s_helper):
    secret_data = {"key1": "value1"}
    k8s_secret_mock = unittest.mock.MagicMock(data=secret_data)
    k8s_helper.v1api.read_namespaced_secret.return_value = k8s_secret_mock

    result = k8s_helper.delete_secrets("my-secret", ["key1"])
    assert result == mlrun.common.schemas.SecretEventActions.deleted

    k8s_helper.v1api.read_namespaced_secret.assert_called_once_with(
        "my-secret", k8s_helper.namespace
    )
    k8s_helper.v1api.delete_namespaced_secret.assert_called_once_with(
        "my-secret", k8s_helper.namespace
    )
    k8s_helper.v1api.replace_namespaced_secret.assert_not_called()
