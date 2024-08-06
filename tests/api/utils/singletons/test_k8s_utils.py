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
    monkeypatch, run_type, mpi_version, extra_selector
):
    monkeypatch.setattr(
        server.api.runtime_handlers.mpijob,
        "cached_mpijob_crd_version",
        mpi_version or mlrun.common.runtimes.constants.MPIJobCRDVersions.default(),
    )
    uid = "test-uid"
    project = "test-project"
    namespace = "test-namespace"
    selector = (
        f"{mlrun_constants.MLRunInternalLabels.mlrun_class},"
        f"{mlrun_constants.MLRunInternalLabels.project}={project},"
        f"{mlrun_constants.MLRunInternalLabels.uid}={uid}"
    )
    if extra_selector:
        selector += f",{extra_selector}"

    k8s_helper = server.api.utils.singletons.k8s.K8sHelper(namespace, silent=True)
    k8s_helper.list_pods = unittest.mock.MagicMock()

    k8s_helper.get_logger_pods(project, uid, run_type)
    k8s_helper.list_pods.assert_called_once_with(namespace, selector=selector)


@pytest.mark.parametrize(
    "secret_data,secrets,expected",
    [
        # we want to ensure that if the data is None, the function doesn't raise an exception
        (None, {}, {}),
        (None, None, {}),

        # regular case
        ({"a": "b"}, {"a": "c"}, {'a': 'Yw=='}),
        (None, {"a": "b"}, {'a': 'Yg=='}),
    ]
)
def test_store_secret(k8s_helper, secret_data, secrets, expected):
    k8s_helper.v1api.read_namespaced_secret.return_value = unittest.mock.MagicMock(
        data=secret_data
    )
    k8s_helper.v1api.replace_namespaced_secret = unittest.mock.MagicMock()
    k8s_helper.store_secrets("my-secret", secrets)
    data = k8s_helper.v1api.replace_namespaced_secret.call_args.args[2].data
    assert data == expected
