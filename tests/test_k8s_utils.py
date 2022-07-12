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

import unittest.mock

import pytest

import mlrun.k8s_utils
import mlrun.runtimes


@pytest.mark.parametrize(
    "run_type,mpi_version,extra_selector",
    [
        ("job", "", ""),
        ("spark", "", "spark-role=driver"),
        ("mpijob", "v1", "mpi-job-role=launcher"),
        ("mpijob", "v1alpha1", "mpi_role_type=launcher"),
    ],
)
def test_get_logger_pods_label_selector(
    monkeypatch, run_type, mpi_version, extra_selector
):
    monkeypatch.setattr(
        mlrun.runtimes.utils,
        "cached_mpijob_crd_version",
        mpi_version or mlrun.runtimes.constants.MPIJobCRDVersions.default(),
    )
    uid = "test-uid"
    project = "test-project"
    namespace = "test-namespace"
    selector = f"mlrun/class,mlrun/project={project},mlrun/uid={uid}"
    if extra_selector:
        selector += f",{extra_selector}"

    k8s_helper = mlrun.k8s_utils.K8sHelper(namespace, silent=True)
    k8s_helper.list_pods = unittest.mock.MagicMock()

    k8s_helper.get_logger_pods(project, uid, run_type)
    k8s_helper.list_pods.assert_called_once_with(namespace, selector=selector)
