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

import pytest
import unittest.mock

import mlrun.runtimes
import mlrun.k8s_utils


@pytest.mark.parametrize(
    "run_type,mpi_version,selector_fmt",
    [
        (
            "job",
            mlrun.runtimes.constants.MPIJobCRDVersions.v1,
            "mlrun/class,mlrun/project={project},mlrun/uid={uid}",
        ),
        (
            "spark",
            mlrun.runtimes.constants.MPIJobCRDVersions.v1,
            "mlrun/class,mlrun/project={project},mlrun/uid={uid},spark-role=driver",
        ),
        (
            "mpijob",
            mlrun.runtimes.constants.MPIJobCRDVersions.v1,
            "mlrun/class,mlrun/project={project},mlrun/uid={uid},mpi-job-role=launcher",
        ),
        (
            "mpijob",
            mlrun.runtimes.constants.MPIJobCRDVersions.v1alpha1,
            "mlrun/class,mlrun/project={project},mlrun/uid={uid},mpi_role_type=launcher",
        ),
    ],
)
def test_get_logger_pods(monkeypatch, run_type, mpi_version, selector_fmt):
    monkeypatch.setattr(
        mlrun.runtimes.utils,
        "cached_mpijob_crd_version",
        mpi_version,
    )
    uid = "test-uid"
    project = "test-project"
    namespace = "test-namespace"
    selector = selector_fmt.format(project=project, uid=uid)

    k8s_helper = mlrun.k8s_utils.K8sHelper(namespace)
    k8s_helper.list_pods = unittest.mock.MagicMock()

    k8s_helper.get_logger_pods(project, uid, run_type)
    k8s_helper.list_pods.assert_called_once_with(namespace, selector=selector)
