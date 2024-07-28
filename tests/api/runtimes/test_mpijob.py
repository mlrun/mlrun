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
#
import unittest.mock

from fastapi.testclient import TestClient
from kubernetes import client as k8s_client
from sqlalchemy.orm import Session

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import mlrun.runtimes.pod
from mlrun import code_to_function, mlconf
from mlrun.common.runtimes.constants import MPIJobCRDVersions
from server.api.utils.singletons.k8s import get_k8s_helper
from tests.api.runtimes.base import TestRuntimeBase


class TestMpiV1Runtime(TestRuntimeBase):
    def custom_setup(self):
        self.runtime_kind = "mpijob"
        self.code_handler = "test_func"
        self.name = "test-mpi-v1"
        mlconf.mpijob_crd_version = MPIJobCRDVersions.v1

    def test_run_v1_sanity(self, db: Session, client: TestClient, k8s_secrets_mock):
        mlconf.httpdb.builder.docker_registry = "localhost:5000"
        with unittest.mock.patch(
            "server.api.utils.builder.make_kaniko_pod", unittest.mock.MagicMock()
        ):
            self._mock_list_pods()
            self._mock_create_namespaced_custom_object()
            self._mock_get_namespaced_custom_object()
            mpijob_function = self._generate_runtime(self.runtime_kind)
            self.deploy(db, mpijob_function)
            run = mpijob_function.run(
                artifact_path="v3io:///mypath",
                watch=False,
                auth_info=mlrun.common.schemas.AuthInfo(),
            )

            assert run.status.state == "running"

    def _mock_get_namespaced_custom_object(self, workers=1):
        get_k8s_helper().crdapi.get_namespaced_custom_object = unittest.mock.Mock(
            return_value={
                "status": {
                    "replicaStatuses": {
                        "Launcher": {
                            "active": 1,
                        },
                        "Worker": {
                            "active": workers,
                        },
                    }
                },
            }
        )

    def _mock_list_pods(self, workers=1, pods=None, phase="Running"):
        if pods is None:
            pods = [self._get_worker_pod(phase=phase)] * workers
            pods += [self._get_launcher_pod(phase=phase)]
        get_k8s_helper().list_pods = unittest.mock.Mock(return_value=pods)

    def _get_worker_pod(self, phase="Running"):
        return k8s_client.V1Pod(
            metadata=k8s_client.V1ObjectMeta(
                labels={
                    mlrun_constants.MLRunInternalLabels.kind: "mpijob",
                    mlrun_constants.MLRunInternalLabels.owner: "tester",
                    mlrun_constants.MLRunInternalLabels.v3io_user: "tester",
                    "mpijob": f"v1/{mlrun_constants.MLRunInternalLabels.mpi_job_role}=worker",
                },
                name=self.name,
            ),
            status=k8s_client.V1PodStatus(phase=phase),
        )

    def _get_launcher_pod(self, phase="Running"):
        return k8s_client.V1Pod(
            metadata=k8s_client.V1ObjectMeta(
                labels={
                    mlrun_constants.MLRunInternalLabels.kind: "mpijob",
                    mlrun_constants.MLRunInternalLabels.owner: "tester",
                    mlrun_constants.MLRunInternalLabels.v3io_user: "tester",
                    "mpijob": f"v1/{mlrun_constants.MLRunInternalLabels.mpi_job_role}=launcher",
                },
                name=self.name,
            ),
            status=k8s_client.V1PodStatus(phase=phase),
        )

    def _generate_runtime(self, kind=None, labels=None) -> mlrun.runtimes.MpiRuntimeV1:
        runtime = code_to_function(
            name=self.name,
            project=self.project,
            filename=self.code_filename,
            handler=self.code_handler,
            kind=kind or self.runtime_kind,
            image=self.image_name,
            description="test mpijob",
            labels=labels,
        )
        return runtime
