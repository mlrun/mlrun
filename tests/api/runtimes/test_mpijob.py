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
#
import typing
import unittest.mock

from kubernetes import client as k8s_client

import mlrun.runtimes.pod
from mlrun import code_to_function, mlconf
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.runtimes.constants import MPIJobCRDVersions
from tests.api.runtimes.base import TestRuntimeBase


class TestMpiV1Runtime(TestRuntimeBase):
    def custom_setup(self):
        self.runtime_kind = "mpijob"
        self.code_handler = "test_func"
        self.name = "test-mpi-v1"
        mlconf.mpijob_crd_version = MPIJobCRDVersions.v1

    def test_run_v1_sanity(self):
        self._mock_list_pods()
        self._mock_create_namespaced_custom_object()
        self._mock_get_namespaced_custom_object()
        mpijob_function = self._generate_runtime(self.runtime_kind)
        mpijob_function.deploy()
        run = mpijob_function.run(
            artifact_path="v3io:///mypath",
            watch=False,
        )

        assert run.status.state == "running"

    def _mock_get_namespaced_custom_object(self, workers=1):
        get_k8s().crdapi.get_namespaced_custom_object = unittest.mock.Mock(
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
        get_k8s().list_pods = unittest.mock.Mock(return_value=pods)

    def _get_worker_pod(self, phase="Running"):
        return k8s_client.V1Pod(
            metadata=k8s_client.V1ObjectMeta(
                labels={
                    "kind": "mpijob",
                    "owner": "tester",
                    "v3io_user": "tester",
                    "mpijob": "v1/mpi-job-role=worker",
                },
                name=self.name,
            ),
            status=k8s_client.V1PodStatus(phase=phase),
        )

    def _get_launcher_pod(self, phase="Running"):
        return k8s_client.V1Pod(
            metadata=k8s_client.V1ObjectMeta(
                labels={
                    "kind": "mpijob",
                    "owner": "tester",
                    "v3io_user": "tester",
                    "mpijob": "v1/mpi-job-role=launcher",
                },
                name=self.name,
            ),
            status=k8s_client.V1PodStatus(phase=phase),
        )

    def _generate_runtime(
        self, kind=None, labels=None
    ) -> typing.Union[mlrun.runtimes.MpiRuntimeV1, mlrun.runtimes.MpiRuntimeV1Alpha1]:
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
