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

import pytest
import sqlalchemy.orm
from kubernetes import client as k8s_client

import mlrun.common.schemas
import mlrun.errors
import server.api.crud
import server.api.runtime_handlers
import server.api.utils.clients.log_collector
import server.api.utils.singletons.k8s
import tests.api.conftest


class TestRuns(tests.api.conftest.MockedK8sHelper):
    @pytest.mark.asyncio
    async def test_delete_runs_with_resources(self, db: sqlalchemy.orm.Session):
        mlrun.mlconf.log_collector.mode = mlrun.common.schemas.LogsCollectorMode.sidecar

        project = "project-name"
        server.api.crud.Runs().store_run(
            db,
            {
                "metadata": {
                    "name": "run-name",
                    "labels": {
                        "kind": "job",
                    },
                },
            },
            "uid",
            project=project,
        )
        run = server.api.crud.Runs().get_run(db, "uid", 0, project)
        assert run["metadata"]["name"] == "run-name"

        k8s_helper = server.api.utils.singletons.k8s.get_k8s_helper()
        with unittest.mock.patch.object(
            k8s_helper.v1api, "delete_namespaced_pod"
        ) as delete_namespaced_pod_mock, unittest.mock.patch.object(
            k8s_helper.v1api,
            "list_namespaced_pod",
            side_effect=[
                k8s_client.V1PodList(
                    items=[
                        k8s_client.V1Pod(
                            metadata=k8s_client.V1ObjectMeta(
                                name="pod-name",
                                labels={
                                    "mlrun/class": "job",
                                    "mlrun/project": project,
                                    "mlrun/uid": "uid",
                                },
                            ),
                            status=k8s_client.V1PodStatus(phase="Running"),
                        )
                    ]
                ),
                # 2nd time for waiting for pod to be deleted
                k8s_client.V1PodList(items=[]),
            ],
        ), unittest.mock.patch.object(
            server.api.runtime_handlers.BaseRuntimeHandler, "_ensure_run_logs_collected"
        ), unittest.mock.patch.object(
            server.api.utils.clients.log_collector.LogCollectorClient, "stop_logs"
        ) as stop_logs_mock, unittest.mock.patch.object(
            server.api.utils.clients.log_collector.LogCollectorClient, "delete_logs"
        ) as delete_logs_mock:
            await server.api.crud.Runs().delete_run(db, "uid", 0, project)
            delete_namespaced_pod_mock.assert_called_once()
            stop_logs_mock.assert_called_once()
            delete_logs_mock.assert_called_once()

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            server.api.crud.Runs().get_run(db, "uid", 0, project)
