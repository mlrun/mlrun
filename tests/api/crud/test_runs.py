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
import asyncio
import datetime
import http
import select
import time
import unittest.mock
import uuid

import fastapi.testclient
import pytest
import sqlalchemy.orm
from kubernetes import client as k8s_client

import mlrun.errors
import server.api.crud
import server.api.runtime_handlers
import server.api.utils.singletons.k8s
import tests.api.conftest


class TestRuns(tests.api.conftest.MockedK8sHelper):
    def test_delete_runs_with_resources(self, db: sqlalchemy.orm.Session):
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
        ):
            server.api.crud.Runs().delete_run(db, "uid", 0, project)
            delete_namespaced_pod_mock.assert_called_once()

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            server.api.crud.Runs().get_run(db, "uid", 0, project)

    def test_run_abortion_failure(self, db: sqlalchemy.orm.Session):
        project = "project-name"
        run_uid = str(uuid.uuid4())
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
            run_uid,
            project=project,
        )
        with unittest.mock.patch.object(
            server.api.crud.RuntimeResources(),
            "delete_runtime_resources",
            side_effect=mlrun.errors.MLRunInternalServerError("BOOM"),
        ):
            server.api.crud.Runs().abort_run(db, project, run_uid, 0)

        run = server.api.crud.Runs().get_run(db, run_uid, 0, project)
        assert run["status"]["state"] == mlrun.runtimes.constants.RunStates.error
        assert run["status"]["error"] == "Failed to abort run, error: BOOM"
