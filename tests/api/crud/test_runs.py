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
import uuid

import deepdiff
import pytest
import sqlalchemy.orm
from kubernetes import client as k8s_client

import mlrun.common.constants as mlrun_constants
import mlrun.common.runtimes.constants
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
                    "uid": "uid",
                    "labels": {
                        mlrun_constants.MLRunInternalLabels.kind: "job",
                    },
                },
            },
            "uid",
            project=project,
        )
        run = server.api.crud.Runs().get_run(db, "uid", 0, project)
        assert run["metadata"]["name"] == "run-name"

        k8s_helper = server.api.utils.singletons.k8s.get_k8s_helper()
        with (
            unittest.mock.patch.object(
                k8s_helper.v1api, "delete_namespaced_pod"
            ) as delete_namespaced_pod_mock,
            unittest.mock.patch.object(
                k8s_helper.v1api,
                "list_namespaced_pod",
                side_effect=[
                    k8s_client.V1PodList(
                        items=[
                            k8s_client.V1Pod(
                                metadata=k8s_client.V1ObjectMeta(
                                    name="pod-name",
                                    labels={
                                        mlrun_constants.MLRunInternalLabels.mlrun_class: "job",
                                        mlrun_constants.MLRunInternalLabels.project: project,
                                        mlrun_constants.MLRunInternalLabels.uid: "uid",
                                    },
                                ),
                                status=k8s_client.V1PodStatus(phase="Running"),
                            )
                        ],
                        metadata=k8s_client.V1ListMeta(),
                    ),
                    # 2nd time for waiting for pod to be deleted
                    k8s_client.V1PodList(items=[], metadata=k8s_client.V1ListMeta()),
                ],
            ),
            unittest.mock.patch.object(
                server.api.runtime_handlers.BaseRuntimeHandler,
                "_ensure_run_logs_collected",
            ),
            unittest.mock.patch.object(
                server.api.utils.clients.log_collector.LogCollectorClient, "delete_logs"
            ) as delete_logs_mock,
        ):
            await server.api.crud.Runs().delete_run(db, "uid", 0, project)
            delete_namespaced_pod_mock.assert_called_once()
            delete_logs_mock.assert_called_once()

        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            server.api.crud.Runs().get_run(db, "uid", 0, project)

    @pytest.mark.asyncio
    async def test_delete_runs(self, db: sqlalchemy.orm.Session):
        mlrun.mlconf.log_collector.mode = mlrun.common.schemas.LogsCollectorMode.sidecar

        project = "project-name"
        run_name = "run-name"
        for uid in range(20):
            server.api.crud.Runs().store_run(
                db,
                {
                    "metadata": {
                        "name": run_name,
                        "labels": {
                            mlrun_constants.MLRunInternalLabels.kind: "job",
                        },
                        "uid": str(uid),
                        "iteration": 0,
                    },
                },
                str(uid),
                project=project,
            )

        runs = server.api.crud.Runs().list_runs(db, run_name, project=project)
        assert len(runs) == 20

        k8s_helper = server.api.utils.singletons.k8s.get_k8s_helper()
        with (
            unittest.mock.patch.object(
                k8s_helper.v1api, "delete_namespaced_pod"
            ) as delete_namespaced_pod_mock,
            unittest.mock.patch.object(
                k8s_helper.v1api,
                "list_namespaced_pod",
                return_value=k8s_client.V1PodList(
                    items=[], metadata=k8s_client.V1ListMeta()
                ),
            ),
            unittest.mock.patch.object(
                server.api.runtime_handlers.BaseRuntimeHandler,
                "_ensure_run_logs_collected",
            ),
            unittest.mock.patch.object(
                server.api.utils.clients.log_collector.LogCollectorClient, "delete_logs"
            ) as delete_logs_mock,
        ):
            await server.api.crud.Runs().delete_runs(db, name=run_name, project=project)
            runs = server.api.crud.Runs().list_runs(db, run_name, project=project)
            assert len(runs) == 0
            delete_namespaced_pod_mock.assert_not_called()
            assert delete_logs_mock.call_count == 20

    @pytest.mark.asyncio
    async def test_delete_runs_failure(self, db: sqlalchemy.orm.Session):
        """
        This test creates 3 runs, and then tries to delete them.
        The first run is deleted successfully, the second one fails with an exception, and the third one is deleted
        """
        mlrun.mlconf.log_collector.mode = mlrun.common.schemas.LogsCollectorMode.sidecar
        project = "project-name"
        run_name = "run-name"
        for uid in range(3):
            server.api.crud.Runs().store_run(
                db,
                {
                    "metadata": {
                        "name": run_name,
                        "labels": {
                            mlrun_constants.MLRunInternalLabels.kind: "job",
                        },
                        "uid": str(uid),
                        "iteration": 0,
                    },
                },
                str(uid),
                project=project,
            )

        runs = server.api.crud.Runs().list_runs(db, run_name, project=project)
        assert len(runs) == 3

        k8s_helper = server.api.utils.singletons.k8s.get_k8s_helper()
        with (
            unittest.mock.patch.object(k8s_helper.v1api, "delete_namespaced_pod"),
            unittest.mock.patch.object(
                k8s_helper.v1api,
                "list_namespaced_pod",
                side_effect=[
                    k8s_client.V1PodList(items=[], metadata=k8s_client.V1ListMeta()),
                    Exception("Boom!"),
                    k8s_client.V1PodList(items=[], metadata=k8s_client.V1ListMeta()),
                ],
            ),
            unittest.mock.patch.object(
                server.api.runtime_handlers.BaseRuntimeHandler,
                "_ensure_run_logs_collected",
            ),
            unittest.mock.patch.object(
                server.api.utils.clients.log_collector.LogCollectorClient, "delete_logs"
            ) as delete_logs_mock,
        ):
            with pytest.raises(mlrun.errors.MLRunBadRequestError) as exc:
                await server.api.crud.Runs().delete_runs(
                    db, name=run_name, project=project
                )
            assert "Failed to delete 1 run(s). Error: Boom!" in str(exc.value)
            assert delete_logs_mock.call_count == 2

            runs = server.api.crud.Runs().list_runs(db, run_name, project=project)
            assert len(runs) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "run_state",
        [
            mlrun.common.runtimes.constants.RunStates.running,
            mlrun.common.runtimes.constants.RunStates.pending,
        ],
    )
    async def test_delete_run_failure(self, db: sqlalchemy.orm.Session, run_state):
        project = "project-name"
        server.api.crud.Runs().store_run(
            db,
            {
                "metadata": {
                    "name": "run-name",
                    "labels": {
                        mlrun_constants.MLRunInternalLabels.kind: "job",
                    },
                },
                "status": {"state": run_state},
            },
            "uid",
            project=project,
        )
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
            await server.api.crud.Runs().delete_run(db, "uid", 0, project)

        assert (
            f"Can not delete run in {run_state} state, consider aborting the run first"
            in str(exc.value)
        )

    def test_run_abortion_failure(self, db: sqlalchemy.orm.Session):
        project = "project-name"
        run_uid = str(uuid.uuid4())
        server.api.crud.Runs().store_run(
            db,
            {
                "metadata": {
                    "name": "run-name",
                    "uid": run_uid,
                    "labels": {
                        mlrun_constants.MLRunInternalLabels.kind: "job",
                    },
                },
            },
            run_uid,
            project=project,
        )
        with (
            unittest.mock.patch.object(
                server.api.crud.RuntimeResources(),
                "delete_runtime_resources",
                side_effect=mlrun.errors.MLRunInternalServerError("BOOM"),
            ),
            pytest.raises(mlrun.errors.MLRunInternalServerError) as exc,
        ):
            server.api.crud.Runs().abort_run(db, project, run_uid, 0)
        assert "BOOM" == str(exc.value)

        run = server.api.crud.Runs().get_run(db, run_uid, 0, project)
        assert run["status"]["state"] == mlrun.common.runtimes.constants.RunStates.error
        assert run["status"]["error"] == "Failed to abort run, error: BOOM"

    def test_store_run_strip_artifacts_metadata(self, db: sqlalchemy.orm.Session):
        project = "project-name"
        run_uid = str(uuid.uuid4())
        server.api.crud.Runs().store_run(
            db,
            {
                "metadata": {
                    "name": "run-name",
                    "labels": {
                        mlrun_constants.MLRunInternalLabels.kind: "job",
                    },
                },
                "status": {
                    "artifact_uris": {
                        "key1": "this should be replaced",
                        "key2": "store://artifacts/project-name/db_key2@tree2",
                    },
                    "artifacts": [
                        {
                            "metadata": {
                                "key": "key1",
                                "tree": "tree1",
                                "uid": "uid1",
                                "project": project,
                            },
                            "spec": {
                                "db_key": "db_key1",
                            },
                        },
                        {
                            "metadata": {
                                "key": "key3",
                                "tree": "tree3",
                                "uid": "uid3",
                                "project": project,
                            },
                            "spec": {
                                "db_key": "db_key3",
                            },
                        },
                    ],
                },
            },
            run_uid,
            project=project,
        )

        runs = server.api.crud.Runs().list_runs(db, project=project)
        assert len(runs) == 1
        run = runs[0]
        assert "artifacts" not in run["status"]
        assert run["status"]["artifact_uris"] == {
            "key1": "store://artifacts/project-name/db_key1@tree1",
            "key2": "store://artifacts/project-name/db_key2@tree2",
            "key3": "store://artifacts/project-name/db_key3@tree3",
        }

    def test_update_run_strip_artifacts_metadata(self, db: sqlalchemy.orm.Session):
        project = "project-name"
        run_uid = str(uuid.uuid4())
        server.api.crud.Runs().store_run(
            db,
            {
                "metadata": {
                    "name": "run-name",
                    "labels": {
                        mlrun_constants.MLRunInternalLabels.kind: "job",
                    },
                },
                "status": {
                    "artifact_uris": {
                        "key1": "this should be replaced",
                        "key2": "store://artifacts/project-name/db_key2@tree2",
                    },
                },
            },
            run_uid,
            project=project,
        )

        server.api.crud.Runs().update_run(
            db,
            project,
            run_uid,
            iter=0,
            data={
                "status.artifact_uris": {
                    "key1": "this should be replaced",
                    "key2": "store://artifacts/project-name/db_key2@tree2",
                },
                "status.artifacts": [
                    {
                        "metadata": {
                            "key": "key1",
                            "tree": "tree1",
                            "uid": "uid1",
                            "project": project,
                        },
                        "spec": {
                            "db_key": "db_key1",
                        },
                    },
                    {
                        "metadata": {
                            "key": "key3",
                            "tree": "tree3",
                            "uid": "uid3",
                            "project": project,
                        },
                        "spec": {
                            "db_key": "db_key3",
                        },
                    },
                ],
            },
        )

        runs = server.api.crud.Runs().list_runs(db, project=project)
        assert len(runs) == 1
        run = runs[0]
        assert "artifacts" not in run["status"]
        assert run["status"]["artifact_uris"] == {
            "key1": "store://artifacts/project-name/db_key1@tree1",
            "key2": "store://artifacts/project-name/db_key2@tree2",
            "key3": "store://artifacts/project-name/db_key3@tree3",
        }

    def test_get_run_restore_artifacts_metadata(self, db: sqlalchemy.orm.Session):
        project = "project-name"
        run_uid = str(uuid.uuid4())
        artifacts = self._generate_artifacts(project, run_uid)

        for artifact in artifacts:
            server.api.crud.Artifacts().store_artifact(
                db,
                artifact["spec"]["db_key"],
                artifact,
                project=project,
            )

        server.api.crud.Runs().store_run(
            db,
            {
                "metadata": {
                    "name": "run-name",
                    "uid": run_uid,
                    "labels": {
                        mlrun_constants.MLRunInternalLabels.kind: "job",
                    },
                },
                "status": {
                    "artifacts": artifacts,
                },
            },
            run_uid,
            project=project,
        )

        self._validate_run_artifacts(artifacts, db, project, run_uid)

    def test_get_workflow_run_restore_artifacts_metadata(
        self, db: sqlalchemy.orm.Session
    ):
        project = "project-name"
        run_uid = str(uuid.uuid4())
        workflow_uid = str(uuid.uuid4())
        artifacts = self._generate_artifacts(project, run_uid, workflow_uid)

        for artifact in artifacts:
            server.api.crud.Artifacts().store_artifact(
                db,
                artifact["spec"]["db_key"],
                artifact,
                iter=artifact["metadata"]["iter"],
                project=project,
                producer_id=workflow_uid,
            )

        server.api.crud.Runs().store_run(
            db,
            {
                "metadata": {
                    "name": "run-name",
                    "uid": run_uid,
                    "labels": {
                        mlrun_constants.MLRunInternalLabels.kind: "job",
                        mlrun_constants.MLRunInternalLabels.workflow: workflow_uid,
                    },
                },
                "status": {
                    "artifacts": artifacts,
                },
            },
            run_uid,
            project=project,
        )

        self._validate_run_artifacts(artifacts, db, project, run_uid)

    @pytest.mark.parametrize("workflow_id", [None, str(uuid.uuid4())])
    def test_get_run_iteration_restore_artifacts_metadata(
        self, db: sqlalchemy.orm.Session, workflow_id
    ):
        project = "project-name"
        run_uid = str(uuid.uuid4())
        workflow_uid = workflow_id
        iter = 3
        artifacts = self._generate_artifacts(project, run_uid, workflow_uid, iter=iter)

        for artifact in artifacts:
            server.api.crud.Artifacts().store_artifact(
                db,
                artifact["spec"]["db_key"],
                artifact,
                iter=iter,
                project=project,
            )

        labels = {mlrun_constants.MLRunInternalLabels.kind: "job"}
        if workflow_id:
            labels[mlrun_constants.MLRunInternalLabels.workflow] = workflow_id

        server.api.crud.Runs().store_run(
            db,
            {
                "metadata": {
                    "name": "run-name",
                    "uid": run_uid,
                    "iter": iter,
                    "labels": labels,
                },
                "status": {
                    "artifacts": artifacts,
                },
            },
            run_uid,
            iter=iter,
            project=project,
        )

        self._validate_run_artifacts(artifacts, db, project, run_uid, iter)

    def test_get_workflow_run_best_iteration_restore_artifacts_metadata(
        self, db: sqlalchemy.orm.Session
    ):
        project = "project-name"
        run_uid = str(uuid.uuid4())
        workflow_uid = str(uuid.uuid4())

        # Create 5 best iteration artifacts
        best_iteration = 3
        best_iteration_count = 5
        best_iteration_artifacts = self._generate_artifacts(
            project,
            run_uid,
            workflow_uid,
            artifacts_len=best_iteration_count,
            iter=best_iteration,
        )

        for artifact in best_iteration_artifacts:
            server.api.utils.singletons.db.get_db().store_artifact(
                db,
                artifact["spec"]["db_key"],
                artifact,
                None,
                iter=best_iteration,
                tag="latest",
                project=project,
                best_iteration=True,
            )

        # Create 3 bad iteration artifacts
        bad_iteration = 5
        bad_iteration_count = 3
        bad_iteration_artifacts = self._generate_artifacts(
            project,
            run_uid,
            workflow_uid,
            artifacts_len=bad_iteration_count,
            iter=bad_iteration,
            key_prefix="bad_key",
        )
        for artifact in bad_iteration_artifacts:
            server.api.crud.Artifacts().store_artifact(
                db,
                artifact["spec"]["db_key"],
                artifact,
                iter=bad_iteration,
                project=project,
            )

        # Create 1 artifact for the parent run (iteration 0) this should be part of the final result
        parent_run_count = 1
        parent_run_arts = self._generate_artifacts(
            project,
            run_uid,
            workflow_uid,
            artifacts_len=parent_run_count,
            key_prefix="parent_key",
        )
        for artifact in parent_run_arts:
            server.api.crud.Artifacts().store_artifact(
                db,
                artifact["spec"]["db_key"],
                artifact,
                project=project,
            )

        server.api.crud.Runs().store_run(
            db,
            {
                "metadata": {
                    "name": "run-name",
                    "uid": run_uid,
                    "labels": {
                        mlrun_constants.MLRunInternalLabels.kind: "job",
                        mlrun_constants.MLRunInternalLabels.workflow: workflow_uid,
                    },
                },
                "status": {
                    "artifacts": best_iteration_artifacts + parent_run_arts,
                },
            },
            run_uid,
            project=project,
        )

        self._validate_run_artifacts(
            best_iteration_artifacts + parent_run_arts, db, project, run_uid
        )

    @pytest.mark.parametrize("workflow_uid", [None, str(uuid.uuid4())])
    def test_get_run_restore_artifacts_metadata_with_missing_artifact(
        self, db: sqlalchemy.orm.Session, workflow_uid
    ):
        project = "project-name"
        run_uid = str(uuid.uuid4())
        artifacts = self._generate_artifacts(
            project, run_uid, workflow_uid, artifacts_len=3
        )

        # Create only 2 artifacts
        for artifact in artifacts[:2]:
            server.api.crud.Artifacts().store_artifact(
                db,
                artifact["spec"]["db_key"],
                artifact,
                iter=artifact["metadata"]["iter"],
                project=project,
                producer_id=workflow_uid or run_uid,
            )

        labels = {mlrun_constants.MLRunInternalLabels.kind: "job"}
        if workflow_uid:
            labels[mlrun_constants.MLRunInternalLabels.workflow] = workflow_uid

        server.api.crud.Runs().store_run(
            db,
            {
                "metadata": {
                    "name": "run-name",
                    "uid": run_uid,
                    "labels": labels,
                },
                "status": {
                    "artifacts": artifacts,
                },
            },
            run_uid,
            project=project,
        )

        # Expect only the 2 artifacts to be restored
        self._validate_run_artifacts(artifacts[:2], db, project, run_uid)

    @pytest.mark.parametrize(
        "run_format",
        [
            mlrun.common.formatters.RunFormat.full,
            mlrun.common.formatters.RunFormat.standard,
        ],
    )
    def test_run_formats(
        self, db: sqlalchemy.orm.Session, run_format: mlrun.common.formatters.RunFormat
    ):
        project = "project-name"
        run_uid = str(uuid.uuid4())
        artifacts = self._generate_artifacts(project, run_uid)

        for artifact in artifacts:
            server.api.crud.Artifacts().store_artifact(
                db,
                artifact["spec"]["db_key"],
                artifact,
                project=project,
            )

        server.api.crud.Runs().store_run(
            db,
            {
                "metadata": {
                    "name": "run-name",
                    "uid": run_uid,
                    "labels": {mlrun_constants.MLRunInternalLabels.kind: "job"},
                },
                "status": {
                    "artifacts": artifacts,
                },
            },
            run_uid,
            project=project,
        )

        expected_artifacts = artifacts
        if run_format == mlrun.common.formatters.RunFormat.standard:
            expected_artifacts = []
        self._validate_run_artifacts(
            expected_artifacts, db, project, run_uid, run_format=run_format
        )

    def test_get_workflow_run_no_artifacts(self, db: sqlalchemy.orm.Session):
        project = "project-name"
        run_uid_1 = str(uuid.uuid4())
        run_uid_2 = str(uuid.uuid4())
        workflow_uid = str(uuid.uuid4())
        iter = 0

        # Create some artifacts with different producer id
        artifacts = self._generate_artifacts(
            project, str(uuid.uuid4()), str(uuid.uuid4()), artifacts_len=3
        )

        for artifact in artifacts:
            server.api.crud.Artifacts().store_artifact(
                db,
                artifact["spec"]["db_key"],
                artifact,
                iter=artifact["metadata"]["iter"],
                project=project,
                producer_id=str(uuid.uuid4()),
            )

        # run_uid_1 should not list artifacts as it has none
        server.api.crud.Runs().store_run(
            db,
            {
                "metadata": {
                    "name": "run-name",
                    "uid": run_uid_1,
                    "iter": iter,
                    "labels": {
                        mlrun_constants.MLRunInternalLabels.kind: "job",
                        mlrun_constants.MLRunInternalLabels.workflow: workflow_uid,
                    },
                },
                "status": {},
            },
            run_uid_1,
            project=project,
        )

        # run_uid_2 should not list artifacts as the artifacts has different producer id
        server.api.crud.Runs().store_run(
            db,
            {
                "metadata": {
                    "name": "run-name",
                    "uid": run_uid_1,
                    "iter": iter,
                    "labels": {
                        mlrun_constants.MLRunInternalLabels.kind: "job",
                        mlrun_constants.MLRunInternalLabels.workflow: workflow_uid,
                    },
                },
                "status": {"artifacts": artifacts},
            },
            run_uid_2,
            project=project,
        )

        with unittest.mock.patch(
            "server.api.crud.Artifacts.list_artifacts_for_producer_id",
            side_effect=Exception("Should not be called"),
        ):
            run_1 = server.api.crud.Runs().get_run(
                db,
                run_uid_1,
                iter,
                project,
                format_=mlrun.common.formatters.RunFormat.full,
            )

            assert "artifacts" not in run_1["status"]
            assert "artifact_uris" not in run_1["status"]

            run_2 = server.api.crud.Runs().get_run(
                db,
                run_uid_2,
                iter,
                project,
                format_=mlrun.common.formatters.RunFormat.full,
            )

            assert "artifacts" not in run_2["status"]
            # run 2 should still have artifact uris even if the producer id is different
            assert "artifact_uris" in run_2["status"]

    def test_get_run_notifications_format(self, db: sqlalchemy.orm.Session):
        project = "project-name"
        run_uid = str(uuid.uuid4())
        notifications = self._generate_notifications()

        server.api.crud.Runs().store_run(
            db,
            {
                "metadata": {
                    "name": "run-name",
                    "uid": run_uid,
                    "labels": {mlrun_constants.MLRunInternalLabels.kind: "job"},
                },
            },
            run_uid,
            project=project,
        )

        server.api.crud.Notifications().store_run_notifications(
            session=db,
            notification_objects=notifications,
            run_uid=run_uid,
            project=project,
        )

        run = server.api.crud.Runs().get_run(
            db,
            run_uid,
            0,
            project,
            format_=mlrun.common.formatters.RunFormat.notifications,
        )

        assert "notifications" in run["status"]
        assert "notifications" in run["spec"]
        for notification in run["spec"]["notifications"]:
            assert "params" in notification

    @staticmethod
    def _generate_notifications(
        notifications_len=2,
    ):
        notifications = []
        i = 0
        while len(notifications) < notifications_len:
            notification = {
                "kind": "webhook",
                "condition": "",
                "severity": "verbose",
                "params": {
                    "url": "https://webhook.site/3c81ac80-1767-490f-bda3-a241fae47f43",
                    "method": "POST",
                    "verify_ssl": True,
                },
                "name": str(uuid.uuid4()),
                "when": ["completed", "error", "running"],
                "message": "Check1",
            }
            notification = mlrun.model.Notification.from_dict(notification)
            notifications.append(notification)
            i += 1
        return notifications

    @staticmethod
    def _generate_artifacts(
        project,
        run_uid,
        workflow_uid=None,
        artifacts_len=2,
        iter=None,
        key_prefix="key",
    ):
        artifacts = []
        i = 0
        while len(artifacts) < artifacts_len:
            artifact = {
                mlrun_constants.MLRunInternalLabels.kind: "artifact",
                "metadata": {
                    "key": f"{key_prefix}{i}",
                    "tree": workflow_uid or run_uid,
                    "uid": f"uid{i}",
                    "project": project,
                    "iter": iter or 0,
                    "tag": "latest",
                },
                "spec": {
                    "db_key": f"db_key_{key_prefix}{i}",
                },
                "status": {},
            }
            if workflow_uid:
                producer_uri = f"{project}/{run_uid}"
                if iter:
                    producer_uri += f"-{iter}"
                artifact["spec"]["producer"] = {
                    "uri": producer_uri,
                }
            artifacts.append(artifact)
            i += 1
        return artifacts

    @staticmethod
    def _validate_run_artifacts(
        artifacts,
        db,
        project,
        run_uid,
        iter=0,
        run_format: mlrun.common.formatters.RunFormat = None,
    ):
        run = server.api.crud.Runs().get_run(
            db,
            run_uid,
            iter,
            project,
            format_=run_format or mlrun.common.formatters.RunFormat.full,
        )

        enriched_artifacts = []
        if artifacts:
            assert "artifacts" in run["status"]
            enriched_artifacts = list(run["status"]["artifacts"])

        def sort_by_key(e):
            return e["metadata"]["key"]

        assert len(enriched_artifacts) == len(
            artifacts
        ), "Number of artifacts is different"
        enriched_artifacts.sort(key=sort_by_key)
        artifacts.sort(key=sort_by_key)
        for artifact, enriched_artifact in zip(artifacts, enriched_artifacts):
            assert (
                deepdiff.DeepDiff(
                    artifact,
                    enriched_artifact,
                    exclude_paths="root['metadata']['tag']",
                )
                == {}
            )
