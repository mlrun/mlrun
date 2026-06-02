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

import os.path
import unittest.mock

import pytest
import sqlalchemy.orm

import mlrun.common.schemas

import services.api.crud
import services.api.tests.unit.conftest


class TestWorkflows(services.api.tests.unit.conftest.MockedK8sHelper):
    def test_schedule_workflow_with_local_source(
        self,
        db: sqlalchemy.orm.Session,
        k8s_secrets_mock,
    ):
        project = mlrun.common.schemas.ProjectOut(
            metadata=mlrun.common.schemas.ProjectMetadata(name="project-name"),
            spec=mlrun.common.schemas.ProjectSpecOut(),
        )

        services.api.crud.Projects().create_project(db, project)

        run_name = "run-name"
        runner = services.api.crud.WorkflowRunners().create_runner(
            run_name=run_name,
            project=project.metadata.name,
            db_session=db,
            auth_info=mlrun.common.schemas.AuthInfo(),
            image="mlrun/mlrun",
        )

        with unittest.mock.patch(
            "services.api.utils.singletons.scheduler.get_scheduler"
        ):
            services.api.crud.WorkflowRunners().schedule(
                runner=runner,
                project=project,
                workflow_request=mlrun.common.schemas.WorkflowRequest(
                    spec=mlrun.common.schemas.WorkflowSpec(
                        name=run_name,
                        engine="remote",
                        image="mlrun/mlrun",
                    ),
                    source="/home/mlrun/project-name/",
                    artifact_path="/home/mlrun/artifacts",
                    notifications=[
                        mlrun.common.schemas.Notification(
                            name="notification-name",
                            kind="slack",
                            secret_params={"webhook": "http://slack.com/webhook"},
                        )
                    ],
                ),
                auth_info=mlrun.common.schemas.AuthInfo(username="test-user"),
            )
            assert list(k8s_secrets_mock.project_secrets_map["project-name"].keys())[
                0
            ].startswith("mlrun.notifications.")

    @pytest.mark.parametrize(
        "run_setup_kwargs, expected_run_setup",
        [
            # default: setup is skipped on the runner pod (DB is the source of truth)
            ({}, False),
            ({"run_setup": False}, False),
            # users can opt-in to running the setup script on the runner pod
            ({"run_setup": True}, True),
        ],
    )
    def test_run_workflow_run_setup_flag(
        self,
        db: sqlalchemy.orm.Session,
        k8s_secrets_mock,
        run_setup_kwargs: dict,
        expected_run_setup: bool,
    ):
        project = mlrun.common.schemas.ProjectOut(
            metadata=mlrun.common.schemas.ProjectMetadata(name="project-name"),
            spec=mlrun.common.schemas.ProjectSpecOut(),
        )
        services.api.crud.Projects().create_project(db, project)

        run_name = "run-name"
        runner = services.api.crud.WorkflowRunners().create_runner(
            run_name=run_name,
            project=project.metadata.name,
            db_session=db,
            auth_info=mlrun.common.schemas.AuthInfo(),
            image="mlrun/mlrun",
        )

        run = services.api.crud.WorkflowRunners().run(
            runner=runner,
            project=project,
            workflow_request=mlrun.common.schemas.WorkflowRequest(
                spec=mlrun.common.schemas.WorkflowSpec(
                    name=run_name,
                    engine="remote",
                    image="mlrun/mlrun",
                    **run_setup_kwargs,
                ),
                source="/home/mlrun/project-name/",
                artifact_path="/home/mlrun/artifacts",
            ),
            auth_info=mlrun.common.schemas.AuthInfo(username="test-user"),
        )

        assert run.spec.parameters["run_setup"] == expected_run_setup

    @pytest.mark.parametrize(
        "source_code_target_dir",
        [
            "/home/mlrun_code",
            None,
        ],
    )
    @pytest.mark.parametrize(
        "source",
        [
            "/home/mlrun/project-name/",
            "./project-name",
            "git://github.com/mlrun/project-name.git",
        ],
    )
    def test_run_workflow_with_local_source(
        self,
        db: sqlalchemy.orm.Session,
        k8s_secrets_mock,
        source_code_target_dir: str,
        source: str,
    ):
        project = mlrun.common.schemas.ProjectOut(
            metadata=mlrun.common.schemas.ProjectMetadata(name="project-name"),
            spec=mlrun.common.schemas.ProjectSpecOut(),
        )
        if source_code_target_dir:
            project.spec.build = {"source_code_target_dir": source_code_target_dir}

        services.api.crud.Projects().create_project(db, project)

        run_name = "run-name"
        runner = services.api.crud.WorkflowRunners().create_runner(
            run_name=run_name,
            project=project.metadata.name,
            db_session=db,
            auth_info=mlrun.common.schemas.AuthInfo(),
            image="mlrun/mlrun",
        )

        run = services.api.crud.WorkflowRunners().run(
            runner=runner,
            project=project,
            workflow_request=mlrun.common.schemas.WorkflowRequest(
                spec=mlrun.common.schemas.WorkflowSpec(
                    name=run_name,
                    engine="remote",
                    image="mlrun/mlrun",
                ),
                source=source,
                artifact_path="/home/mlrun/artifacts",
                notifications=[
                    mlrun.common.schemas.Notification(
                        name="notification-name",
                        kind="slack",
                        secret_params={"webhook": "http://slack.com/webhook"},
                    )
                ],
            ),
            auth_info=mlrun.common.schemas.AuthInfo(username="test-user"),
        )

        assert run.metadata.name == run_name
        assert run.metadata.project == project.metadata.name
        if "://" in source:
            assert run.spec.parameters["url"] == source
            assert "project_context" not in run.spec.parameters
        else:
            if source_code_target_dir and source.startswith("."):
                expected_project_context = os.path.normpath(
                    os.path.join(source_code_target_dir, source)
                )
                assert (
                    run.spec.parameters["project_context"] == expected_project_context
                )
            else:
                assert run.spec.parameters["project_context"] == source
            assert "url" not in run.spec.parameters
        assert (
            run.spec.notifications[0]
            .secret_params.get("secret", "")
            .startswith("mlrun.notifications.")
        )
        assert run.spec.handler == "mlrun.projects.load_and_run"
        assert run.metadata.labels.get("owner") == "test-user"

    @pytest.mark.parametrize(
        "runner_class, source, expected_save",
        [
            (services.api.crud.WorkflowRunners, "./project-name", False),
            (services.api.crud.WorkflowRunners, "", True),
            (services.api.crud.LoadRunner, "s3://project-name", True),
            (services.api.crud.LoadRunner, "", True),
        ],
    )
    def test_run_workflow_save_project(
        self,
        db: sqlalchemy.orm.Session,
        k8s_secrets_mock,
        runner_class,
        source: str,
        expected_save: bool,
    ):
        project = mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name="project-name"),
            spec=mlrun.common.schemas.ProjectSpec(
                source="s3://some-source", artifact_path="/home/mlrun/artifacts"
            ),
        )
        services.api.crud.Projects().create_project(db, project)

        run_name = "run-name"
        runner = runner_class().create_runner(
            run_name=run_name,
            project=project.metadata.name,
            db_session=db,
            auth_info=mlrun.common.schemas.AuthInfo(),
            image="mlrun/mlrun",
        )
        params = dict(
            runner=runner, project=project, auth_info=mlrun.common.schemas.AuthInfo()
        )
        if runner_class == services.api.crud.WorkflowRunners:
            params.update(
                dict(
                    workflow_request=mlrun.common.schemas.WorkflowRequest(
                        spec=mlrun.common.schemas.WorkflowSpec(
                            name=run_name,
                            engine="remote",
                            image="mlrun/mlrun",
                        ),
                        source=source,
                        artifact_path="/home/mlrun/artifacts",
                    ),
                )
            )
        run = runner_class().run(**params)
        assert run.spec.parameters["save"] == expected_save

    def test_schedule_workflow_with_custom_source(
        self,
        db: sqlalchemy.orm.Session,
        k8s_secrets_mock,
    ):
        # This test verifies that if a custom source is provided in the workflow request,
        # it is used for the scheduled workflow instead of the project's source.

        project = mlrun.common.schemas.ProjectOut(
            metadata=mlrun.common.schemas.ProjectMetadata(name="project-name"),
            spec=mlrun.common.schemas.ProjectSpecOut(
                source="git://github.com/mlrun/project-demo.git"
            ),
        )
        services.api.crud.Projects().create_project(db, project)

        run_name = "scheduled-run"
        runner = services.api.crud.WorkflowRunners().create_runner(
            run_name=run_name,
            project=project.metadata.name,
            db_session=db,
            auth_info=mlrun.common.schemas.AuthInfo(),
            image="mlrun/mlrun",
        )

        custom_source = "v3io:///users/testuser/custom_project/project.zip"

        with unittest.mock.patch(
            "services.api.utils.singletons.scheduler.get_scheduler"
        ) as scheduler_mock:
            services.api.crud.WorkflowRunners().schedule(
                runner=runner,
                project=project,
                workflow_request=mlrun.common.schemas.WorkflowRequest(
                    spec=mlrun.common.schemas.WorkflowSpec(
                        name=run_name,
                        engine="remote",
                        image="mlrun/mlrun",
                    ),
                    source=custom_source,
                ),
                auth_info=mlrun.common.schemas.AuthInfo(),
            )

            scheduler_mock.return_value.store_schedule.assert_called_once()
            scheduled_call_kwargs = (
                scheduler_mock.return_value.store_schedule.call_args.kwargs
            )

            assert (
                scheduled_call_kwargs["scheduled_object"]["task"]["spec"]["parameters"][
                    "url"
                ]
                == custom_source
            )
