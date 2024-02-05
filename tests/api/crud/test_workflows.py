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
import os.path

import pytest
import sqlalchemy.orm

import mlrun.common.schemas
import server.api.crud
import tests.api.conftest


class TestWorkflows(tests.api.conftest.MockedK8sHelper):
    @pytest.mark.parametrize(
        "source_target_dir",
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
        source_target_dir: str,
        source: str,
    ):
        project = mlrun.common.schemas.Project(
            metadata=mlrun.common.schemas.ProjectMetadata(name="project-name"),
            spec=mlrun.common.schemas.ProjectSpec(),
        )
        if source_target_dir:
            project.spec.build = mlrun.common.schemas.common.ImageBuilder(
                source_target_dir=source_target_dir
            )

        server.api.crud.Projects().create_project(db, project)

        run_name = "run-name"
        runner = server.api.crud.WorkflowRunners().create_runner(
            run_name=run_name,
            project=project.metadata.name,
            db_session=db,
            auth_info=mlrun.common.schemas.AuthInfo(),
            image="mlrun/mlrun",
        )

        run = server.api.crud.WorkflowRunners().run(
            runner=runner,
            project=project,
            workflow_request=mlrun.common.schemas.WorkflowRequest(
                spec=mlrun.common.schemas.WorkflowSpec(
                    name=run_name,
                    engine="remote",
                    code=None,
                    path=None,
                    args=None,
                    handler=None,
                    ttl=None,
                    args_schema=None,
                    schedule=None,
                    run_local=None,
                    image="mlrun/mlrun",
                ),
                source=source,
                artifact_path="/home/mlrun/artifacts",
            ),
            auth_info=mlrun.common.schemas.AuthInfo(),
        )

        assert run.metadata.name == run_name
        assert run.metadata.project == project.metadata.name
        if "://" in source:
            assert run.spec.parameters["url"] == source
            assert "project_context" not in run.spec.parameters
        else:
            if source_target_dir and source.startswith("."):
                expected_project_context = os.path.normpath(
                    os.path.join(source_target_dir, source)
                )
                assert (
                    run.spec.parameters["project_context"] == expected_project_context
                )
            else:
                assert run.spec.parameters["project_context"] == source
            assert "url" not in run.spec.parameters

        assert run.spec.handler == "mlrun.projects.load_and_run"
