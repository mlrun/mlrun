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
import pathlib
from contextlib import nullcontext as does_not_raise

import pytest

import mlrun
import mlrun.artifacts
import tests.conftest
import tests.projects.base_pipeline


class TestLocalPipeline(tests.projects.base_pipeline.TestPipeline):
    pipeline_path = "localpipe.py"

    def _set_functions(self):
        self.project.set_function(
            str(f"{self.assets_path / self.pipeline_path}"),
            "tstfunc",
            image="mlrun/mlrun",
        )

    def test_set_artifact(self, rundb_mock):
        self.project = mlrun.new_project("test-sa", save=False)
        self.project.set_artifact(
            "data1", mlrun.artifacts.Artifact(target_path=self.data_url)
        )
        self.project.set_artifact(
            "data2", target_path=self.data_url, tag="x"
        )  # test the short form
        self.project.register_artifacts()

        for artifact in self.project.spec.artifacts:
            assert artifact["metadata"]["key"] in ["data1", "data2"]
            assert artifact["spec"]["target_path"] == self.data_url

        artifacts = self.project.list_artifacts(tag="x")
        assert len(artifacts) == 1

    def test_import_artifacts(self, rundb_mock):
        results_path = str(pathlib.Path(tests.conftest.results) / "project")
        project = mlrun.new_project(
            "test-sa2", context=str(self.assets_path), save=False
        )
        project.spec.artifact_path = results_path
        # use inline body (in the yaml)
        project.set_artifact("y", "artifact.yaml")
        # use body from the project context dir
        project.set_artifact("z", mlrun.artifacts.Artifact(src_path="body.txt"))
        project.register_artifacts()

        artifacts = project.list_artifacts().to_objects()
        assert len(artifacts) == 2

        expected_body_map = {"y": "123", "z": b"ABC"}
        for artifact in artifacts:
            assert artifact.metadata.key in expected_body_map
            assert expected_body_map[artifact.metadata.key] == artifact._get_file_body()

            db_key = artifact.db_key or artifact.metadata.key
            some_artifact = project.get_artifact(db_key)
            assert some_artifact.metadata.key == artifact.metadata.key
            assert (
                some_artifact._get_file_body()
                == expected_body_map[artifact.metadata.key]
            )

    def test_run_alone(self):
        mlrun.projects.pipeline_context.clear(with_project=True)
        function = mlrun.code_to_function(
            "test1",
            filename=str(f"{self.assets_path / self.pipeline_path}"),
            handler="func1",
            kind="job",
        )
        run_result = mlrun.run_function(function, params={"p1": 5}, local=True)
        print(run_result.to_yaml())
        assert run_result.state() == "completed", "run didnt complete"
        # expect y = param1 * 2 = 10
        assert run_result.output("accuracy") == 10, "unexpected run result"

    def test_run_project(self):
        mlrun.projects.pipeline_context.clear(with_project=True)
        self._create_project("localpipe1")
        self._set_functions()
        run1 = mlrun.run_function(
            "tstfunc", handler="func1", params={"p1": 3}, local=True
        )
        run2 = mlrun.run_function(
            "tstfunc",
            handler="func2",
            params={"x": run1.outputs["accuracy"]},
            local=True,
        )
        assert run2.state() == "completed", "run didnt complete"
        # expect y = (param1 * 2) + 1 = 7
        assert run2.output("y") == 7, "unexpected run result"

    @pytest.mark.parametrize(
        "local,expectation",
        [
            # If local is not specified, and kfp isn't configured, the pipeline will run locally anyway
            (None, does_not_raise()),
            # If the pipeline is local, the pipeline should run successfully
            (True, does_not_raise()),
            # If the pipeline is explicitly remote, the pipeline should fail because kfp isn't configured
            (False, pytest.raises(ValueError)),
        ],
    )
    def test_run_pipeline(self, local, expectation):
        mlrun.projects.pipeline_context.clear(with_project=True)
        self._create_project("localpipe2")
        self._set_functions()

        with expectation:
            self.project.run(
                "p1",
                workflow_path=str(f"{self.assets_path / self.pipeline_path}"),
                workflow_handler="my_pipe",
                arguments={"param1": 7},
                local=local,
            )

            run_result: mlrun.RunObject = mlrun.projects.pipeline_context._test_result
            assert run_result.state() == "completed", "run didnt complete"
            # expect y = (param1 * 2) + 1 = 15
            assert run_result.output("y") == 15, "unexpected run result"

    def test_run_pipeline_no_workflow(self):
        mlrun.projects.pipeline_context.clear(with_project=True)
        self._create_project("localpipe2")
        self._set_functions()

        with pytest.raises(ValueError):
            self.project.run(
                local=True,
            )

    def test_pipeline_args(self):
        mlrun.projects.pipeline_context.clear(with_project=True)
        self._create_project("localpipe3")
        self._set_functions()
        args = [
            mlrun.model.EntrypointParam("param1", type="int", doc="p1", required=True),
            mlrun.model.EntrypointParam("param2", type="str", default="abc", doc="p2"),
        ]
        self.project.set_workflow(
            "main",
            str(f"{self.assets_path / self.pipeline_path}"),
            handler="args_pipe",
            args_schema=args,
        )
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            # expect an exception when param1 (required) arg is not specified
            self.project.run("main", local=True)

        self.project.run("main", local=True, arguments={"param1": 6})
        run_result: mlrun.RunObject = mlrun.projects.pipeline_context._test_result
        print(run_result.to_yaml())
        # expect p1 = param1, p2 = default for param2 (abc)
        assert (
            run_result.output("p1") == 6 and run_result.output("p2") == "abc"
        ), "wrong arg values"

        self.project.run("main", local=True, arguments={"param1": 6, "param2": "xy"})
        run_result: mlrun.RunObject = mlrun.projects.pipeline_context._test_result
        print(run_result.to_yaml())
        # expect p1=param1, p2=xy
        assert (
            run_result.output("p1") == 6 and run_result.output("p2") == "xy"
        ), "wrong arg values"

    def test_run_pipeline_artifact_path(self):
        mlrun.projects.pipeline_context.clear(with_project=True)
        self._create_project("localpipe2")
        self._set_functions()
        generic_path = "/path/without/workflow/id"
        self.project.spec.artifact_path = generic_path

        self.project.run(
            "p4",
            workflow_path=str(f"{self.assets_path / self.pipeline_path}"),
            workflow_handler="my_pipe",
            arguments={"param1": 7},
            local=True,
            artifact_path=generic_path,
        )

        # When user provided a path, it will be used as-is
        assert mlrun.projects.pipeline_context._artifact_path == generic_path

        mlrun.projects.pipeline_context.clear(with_project=True)
        run_status = self.project.run(
            "p4",
            workflow_path=str(f"{self.assets_path / self.pipeline_path}"),
            workflow_handler="my_pipe",
            arguments={"param1": 7},
            local=True,
        )

        # Otherwise, the artifact_path should automatically have the run id injected in it.
        assert (
            mlrun.projects.pipeline_context._artifact_path
            == f"{generic_path}/{run_status.run_id}"
        )

    def test_run_pipeline_with_ttl(self):
        mlrun.projects.pipeline_context.clear(with_project=True)
        self._create_project("localpipettl")
        self._set_functions()
        workflow_path = str(f"{self.assets_path / self.pipeline_path}")
        cleanup_ttl = 1234
        run = self.project.run(
            "p4",
            workflow_path=workflow_path,
            workflow_handler="my_pipe",
            arguments={"param1": 7},
            local=True,
            cleanup_ttl=cleanup_ttl,
        )
        assert run.workflow.cleanup_ttl == cleanup_ttl

        self.project.set_workflow("my-workflow", workflow_path=workflow_path)

        run = self.project.run(
            "my-workflow",
            workflow_handler="my_pipe",
            arguments={"param1": 7},
            local=True,
            cleanup_ttl=cleanup_ttl,
        )
        assert run.workflow.cleanup_ttl == cleanup_ttl
