import pathlib

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
            # kind="job"
        )

    def test_set_artifact(self):
        self.project = mlrun.new_project("test-sa", skip_save=True)
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

    def test_import_artifacts(self):
        results_path = str(pathlib.Path(tests.conftest.results) / "project")
        project = mlrun.new_project("test-sa2", context=str(self.assets_path))
        project.spec.artifact_path = results_path
        # use inline body (in the yaml)
        project.set_artifact("y", "artifact.yaml")
        # use body from the project context dir
        project.set_artifact("z", mlrun.artifacts.Artifact(src_path="body.txt"))
        project.register_artifacts()

        artifacts = project.list_artifacts().objects()
        assert len(artifacts) == 2
        assert artifacts[0].metadata.key == "y"
        assert artifacts[0]._get_file_body() == "123"

        z_artifact = project.get_artifact("z")
        assert z_artifact.metadata.key == "z"
        assert z_artifact._get_file_body() == b"ABC"

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

    def test_run_pipeline(self):
        mlrun.projects.pipeline_context.clear(with_project=True)
        self._create_project("localpipe2")
        self._set_functions()
        self.project.run(
            "p1",
            workflow_path=str(f"{self.assets_path / self.pipeline_path}"),
            workflow_handler="my_pipe",
            arguments={"param1": 7},
            local=True,
        )

        run_result: mlrun.RunObject = mlrun.projects.pipeline_context._test_result
        assert run_result.state() == "completed", "run didnt complete"
        # expect y = (param1 * 2) + 1 = 15
        assert run_result.output("y") == 15, "unexpected run result"

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
