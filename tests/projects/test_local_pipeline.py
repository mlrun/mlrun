import pathlib
import sys

import pytest

import mlrun
from tests.conftest import out_path

project_dir = f"{out_path}/project_dir"
data_url = "https://s3.wasabisys.com/iguazio/data/iris/iris.data.raw.csv"


class TestNewProject:
    def setup_method(self, method):
        self.assets_path = (
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent
            / "assets"
        )

    def _create_project(self, project_name):
        proj = mlrun.new_project(project_name, f"{project_dir}/{project_name}")
        proj.set_function(
            str(f'{self.assets_path / "localpipe.py"}'),
            "tstfunc",
            image="mlrun/mlrun",
            # kind="job"
        )
        proj.set_artifact("data", target_path=data_url)
        proj.spec.params = {"label_column": "label"}
        return proj

    def test_set_artifact(self):
        project = mlrun.new_project("test-sa")
        project.set_artifact("data1", mlrun.artifacts.Artifact(target_path=data_url))
        project.set_artifact("data2", target_path=data_url)  # test the short form

        for artifact in project.spec.artifacts:
            assert artifact["key"] in ["data1", "data2"]
            assert artifact["target_path"] == data_url

    def test_run_alone(self):
        mlrun.projects.pipeline_context.clear(with_project=True)
        function = mlrun.code_to_function(
            "test1",
            filename=str(f'{self.assets_path / "localpipe.py"}'),
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
        project = self._create_project("localpipe2")
        project.run(
            "p1",
            workflow_path=str(f'{self.assets_path / "localpipe.py"}'),
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
        project = self._create_project("localpipe3")
        args = [
            mlrun.model.EntrypointParam("param1", type="int", doc="p1", required=True),
            mlrun.model.EntrypointParam("param2", type="str", default="abc", doc="p2"),
        ]
        project.set_workflow(
            "main",
            str(f'{self.assets_path / "localpipe.py"}'),
            handler="args_pipe",
            args_schema=args,
        )
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            # expect an exception when param1 (required) arg is not specified
            project.run("main", local=True)

        project.run("main", local=True, arguments={"param1": 6})
        run_result: mlrun.RunObject = mlrun.projects.pipeline_context._test_result
        print(run_result.to_yaml())
        # expect p1 = param1, p2 = default for param2 (abc)
        assert (
            run_result.output("p1") == 6 and run_result.output("p2") == "abc"
        ), "wrong arg values"

        project.run("main", local=True, arguments={"param1": 6, "param2": "xy"})
        run_result: mlrun.RunObject = mlrun.projects.pipeline_context._test_result
        print(run_result.to_yaml())
        # expect p1=param1, p2=xy
        assert (
            run_result.output("p1") == 6 and run_result.output("p2") == "xy"
        ), "wrong arg values"
