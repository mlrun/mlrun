import os
import pathlib
import shutil
import sys

import pytest
from kfp import dsl

import mlrun
from mlrun.projects import run_function
from tests.conftest import out_path

project_dir = f"{out_path}/project_dir"
data_url = "https://s3.wasabisys.com/iguazio/data/iris/iris.data.raw.csv"
os.environ["SLACK_WEBHOOK"] = "https://hooks.slack.com/services/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"


def my_pipe(param1=0):
    run1 = run_function("tstfunc", handler="func1", params={"p1": param1})
    print(run1.to_yaml())

    run2 = run_function("tstfunc", handler="func2", params={"x": run1.outputs["accuracy"]})
    print(run2.to_yaml())



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
            #kind="job"
        )
        proj.set_artifact("data", mlrun.artifacts.Artifact(target_path=data_url))
        proj.spec.params = {"label_column": "label"}
        return proj

    def test_run(self):
        project = self._create_project("localpipe1")
        project.run("p1", workflow_handler=my_pipe, arguments={"param1": 7}, engine="local", local=True)


