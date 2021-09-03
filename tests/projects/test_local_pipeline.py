import pathlib
import sys

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
        proj.set_artifact("data", mlrun.artifacts.Artifact(target_path=data_url))
        proj.spec.params = {"label_column": "label"}
        return proj

    def test_run(self):
        project = self._create_project("localpipe1")
        project.run(
            "p1",
            workflow_path=str(f'{self.assets_path / "localpipe.py"}'),
            workflow_handler="my_pipe",
            arguments={"param1": 7},
            engine="local",
            local=True,
        )
