import pathlib
import sys

import mlrun
from tests.conftest import out_path

project_dir = f"{out_path}/project_dir"
data_url = "https://s3.wasabisys.com/iguazio/data/iris/iris.data.raw.csv"


class TestPipeline:
    def setup_method(self, method):
        self.assets_path = (
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent
            / "assets"
        )

    def _create_project(
        self,
        project_name,
    ):
        proj = mlrun.new_project(project_name, f"{project_dir}/{project_name}")
        proj.set_artifact("data", target_path=data_url)
        proj.spec.params = {"label_column": "label"}
        return proj
