import pathlib
import sys

import mlrun
from tests.conftest import out_path


class TestPipeline:
    project_dir = f"{out_path}/project_dir"
    data_url = "https://s3.wasabisys.com/iguazio/data/iris/iris.data.raw.csv"

    def setup_method(self, method):
        self.assets_path = (
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent
            / "assets"
        )

    def _create_project(
        self,
        project_name,
    ):
        self.project = mlrun.new_project(
            project_name, f"{self.project_dir}/{project_name}", skip_save=True
        )
        self.project.set_artifact("data", target_path=self.data_url)
        self.project.spec.params = {"label_column": "label"}
        return self.project
