# Copyright 2018 Iguazio
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
            project_name, f"{self.project_dir}/{project_name}", save=False
        )
        self.project.set_artifact("data", target_path=self.data_url)
        self.project.spec.params = {"label_column": "label"}
        self.project.spec.artifact_path = "/tmp"
        return self.project
