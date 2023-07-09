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
import os
import pathlib
import shutil
import sys

import pytest

import mlrun
from tests.conftest import out_path

project_dir = f"{out_path}/project_dir"


class TestNewProject:
    def setup_method(self, method):
        self.assets_path = (
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent
            / "assets"
        )

    def test_yaml_template(self):
        project = mlrun.new_project(
            "newproj",
            "./",
            from_template=str(self.assets_path / "project.yaml"),
            save=False,
        )
        assert project.spec.description == "test", "failed to load yaml template"

    def test_zip_template(self):
        shutil.rmtree(project_dir, ignore_errors=True)
        project = mlrun.new_project(
            "newproj2",
            project_dir,
            from_template=str(self.assets_path / "project.zip"),
            save=False,
        )
        assert project.spec.description == "test", "failed to load yaml template"

        filepath = os.path.join(project_dir, "prep_data.py")
        assert os.path.isfile(filepath), "file not copied"

    @pytest.mark.skipif(os.name == "nt", reason="Does not work on Windows")
    def test_git_template(self):
        shutil.rmtree(project_dir, ignore_errors=True)
        project = mlrun.new_project(
            "newproj3",
            project_dir,
            from_template="git://github.com/mlrun/project-demo.git",
            save=False,
        )
        assert project.spec.description == "test", "failed to load yaml template"

        filepath = os.path.join(project_dir, "prep_data.py")
        assert os.path.isfile(filepath), "file not copied"
