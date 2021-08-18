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
            "newproj", "./", from_template=str(self.assets_path / "project.yaml")
        )
        assert project.spec.description == "test", "failed to load yaml template"

    def test_zip_template(self):
        shutil.rmtree(project_dir, ignore_errors=True)
        project = mlrun.new_project(
            "newproj2", project_dir, from_template=str(self.assets_path / "project.zip")
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
        )
        assert project.spec.description == "test", "failed to load yaml template"

        filepath = os.path.join(project_dir, "prep_data.py")
        assert os.path.isfile(filepath), "file not copied"
