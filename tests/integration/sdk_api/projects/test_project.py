import pathlib
import unittest

import pytest

import mlrun
import tests.conftest
import tests.integration.sdk_api.base


class TestProject(tests.integration.sdk_api.base.TestMLRunIntegration):
    def test_create_project(self):
        project_name = "some-project"
        project = mlrun.new_project(project_name)
        project.save_to_db()
        projects = mlrun.get_run_db().list_projects()
        assert len(projects) == 1
        assert projects[0].metadata.name == project_name

    def test_load_project_from_db(self):
        project_name = "some-project"
        project = mlrun.new_project(project_name)
        project.save_to_db()
        mlrun.load_project(".", f"db://{project_name}")

    def test_load_project_with_save(self):
        project_name = "some-project"
        project = mlrun.new_project(project_name)
        project_file_path = pathlib.Path(tests.conftest.results) / "project.yaml"
        project.export(str(project_file_path))

        imported_project_name = "imported-project"
        # loaded project but didn't saved
        mlrun.load_project(
            "./", str(project_file_path), name=imported_project_name, save=False
        )

        # loading project from db, but earlier load didn't saved, expected to fail
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            mlrun.load_project(".", f"db://{imported_project_name}", save=False)

        # loading project and saving
        expected_project = mlrun.load_project(
            "./", str(project_file_path), name=imported_project_name
        )

        # loading project from db, expected to succeed
        load_project_from_db = mlrun.load_project(
            ".", f"db://{imported_project_name}", save=False
        )
        print(expected_project.to_dict())
        print(load_project_from_db.to_dict())
        assert expected_project.to_dict() == load_project_from_db.to_dict()
