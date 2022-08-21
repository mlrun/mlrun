import pathlib

import deepdiff
import pytest

import mlrun
import mlrun.api.schemas
import tests.conftest
import tests.integration.sdk_api.base


class TestProject(tests.integration.sdk_api.base.TestMLRunIntegration):
    def test_create_project(self):
        project_name = "some-project"
        mlrun.new_project(project_name)
        projects = mlrun.get_run_db().list_projects()
        assert len(projects) == 1
        assert projects[0].metadata.name == project_name

    def test_create_project_failure_already_exists(self):
        project_name = "some-project"
        mlrun.new_project(project_name)

        with pytest.raises(mlrun.errors.MLRunConflictError) as exc:
            mlrun.new_project(project_name)
        assert "Conflict - Project already exists: some-project" in str(exc.value)

    def test_override_project(self):
        project_name = "some-project"
        mlrun.new_project(project_name)
        projects = mlrun.get_run_db().list_projects()
        assert len(projects) == 1
        assert projects[0].metadata.name == project_name
        old_created_time = projects[0].metadata.created

        mlrun.new_project(project_name, override=True)
        projects = mlrun.get_run_db().list_projects()
        assert len(projects) == 1
        assert projects[0].metadata.name == project_name

        assert projects[0].metadata.created > old_created_time

    def test_load_project_from_db(self):
        project_name = "some-project"
        mlrun.new_project(project_name)
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
        loaded_project_from_db = mlrun.load_project(
            ".", f"db://{imported_project_name}", save=False
        )
        _assert_projects(expected_project, loaded_project_from_db)


def _assert_projects(expected_project, project):
    assert (
        deepdiff.DeepDiff(
            expected_project.to_dict(),
            project.to_dict(),
            ignore_order=True,
            exclude_paths={
                "root['metadata']['created']",
                "root['spec']['desired_state']",
                "root['status']",
            },
        )
        == {}
    )
    assert expected_project.spec.desired_state == project.spec.desired_state
    assert expected_project.spec.desired_state == project.status.state
