import pytest

import mlrun
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

    def test_load_project(self):
        project_name = "some-project"
        project = mlrun.new_project(project_name)
        project.save_to_db()

        loaded_project_name = "loaded-project"
        mlrun.load_project(
            ".", f"db://{project_name}", name=loaded_project_name, save=False
        )
        # loaded project but didn't saved expected to fail
        with pytest.raises(mlrun.errors.MLRunNotFoundError):
            mlrun.load_project(".", f"db://{loaded_project_name}", save=False)

        # loaded project and auto saved
        loaded_project = mlrun.load_project(".", f"db://{project_name}", name=loaded_project_name)
        print(loaded_project.name, loaded_project, loaded_project)
        # load project expects to succeed because already saved in DB
        mlrun.load_project("./", f"db://{loaded_project_name}")

    # def test_get_or_create_project(self):
    #     project_name = "some-project"
    #     with pytest.raises(mlrun.errors.MLRunNotFoundError):
    #         project = mlrun.get_or_create_project(project_name, "./",save=False)
    #
    #     project = mlrun.new_project(project_name, save=False)
    #     project.set_function("hub://describe", "describe")
    #     project_function_object = project.spec._function_objects
    #     project_file_path = pathlib.Path(tests.conftest.results) / "project.yaml"
    #     project.export(str(project_file_path))
    #     imported_project = mlrun.load_project("./", str(project_file_path))
    #
    #     project = mlrun.get_or_create_project(project_name, "./", save=False)
    #
