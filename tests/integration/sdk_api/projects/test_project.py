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
