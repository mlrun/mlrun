import mlrun
import tests.system.base


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestKubejobRuntime(tests.system.base.TestMLRunSystem):

    project_name = "kubejob-system-test"

    def test_deploy_function(self):
        code_path = str(self.assets_path / "kubejob_function.py")

        function = mlrun.code_to_function(
            name="simple-function",
            kind="job",
            project=self.project_name,
            filename=code_path,
        )
        function.build_config(base_image="mlrun/mlrun", commands=["pip install pandas"])

        self._logger.debug("Deploying kubejob function")
        function.deploy()

    def test_function_with_param(self):
        code_path = str(self.assets_path / "function_with_params.py")

        proj = mlrun.get_or_create_project(self.project_name, self.results_path)
        project_param = "some value"
        local_param = "my local param"
        proj.spec.params = {"project_param": project_param}
        proj.save()

        function = mlrun.code_to_function(
            name="function-with-params",
            kind="job",
            handler="handler",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
        )
        run = function.run(params={"param1": local_param})
        assert run.status.results["project_param"] == project_param
        assert run.status.results["param1"] == local_param
