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
