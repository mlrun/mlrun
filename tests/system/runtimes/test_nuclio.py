import pathlib

import mlrun
import tests.system.base


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestNuclioRuntime(tests.system.base.TestMLRunSystem):

    project_name = "does-not-exist-3"

    @staticmethod
    def _skip_set_environment():
        # Skip to make sure project ensured in Nuclio function deployment flow
        return True

    def test_deploy_function_without_project(self):
        code_path = str(self.assets_path / "nuclio_function.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="simple-function",
            kind="nuclio",
            project=self.project_name,
            filename=code_path,
        )

        self._logger.debug("Deploying nuclio function")
        function.deploy()
