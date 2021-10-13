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

    def test_deploy_function_with_error_handler(self):
        code_path = str(self.assets_path / "function-with-catcher.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="function-with-catcher",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
        )

        graph = function.set_topology("flow", engine="async")

        graph.to(name="step1", handler="inc")
        graph.add_step(name="catcher", handler="catcher", full_event=True, after="")

        graph.error_handler("catcher")

        self._logger.debug("Deploying nuclio function")
        function.deploy()

    def test_serving_with_child_function(self):
        code_path = str(self.assets_path / "nuclio_function.py")
        child_code_path = str(self.assets_path / "child_function.py")

        self._logger.debug("Creating nuclio function")
        function = mlrun.code_to_function(
            name="function-with-child",
            kind="serving",
            project=self.project_name,
            filename=code_path,
            image="mlrun/mlrun",
        )

        graph = function.set_topology("flow", engine="async")

        graph.to(
            ">>",
            "q1",
            path="v3io:///bigdata/test_nuclio/test_serving_with_child_function/",
        ).to(name="child", class_name="Identity", function="child")

        function.add_child_function("child", child_code_path, "mlrun/mlrun")

        self._logger.debug("Deploying nuclio function")
        function.deploy()
