import os

import pytest
import v3io
from v3io.dataplane import RaiseForStatus

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


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestNuclioRuntimeWithStream(tests.system.base.TestMLRunSystem):
    project_name = "nuclio-stream-project"
    stream_container = "bigdata"
    stream_path = "/test_nuclio/test_serving_with_child_function/"

    def custom_teardown(self):
        v3io_client = v3io.dataplane.Client(
            endpoint=os.environ["V3IO_API"], access_key=os.environ["V3IO_ACCESS_KEY"]
        )
        v3io_client.delete_stream(
            self.stream_container,
            self.stream_path,
            raise_for_status=RaiseForStatus.never,
        )

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
            ">>", "q1", path=f"v3io:///{self.stream_container}{self.stream_path}"
        ).to(name="child", class_name="Identity", function="child")

        function.add_child_function("child", child_code_path, "mlrun/mlrun")

        self._logger.debug("Deploying nuclio function")
        function.deploy()


@tests.system.base.TestMLRunSystem.skip_test_if_env_not_configured
class TestNuclioMLRunJobs(tests.system.base.TestMLRunSystem):
    project_name = "nuclio-mlrun-jobs"

    def _deploy_function(self, replicas=1):
        filename = str(self.assets_path / "handler.py")
        fn = mlrun.code_to_function(
            filename=filename,
            name="nuclio-mlrun",
            kind="nuclio:mlrun",
            image="mlrun/mlrun",
            handler="my_func",
        )
        # replicas * workers need to match or exceed parallel_runs
        fn.spec.replicas = replicas
        fn.with_http(workers=2)
        fn.deploy()
        return fn

    def test_single_run(self):
        fn = self._deploy_function()
        run_result = fn.run(params={"p1": 8})

        print(run_result.to_yaml())
        assert run_result.state() == "completed", "wrong state"
        # accuracy = p1 * 2
        assert run_result.output("accuracy") == 16, "unexpected results"

    def test_hyper_run(self):
        fn = self._deploy_function(2)

        hyper_param_options = mlrun.model.HyperParamOptions(
            parallel_runs=4, selector="max.accuracy", max_errors=1,
        )

        p1 = [4, 2, 5, 8, 9, 6, 1, 11, 1, 1, 2, 1, 1]
        run_result = fn.run(
            params={"p2": "xx"},
            hyperparams={"p1": p1},
            hyper_param_options=hyper_param_options,
        )
        print(run_result.to_yaml())
        assert run_result.state() == "completed", "wrong state"
        # accuracy = max(p1) * 2
        assert run_result.output("accuracy") == 22, "unexpected results"

        # test early stop
        hyper_param_options = mlrun.model.HyperParamOptions(
            parallel_runs=1,
            selector="max.accuracy",
            max_errors=1,
            stop_condition="accuracy>9",
        )

        run_result = fn.run(
            params={"p2": "xx"},
            hyperparams={"p1": p1},
            hyper_param_options=hyper_param_options,
        )
        print(run_result.to_yaml())
        assert run_result.state() == "completed", "wrong state"
        # accuracy = max(p1) * 2, stop where accuracy > 9
        assert run_result.output("accuracy") == 10, "unexpected results"
