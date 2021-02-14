import copy
import os
import pathlib
import subprocess
import sys

import mlrun
import mlrun.api.schemas
import tests.conftest
from mlrun.utils import create_logger

logger = create_logger(level="debug", name="test-integration")


class TestMLRunIntegration:

    project_name = "system-test-project"
    root_path = pathlib.Path(__file__).absolute().parent.parent.parent.parent
    results_path = root_path / "tests" / "test_results" / "integration"

    def setup_method(self, method):
        self._logger = logger
        self._logger.info(
            f"Setting up test {self.__class__.__name__}::{method.__name__}"
        )
        self.container_id, api_url = self._run_api()
        self._test_env = {}
        self._old_env = {}
        self._setup_env({"MLRUN_DBPATH": api_url})

        self.custom_setup()

        self._logger.info(
            f"Finished setting up test {self.__class__.__name__}::{method.__name__}"
        )

    def teardown_method(self, method):
        self._logger.info(
            f"Tearing down test {self.__class__.__name__}::{method.__name__}"
        )

        self.custom_teardown()

        self._remove_api()

        self._teardown_env()
        self._logger.info(
            f"Finished tearing down test {self.__class__.__name__}::{method.__name__}"
        )

    def custom_setup(self):
        pass

    def custom_teardown(self):
        pass

    @property
    def assets_path(self):
        return (
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent
            / "assets"
        )

    def _setup_env(self, env: dict):
        self._logger.debug("Setting up test environment")
        self._test_env.update(env)

        # save old env vars for returning them on teardown
        for env_var, value in env.items():
            if env_var in os.environ:
                self._old_env[env_var] = os.environ[env_var]

            if value:
                os.environ[env_var] = value
        # reload the config so changes to the env vars will take affect
        mlrun.config.config.reload()

    def _teardown_env(self):
        self._logger.debug("Tearing down test environment")
        for env_var in self._test_env:
            if env_var in os.environ:
                del os.environ[env_var]
        os.environ.update(self._old_env)
        # reload the config so changes to the env vars will take affect
        mlrun.config.config.reload()

    def _run_api(self):
        self._logger.debug("Starting API")
        self._run_command(
            "make",
            args=["run-api"],
            env=self._extend_current_env({"MLRUN_VERSION": "test-integration"}),
        )
        output = self._run_command("docker", args=["ps", "--last", "1", "-q"],)
        container_id = output.strip()
        # retrieve container bind port + host
        output = self._run_command("docker", args=["port", container_id, "8080"])
        host = output.strip()
        url = f"http://{host}"
        self._check_api_is_healthy(url)
        self._logger.info("Successfully started API", url=url)
        return container_id, url

    def _remove_api(self):
        if self.container_id:
            logs = self._run_command("docker", args=["logs", self.container_id])
            self._logger.debug(
                "Removing API container", container_id=self.container_id, logs=logs
            )
            self._run_command("docker", args=["rm", "--force", self.container_id])

    @staticmethod
    def _extend_current_env(env):
        current_env = copy.deepcopy(os.environ)
        current_env.update(env)
        return current_env

    @staticmethod
    def _check_api_is_healthy(url):
        health_url = f"{url}/api/healthz"
        timeout = 30
        if not tests.conftest.wait_for_server(health_url, timeout):
            raise RuntimeError(f"API did not start after {timeout} sec")

    @staticmethod
    def _run_command(command, args=None, cwd=None, env=None):
        if args:
            command += " " + " ".join(args)

        try:
            process = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                encoding="utf-8",
                cwd=cwd,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "Command failed",
                stdout=exc.stdout,
                stderr=exc.stderr,
                return_code=exc.returncode,
                cmd=exc.cmd,
                env=env,
                args=exc.args,
            )
            raise
        output = process.stdout

        return output
