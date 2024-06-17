# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import copy
import os
import pathlib
import subprocess
import sys

import pymysql

import mlrun
import mlrun.common.schemas
import tests.conftest
from mlrun.db.httpdb import HTTPRunDB
from mlrun.utils import FormatterKinds, create_test_logger, retry_until_successful

logger = create_test_logger(name="test-integration")


class TestMLRunIntegration:
    project_name = "system-test-project"
    root_path = pathlib.Path(__file__).absolute().parent.parent.parent.parent
    results_path = root_path / "tests" / "test_results" / "integration"
    extra_env = None
    db_container_name = "test-db"
    api_container_name = "mlrun-api"
    db_liveness_timeout = 40
    db_host_internal = "host.docker.internal"
    db_host_external = "localhost"
    db_user = "root"
    db_port = 3306
    db_name = "mlrun"
    db_dsn = f"mysql+pymysql://{db_user}@{db_host_internal}:{db_port}/{db_name}"

    @classmethod
    def setup_class(cls):
        cls._logger = logger
        cls._logger.info(f"Setting up class {cls.__name__}")
        cls._run_db()
        cls._run_api(cls.extra_env)

    def setup_method(self, method):
        self._logger.info(
            f"Setting up test {self.__class__.__name__}::{method.__name__}"
        )
        self._test_env = {}
        self._old_env = {}
        api_url = self._start_api()
        self._setup_env({"MLRUN_DBPATH": api_url})
        self.custom_setup()
        self._logger.info(
            f"Finished setting up test {self.__class__.__name__}::{method.__name__}"
        )

    @classmethod
    def teardown_class(cls):
        cls._logger.info(f"Tearing down class {cls.__class__.__name__}")
        cls._log_container_logs(cls.db_container_name)
        cls._remove_container(cls.db_container_name)
        cls._log_container_logs(cls.api_container_name)
        cls._remove_container(cls.api_container_name)

    def teardown_method(self, method):
        self._logger.info(
            f"Tearing down test {self.__class__.__name__}::{method.__name__}"
        )
        self.custom_teardown()
        self._teardown_env()
        self._stop_api()
        self._reset_db()
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

    @property
    def base_url(self):
        return mlrun.mlconf.dbpath + "/api/"

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
        mlrun.mlconf.reload()

    def _teardown_env(self):
        self._logger.debug("Tearing down test environment")
        for env_var in self._test_env:
            if env_var in os.environ:
                del os.environ[env_var]
        os.environ.update(self._old_env)
        # reload the config so changes to the env vars will take affect
        mlrun.mlconf.reload()

    @classmethod
    def _run_db(cls):
        cls._logger.debug("Starting mlrun database")
        cls._run_command(
            "make",
            args=["run-test-db"],
            cwd=TestMLRunIntegration.root_path,
        )
        cls._logger.debug("Started mlrun database")

        cls._ensure_database_liveness(timeout=cls.db_liveness_timeout)
        return False

    @classmethod
    def _run_api(cls, extra_env=None):
        cls._logger.debug("Starting API")
        cls._run_command(
            # already compiled schemas in run-test-db
            "MLRUN_SKIP_COMPILE_SCHEMAS=true make",
            args=["run-api"],
            env=cls._extend_current_env(
                {
                    "MLRUN_VERSION": "0.0.0+unstable",
                    "MLRUN_HTTPDB__DSN": cls.db_dsn,
                    "MLRUN_LOG_LEVEL": "DEBUG",
                    "MLRUN_LOG_FORMATTER": FormatterKinds.HUMAN_EXTENDED.value,
                    "MLRUN_SECRET_STORES__TEST_MODE_MOCK_SECRETS": "True",
                },
                extra_env,
            ),
            cwd=TestMLRunIntegration.root_path,
        )

    def _stop_api(self):
        self._run_command("docker", args=["kill", self.api_container_name])

    def _start_api(self):
        running = False
        try:
            output = self._run_command(
                "docker",
                args=["inspect", "-f", "{{.State.Running}}", self.api_container_name],
            )
            running = output.strip().lower() == "true"
        except Exception as exc:
            self._logger.debug(
                "Failed to check if API container is running",
                exc=exc,
            )
        if not running:
            self._run_command("docker", args=["start", self.api_container_name])
        api_url = self._resolve_mlrun_api_url()
        self._check_api_is_healthy(api_url)
        self._logger.info("Successfully started API", api_url=api_url)
        return api_url

    def _reset_db(self):
        self._logger.debug(
            "Recreating MLRun database",
        )
        self._run_command(
            "docker",
            args=[
                "exec",
                self.db_container_name,
                "mysql",
                "-u",
                "root",
                "-e",
                "'DROP DATABASE IF EXISTS mlrun'",
            ],
        )
        self._run_command(
            "docker",
            args=[
                "exec",
                self.db_container_name,
                "mysql",
                "-u",
                "root",
                "-e",
                "'CREATE DATABASE mlrun'",
            ],
        )
        self._logger.debug(
            "Recreated MLRun database",
        )

    @classmethod
    def _remove_container(cls, container_id):
        cls._logger.debug(
            "Removing container",
            container_id=container_id,
        )
        cls._run_command("docker", args=["rm", "--force", container_id])
        cls._logger.debug(
            "Removed container",
            container_id=container_id,
        )

    @classmethod
    def _log_container_logs(cls, container_id):
        logs = cls._run_command("docker", args=["logs", container_id])
        logs = logs.replace("\n", "\n\t")
        cls._logger.debug(
            f"Retrieved container logs:\n {logs}",
            container_name=container_id,
        )

    @classmethod
    def _ensure_database_liveness(cls, retry_interval=2, timeout=30):
        cls._logger.debug("Ensuring database liveness")
        retry_until_successful(
            retry_interval,
            timeout,
            cls._logger,
            True,
            pymysql.connect,
            host=cls.db_host_external,
            user=cls.db_user,
            port=cls.db_port,
            database=cls.db_name,
        )
        cls._logger.debug("Database ready for connection")

    @staticmethod
    def _extend_current_env(env, extra_env=None):
        current_env = copy.deepcopy(os.environ)
        current_env.update(env)
        if extra_env:
            current_env.update(extra_env)
        return current_env

    @staticmethod
    def _check_api_is_healthy(url):
        health_url = f"{url}/{HTTPRunDB.get_api_path_prefix()}/healthz"
        timeout = 90
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

    def _resolve_mlrun_api_url(self):
        # retrieve container bind port + host
        output = self._run_command(
            "docker", args=["port", self.api_container_name, "8080"]
        )
        # usually the output is something like '0.0.0.0:49154\n' but sometimes (in GH actions) it's something like
        # '0.0.0.0:49154\n:::49154\n' for some reason, so just taking the first line
        host = output.splitlines()[0]
        url = f"http://{host}"
        return url
