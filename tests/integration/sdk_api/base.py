# Copyright 2018 Iguazio
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
import mlrun.api.schemas
import tests.conftest
from mlrun.db.httpdb import HTTPRunDB
from mlrun.utils import create_logger, retry_until_successful

logger = create_logger(level="debug", name="test-integration")


class TestMLRunIntegration:

    project_name = "system-test-project"
    root_path = pathlib.Path(__file__).absolute().parent.parent.parent.parent
    results_path = root_path / "tests" / "test_results" / "integration"

    db_liveness_timeout = 30
    db_host_internal = "host.docker.internal"
    db_host_external = "localhost"
    db_user = "root"
    db_port = 3306
    db_name = "mlrun"
    db_dsn = f"mysql+pymysql://{db_user}@{db_host_internal}:{db_port}/{db_name}"

    def setup_method(self, method):
        self._logger = logger
        self._logger.info(
            f"Setting up test {self.__class__.__name__}::{method.__name__}"
        )
        self._run_db()
        api_url = self._run_api()
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
        self._remove_db()

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

    @property
    def base_url(self):
        return mlrun.config.config.dbpath + "/api/v1/"

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

    def _run_db(self):
        self._logger.debug("Starting DataBase")
        self._run_command(
            "make",
            args=["run-test-db"],
            cwd=TestMLRunIntegration.root_path,
        )
        output = self._run_command(
            "docker",
            args=["ps", "--last", "1", "-q"],
        )
        self.db_container_id = output.strip()

        self._logger.debug("Started DataBase", container_id=self.db_container_id)

        self._ensure_database_liveness(timeout=self.db_liveness_timeout)

    def _run_api(self):
        self._logger.debug("Starting API")
        self._run_command(
            "make",
            args=["run-api"],
            env=self._extend_current_env(
                {"MLRUN_VERSION": "test-integration", "MLRUN_HTTPDB__DSN": self.db_dsn}
            ),
            cwd=TestMLRunIntegration.root_path,
        )
        output = self._run_command(
            "docker",
            args=["ps", "--last", "1", "-q"],
        )
        self.api_container_id = output.strip()
        # retrieve container bind port + host
        output = self._run_command(
            "docker", args=["port", self.api_container_id, "8080"]
        )
        # usually the output is something like '0.0.0.0:49154\n' but sometimes (in GH actions) it's something like
        # '0.0.0.0:49154\n:::49154\n' for some reason, so just taking the first line
        host = output.splitlines()[0]
        url = f"http://{host}"
        self._check_api_is_healthy(url)
        self._logger.info(
            "Successfully started API", url=url, container_id=self.api_container_id
        )
        return url

    def _remove_api(self):
        if self.api_container_id:
            logs = self._run_command("docker", args=["logs", self.api_container_id])
            self._logger.debug(
                "Removing API container", container_id=self.api_container_id, logs=logs
            )
            self._run_command("docker", args=["rm", "--force", self.api_container_id])

    def _remove_db(self):
        if self.db_container_id:
            logs = self._run_command("docker", args=["logs", self.db_container_id])
            self._logger.debug(
                "Removing Database container",
                container_name=self.db_container_id,
                logs=logs,
            )
            out = self._run_command(
                "docker", args=["rm", "--force", self.db_container_id]
            )
            self._logger.debug(
                "Removed Database container",
                out=out,
            )

    def _ensure_database_liveness(self, retry_interval=2, timeout=30):
        self._logger.debug("Ensuring database liveness")
        retry_until_successful(
            retry_interval,
            timeout,
            self._logger,
            True,
            pymysql.connect,
            host=self.db_host_external,
            user=self.db_user,
            port=self.db_port,
            database=self.db_name,
        )
        self._logger.debug("Database ready for connection")

    @staticmethod
    def _extend_current_env(env):
        current_env = copy.deepcopy(os.environ)
        current_env.update(env)
        return current_env

    @staticmethod
    def _check_api_is_healthy(url):
        health_url = f"{url}/{HTTPRunDB.get_api_path_prefix()}/healthz"
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
