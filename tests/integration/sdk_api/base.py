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

import copy
import os
import pathlib
import subprocess
import sys
import time
import typing
from typing import Optional

import docker
import docker.errors
import pytest
import requests
import sqlalchemy

import mlrun
import mlrun.common.schemas
import mlrun.db.httpdb
import mlrun.utils
import tests.conftest

logger = mlrun.utils.create_test_logger(name="test-integration")


class TestMLRunIntegration:
    project_name = "integration-test-project"
    root_path = pathlib.Path(__file__).absolute().parent.parent.parent.parent
    results_path = root_path / "tests" / "test_results" / "integration"
    api_container_name = "mlrun-api"
    api_url = None
    _test_env = {}
    _old_env = {}

    @classmethod
    def setup_class(cls):
        cls._logger = logger
        cls._logger.info(f"Setting up class {cls.__name__}")

    def setup_method(self, method):
        self._logger.info(
            f"Setting up test {self.__class__.__name__}::{method.__name__}"
        )
        self._test_env = {}
        self._old_env = {}
        self._logger.info(
            f"Finished setting up test {self.__class__.__name__}::{method.__name__}"
        )

    @classmethod
    def teardown_class(cls):
        cls._logger.info(f"Tearing down class {cls.__class__.__name__}")
        cls._log_container_logs(cls.api_container_name)
        cls._remove_container(cls.api_container_name)

    def teardown_method(self, method):
        self._logger.info(
            f"Tearing down test {self.__class__.__name__}::{method.__name__}"
        )
        self.custom_teardown()
        self._teardown_env()
        self._stop_api()
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

        for env_var, value in env.items():
            if env_var in os.environ:
                self._old_env[env_var] = os.environ[env_var]

            if value:
                os.environ[env_var] = value
        mlrun.mlconf.reload()

    def _teardown_env(self):
        self._logger.debug("Tearing down test environment")
        for env_var in self._test_env:
            if env_var in os.environ:
                del os.environ[env_var]
        os.environ.update(self._old_env)
        mlrun.mlconf.reload()

    @pytest.fixture(scope="function", autouse=True)
    def _api(self, db_engine):
        self._run_api(db_engine)
        self.api_url = self._resolve_mlrun_api_url()
        self._setup_env({"MLRUN_DBPATH": self.api_url})
        self._check_api_is_healthy(self.api_url)
        self._logger.info("Successfully started API", api_url=self.api_url)
        self.custom_setup()

        try:
            yield
        finally:
            self.custom_teardown()
            self._stop_api()

    @classmethod
    def _run_api(
        cls,
        db_engine: sqlalchemy.engine.Engine,
        *,
        publish_port: typing.Union[int, str] = 8080,
        container_name: str = "mlrun-api",
        image: Optional[str] = None,
        wait_timeout: int = 60,
    ) -> None:
        cls._logger.debug("Starting API")

        url_obj = sqlalchemy.engine.make_url(db_engine.url).set(
            host="host.docker.internal"
        )

        env_vars = {
            "MLRUN_HTTPDB__DSN": url_obj.render_as_string(hide_password=False),
            "MLRUN_VERSION": "0.0.0+unstable",
            "MLRUN_LOG_LEVEL": "DEBUG",
            "MLRUN_LOG_FORMATTER": mlrun.utils.FormatterKinds.HUMAN_EXTENDED.value,
            "MLRUN_SECRET_STORES__TEST_MODE_MOCK_SECRETS": "True",
        }
        if real_path := os.getenv("MLRUN_HTTPDB__REAL_PATH"):
            env_vars["MLRUN_HTTPDB__REAL_PATH"] = real_path

        registry = os.getenv("MLRUN_DOCKER_REGISTRY", "ghcr.io/").rstrip("/")
        tag = os.getenv("MLRUN_DOCKER_CACHE_FROM_TAG", "unstable")
        image = image or f"{registry}/mlrun/mlrun-api:{tag}"

        client = docker.from_env()

        cls._stop_api()
        cls._logger.debug("Running API")

        container = client.containers.run(
            image=image,
            name=container_name,
            detach=True,
            ports={"8080/tcp": publish_port},
            extra_hosts={"host.docker.internal": "host-gateway"},
            environment=env_vars,
        )

        # Wait until the API responds healthy
        deadline = time.time() + wait_timeout
        health_url = f"http://localhost:{publish_port}/api/v1/client-spec"
        while time.time() < deadline:
            try:
                if requests.get(health_url, timeout=2).status_code == 200:
                    cls._logger.debug("API is ready")
                    return
            except requests.RequestException:
                pass
            time.sleep(1)

        # If we got here, container started but never became healthy
        logs = container.logs(tail=100).decode()
        raise RuntimeError(
            f"mlrun‑api failed to become ready within {wait_timeout}s.\nLast logs:\n{logs}"
        )

    @classmethod
    def _stop_api(cls):
        client = cls._docker_client()
        cls._logger.debug("Stopping API container")
        try:
            client.containers.get(cls.api_container_name).remove(force=True)
        except docker.errors.NotFound:
            pass

    def _resolve_mlrun_api_url(self):
        client = self._docker_client()
        container = client.containers.get(self.api_container_name)
        ports = container.attrs["NetworkSettings"]["Ports"]
        host_port = ports["8080/tcp"][0]["HostPort"]
        return f"http://0.0.0.0:{host_port}"

    @classmethod
    def _remove_container(cls, container_id):
        client = cls._docker_client()
        try:
            container = client.containers.get(container_id)
            container.remove(force=True)
        except docker.errors.NotFound:
            pass

    @classmethod
    def _log_container_logs(cls, container_id):
        client = cls._docker_client()
        try:
            container = client.containers.get(container_id)
            logs = container.logs().decode()
            logs = logs.replace("\n", "\n\t")
            cls._logger.debug(
                f"Retrieved container logs:\n {logs}",
                container_name=container_id,
            )
        except docker.errors.NotFound:
            cls._logger.debug(
                "Container not found for logs", container_name=container_id
            )

    @staticmethod
    def _docker_client():
        return docker.from_env()

    @staticmethod
    def _extend_current_env(env):
        current_env = copy.deepcopy(os.environ)
        current_env.update(env)
        return current_env

    @staticmethod
    def _check_api_is_healthy(url):
        health_url = f"{url}/{mlrun.db.httpdb.HTTPRunDB.get_api_path_prefix()}/healthz"
        timeout = 90
        if not tests.conftest.wait_for_server(health_url, timeout):
            raise RuntimeError(f"API did not start after {timeout} sec")

    @staticmethod
    def _run_command(command, args=None, cwd=None, env=None):
        if args:
            command += " " + " ".join(args)

        process = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            encoding="utf-8",
            cwd=cwd,
            env=env,
        )
        return process.returncode
