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
import os
import pathlib
import sys
import typing

import pytest
import yaml

import mlrun.api.schemas
from mlrun import get_run_db, mlconf, set_environment
from mlrun.utils import create_logger

logger = create_logger(level="debug", name="test-system")


class TestMLRunSystem:
    project_name = "system-test-project"
    root_path = pathlib.Path(__file__).absolute().parent.parent.parent
    env_file_path = root_path / "tests" / "system" / "env.yml"
    results_path = root_path / "tests" / "test_results" / "system"
    enterprise_marker_name = "enterprise"
    mandatory_env_vars = [
        "MLRUN_DBPATH",
    ]
    mandatory_enterprise_env_vars = mandatory_env_vars + [
        "V3IO_API",
        "V3IO_FRAMESD",
        "V3IO_USERNAME",
        "V3IO_ACCESS_KEY",
        "MLRUN_SYSTEM_TESTS_DEFAULT_SPARK_SERVICE",
    ]

    _logger = logger

    _test_env = {}
    _old_env = {}

    @classmethod
    def setup_class(cls):
        env = cls._get_env_from_file()
        cls._test_env.update(env)
        cls._setup_env(cls._get_env_from_file())
        cls._run_db = get_run_db()
        cls.custom_setup_class()

        # the dbpath is already configured on the test startup before this stage
        # so even though we set the env var, we still need to directly configure
        # it in mlconf.
        mlconf.dbpath = cls._test_env["MLRUN_DBPATH"]

    @classmethod
    def custom_setup_class(cls):
        pass

    def setup_method(self, method):
        logger.info(f"Setting up test {self.__class__.__name__}::{method.__name__}")

        self._setup_env(self._get_env_from_file())
        self._run_db = get_run_db()

        if not self._skip_set_environment():
            set_environment(project=self.project_name)
            self.project = mlrun.get_or_create_project(self.project_name, "./")

        self.custom_setup()

        logger.info(
            f"Finished setting up test {self.__class__.__name__}::{method.__name__}"
        )

    @staticmethod
    def _should_clean_resources():
        return os.environ.get("MLRUN_SYSTEM_TESTS_CLEAN_RESOURCES") != "false"

    def _delete_test_project(self, name=None):
        if self._should_clean_resources():
            self._run_db.delete_project(
                name or self.project_name,
                deletion_strategy=mlrun.api.schemas.DeletionStrategy.cascading,
            )

    def teardown_method(self, method):
        logger.info(f"Tearing down test {self.__class__.__name__}::{method.__name__}")

        logger.debug("Removing test data from database")
        if self._should_clean_resources():
            fsets = self._run_db.list_feature_sets()
            if fsets:
                for fset in fsets:
                    fset.purge_targets()

        # self._delete_test_project()

        self.custom_teardown()

        logger.info(
            f"Finished tearing down test {self.__class__.__name__}::{method.__name__}"
        )

    @classmethod
    def teardown_class(cls):
        cls.custom_teardown_class()
        cls._teardown_env()

    def custom_setup(self):
        pass

    def custom_teardown(self):
        pass

    @classmethod
    def custom_teardown_class(cls):
        pass

    @staticmethod
    def _skip_set_environment():
        return False

    @classmethod
    def skip_test_if_env_not_configured(cls, test):
        mandatory_env_vars = (
            cls.mandatory_enterprise_env_vars
            if cls._has_marker(test, cls.enterprise_marker_name)
            else cls.mandatory_env_vars
        )
        configured = True
        try:
            env = cls._get_env_from_file()
        except FileNotFoundError:
            configured = False
        else:
            for env_var in mandatory_env_vars:
                if env_var not in env or env[env_var] is None:
                    configured = False

        return pytest.mark.skipif(
            not configured,
            reason=f"This is a system test, add the needed environment variables {*mandatory_env_vars,} "
            "in tests/system/env.yml to run it",
        )(test)

    @classmethod
    def is_enterprise_environment(cls):
        try:
            env = cls._get_env_from_file()
        except FileNotFoundError:
            return False
        else:
            for env_var in cls.mandatory_enterprise_env_vars:
                if env_var not in env or env[env_var] is None:
                    return False
            return True

    @classmethod
    def get_assets_path(cls):
        return (
            pathlib.Path(sys.modules[cls.__module__].__file__).absolute().parent
            / "assets"
        )

    @property
    def assets_path(self):
        return (
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent
            / "assets"
        )

    @classmethod
    def _get_env_from_file(cls) -> dict:
        with cls.env_file_path.open() as f:
            return yaml.safe_load(f)

    @classmethod
    def _setup_env(cls, env: dict):
        logger.debug("Setting up test environment")
        cls._test_env.update(env)

        # save old env vars for returning them on teardown
        for env_var, value in env.items():
            if env_var in os.environ:
                cls._old_env[env_var] = os.environ[env_var]

            if value:
                os.environ[env_var] = value

        # reload the config so changes to the env vars will take effect
        mlrun.config.config.reload()

    @classmethod
    def _teardown_env(cls):
        logger.debug("Tearing down test environment")
        for env_var in cls._test_env:
            if env_var in os.environ:
                del os.environ[env_var]
        os.environ.update(cls._old_env)
        # reload the config so changes to the env vars will take affect
        mlrun.config.config.reload()

    def _get_v3io_user_store_path(self, path: pathlib.Path, remote: bool = True) -> str:
        v3io_user = self._test_env["V3IO_USERNAME"]
        prefixes = {
            "remote": f"v3io:///users/{v3io_user}",
            "local": "/User",
        }
        prefix = prefixes["remote"] if remote else prefixes["local"]
        return prefix + str(path)

    def _verify_run_spec(
        self,
        run_spec,
        parameters: dict = None,
        inputs: dict = None,
        outputs: list = None,
        output_path: str = None,
        function: str = None,
        secret_sources: list = None,
        data_stores: list = None,
        scrape_metrics: bool = None,
    ):
        logger.debug("Verifying run spec", spec=run_spec)
        if parameters:
            assert run_spec["parameters"] == parameters
        if inputs:
            assert run_spec["inputs"] == inputs
        if outputs:
            assert run_spec["outputs"] == outputs
        if output_path:
            assert run_spec["output_path"] == output_path
        if function:
            assert run_spec["function"] == function
        if secret_sources:
            assert run_spec["secret_sources"] == secret_sources
        if data_stores:
            assert run_spec["data_stores"] == data_stores
        if scrape_metrics is not None:
            assert run_spec["scrape_metrics"] == scrape_metrics

    def _verify_run_metadata(
        self,
        run_metadata,
        uid: str = None,
        name: str = None,
        project: str = None,
        labels: dict = None,
        iteration: int = None,
    ):
        logger.debug("Verifying run metadata", spec=run_metadata)
        if uid:
            assert run_metadata["uid"] == uid
        if name:
            assert run_metadata["name"] == name
        if project:
            assert run_metadata["project"] == project
        if iteration:
            assert run_metadata["iteration"] == project
        if labels:
            for label, label_value in labels.items():
                assert label in run_metadata["labels"]
                assert run_metadata["labels"][label] == label_value

    def _verify_run_outputs(
        self,
        run_outputs,
        uid: str,
        name: str,
        project: str,
        output_path: pathlib.Path,
        accuracy: int = None,
        loss: int = None,
        best_iteration: int = None,
        iteration_results: bool = False,
    ):
        logger.debug("Verifying run outputs", spec=run_outputs)
        assert run_outputs["model"].startswith(str(output_path))
        assert run_outputs["html_result"].startswith(str(output_path))
        assert run_outputs["chart"].startswith(str(output_path))
        assert run_outputs["mydf"] == f"store://artifacts/{project}/{name}_mydf:{uid}"
        if accuracy:
            assert run_outputs["accuracy"] == accuracy
        if loss:
            assert run_outputs["loss"] == loss
        if best_iteration:
            assert run_outputs["best_iteration"] == best_iteration
        if iteration_results:
            assert run_outputs["iteration_results"].startswith(str(output_path))

    @staticmethod
    def _has_marker(test: typing.Callable, marker_name: str) -> bool:
        try:
            return (
                len([mark for mark in test.pytestmark if mark.name == marker_name]) > 0
            )
        except AttributeError:
            return False
