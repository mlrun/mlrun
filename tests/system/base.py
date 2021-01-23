import os
import pathlib
import sys

import pytest
import yaml

import mlrun.api.schemas
from mlrun import get_run_db, mlconf, set_environment
from mlrun.utils import create_logger

logger = create_logger(level="debug", name="test")


class TestMLRunSystem:

    project_name = "system-test-project"
    root_path = pathlib.Path(__file__).absolute().parent.parent.parent
    env_file_path = root_path / "tests" / "system" / "env.yml"
    results_path = root_path / "tests" / "test_results" / "system"
    mandatory_env_vars = [
        "MLRUN_DBPATH",
        "V3IO_API",
        "V3IO_USERNAME",
        "V3IO_ACCESS_KEY",
    ]

    def setup_method(self, method):
        self._logger = logger
        self._logger.info(
            f"Setting up test {self.__class__.__name__}::{method.__name__}"
        )
        self._test_env = {}
        self._old_env = {}
        self._setup_env(self._get_env_from_file())

        self._run_db = get_run_db()

        # the dbpath is already configured on the test startup before this stage
        # so even though we set the env var, we still need to directly configure
        # it in mlconf.
        mlconf.dbpath = self._test_env["MLRUN_DBPATH"]

        set_environment(
            artifact_path="/User/data", project=self.project_name,
        )

        self.custom_setup()

        self._logger.info(
            f"Finished setting up test {self.__class__.__name__}::{method.__name__}"
        )

    def teardown_method(self, method):
        self._logger.info(
            f"Tearing down test {self.__class__.__name__}::{method.__name__}"
        )

        self.custom_teardown()

        self._logger.debug("Removing test data from database")
        self._run_db.delete_project(
            self.project_name,
            deletion_strategy=mlrun.api.schemas.DeletionStrategy.cascade,
        )

        self._teardown_env()
        self._logger.info(
            f"Finished tearing down test {self.__class__.__name__}::{method.__name__}"
        )

    def custom_setup(self):
        pass

    def custom_teardown(self):
        pass

    @classmethod
    def skip_test_if_env_not_configured(cls, test):
        configured = True
        try:
            env = cls._get_env_from_file()
        except FileNotFoundError:
            configured = False
        else:
            for env_var in cls.mandatory_env_vars:
                if env_var not in env or env[env_var] is None:
                    configured = False

        return pytest.mark.skipif(
            not configured,
            reason=f"This is a system test, add the needed environment variables {*cls.mandatory_env_vars,} "
            "in tests/system/env.yml to run it",
        )(test)

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

    def _setup_env(self, env: dict):
        self._logger.debug("Setting up test environment")
        self._test_env.update(env)

        # save old env vars for returning them on teardown
        for env_var, value in env.items():
            if env_var in os.environ:
                self._old_env[env_var] = os.environ[env_var]

            if value:
                os.environ[env_var] = value

    def _teardown_env(self):
        self._logger.debug("Tearing down test environment")
        for env_var in self._test_env:
            if env_var in os.environ:
                del os.environ[env_var]
        os.environ.update(self._old_env)

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
        self._logger.debug("Verifying run spec", spec=run_spec)
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
        self._logger.debug("Verifying run metadata", spec=run_metadata)
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
        self._logger.debug("Verifying run outputs", spec=run_outputs)
        iterpath = str(best_iteration) if best_iteration else ""
        assert run_outputs["model"] == str(output_path / iterpath / "model.txt")
        assert run_outputs["html_result"] == str(output_path / iterpath / "result.html")
        assert run_outputs["chart"] == str(output_path / iterpath / "chart.html")
        assert run_outputs["mydf"] == f"store://artifacts/{project}/{name}_mydf:{uid}"
        if accuracy:
            assert run_outputs["accuracy"] == accuracy
        if loss:
            assert run_outputs["loss"] == loss
        if best_iteration:
            assert run_outputs["best_iteration"] == best_iteration
        if iteration_results:
            assert run_outputs["iteration_results"] == str(
                output_path / "iteration_results.csv"
            )
