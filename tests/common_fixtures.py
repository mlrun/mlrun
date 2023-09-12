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
import inspect
import os
import shutil
import unittest
from datetime import datetime
from http import HTTPStatus
from os import environ
from pathlib import Path
from typing import Callable, List, Optional, Union
from unittest.mock import Mock

import deepdiff
import pytest
import requests
import v3io.dataplane
from aioresponses import aioresponses as aioresponses_

import mlrun.common.schemas
import mlrun.config
import mlrun.datastore
import mlrun.db
import mlrun.db.factory
import mlrun.k8s_utils
import mlrun.launcher.factory
import mlrun.projects.project
import mlrun.utils
import mlrun.utils.singleton
from mlrun.config import config
from mlrun.lists import ArtifactList
from mlrun.runtimes import BaseRuntime
from mlrun.runtimes.function import NuclioStatus
from mlrun.runtimes.utils import global_context
from mlrun.utils import update_in
from tests.conftest import logs_path, results, root_path, rundb_path

session_maker: Callable


@pytest.fixture(autouse=True)
# if we'll just call it config it may be overridden by other fixtures with the same name
def config_test_base():
    # recreating the test results path on each test instead of running it on conftest since
    # it is not a threadsafe operation. if we'll run it on conftest it will be called multiple times
    # in parallel and may cause errors.
    shutil.rmtree(results, ignore_errors=True, onerror=None)
    Path(f"{results}/kfp").mkdir(parents=True, exist_ok=True)

    environ["PYTHONPATH"] = root_path
    environ["MLRUN_DBPATH"] = rundb_path
    environ["MLRUN_httpdb__dirpath"] = rundb_path
    environ["MLRUN_httpdb__logs_path"] = logs_path
    environ["MLRUN_httpdb__projects__periodic_sync_interval"] = "0 seconds"
    environ["MLRUN_httpdb__projects__counters_cache_ttl"] = "0 seconds"
    environ["MLRUN_EXEC_CONFIG"] = ""
    global_context.set(None)
    log_level = "DEBUG"
    environ["MLRUN_log_level"] = log_level
    # reload config so that values overridden by tests won't pass to other tests
    mlrun.config.config.reload()

    # remove the run db cache, so it won't pass between tests
    mlrun.db._run_db = None
    mlrun.db._last_db_url = None
    mlrun.datastore.store_manager._db = None
    mlrun.datastore.store_manager._stores = {}

    # no need to raise error when using nop_db
    mlrun.mlconf.httpdb.nop_db.raise_error = False

    # remove the is_running_as_api cache, so it won't pass between tests
    mlrun.config._is_running_as_api = None
    # remove singletons in case they were changed (we don't want changes to pass between tests)
    mlrun.utils.singleton.Singleton._instances = {}

    mlrun.runtimes.runtime_handler_instances_cache = {}

    # TODO: update this to "sidecar" once the default mode is changed
    mlrun.config.config.log_collector.mode = "legacy"

    # revert change of default project after project creation
    mlrun.mlconf.default_project = "default"
    mlrun.projects.project.pipeline_context.set(None)

    # reset factory container overrides
    rundb_factory = mlrun.db.factory.RunDBFactory()
    rundb_factory._rundb_container.reset_override()
    launcher_factory = mlrun.launcher.factory.LauncherFactory()
    launcher_factory._launcher_container.reset_override()


@pytest.fixture
def aioresponses_mock():
    with aioresponses_() as aior:
        # handy function to get how many times requests were made using this specific mock
        aior.called_times = lambda: len(list(aior.requests.values())[0])
        yield aior


@pytest.fixture
def ensure_default_project() -> mlrun.projects.project.MlrunProject:
    return mlrun.get_or_create_project("default")


@pytest.fixture()
def chdir_to_test_location(request):
    """
    Fixture to change the working directory for tests,
    It allows seamless access to files relative to the test file.

    Because the working directory inside the dockerized test is '/mlrun',
    this fixture allows to automatically modify the cwd to the test file directory,
    to ensure the workflow files are located,
    and modify it back after the test case for other tests

    """
    original_working_dir = os.getcwd()
    test_file_path = os.path.dirname(inspect.getfile(request.function))
    os.chdir(test_file_path)

    yield

    os.chdir(original_working_dir)


@pytest.fixture
def patch_file_forbidden(monkeypatch):
    class MockV3ioClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_container_contents(self, *args, **kwargs):
            raise RuntimeError("Permission denied")

    mock_get = mock_failed_get_func(HTTPStatus.FORBIDDEN.value)

    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "head", mock_get)
    monkeypatch.setattr(v3io.dataplane, "Client", MockV3ioClient)


@pytest.fixture
def patch_file_not_found(monkeypatch):
    class MockV3ioClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_container_contents(self, *args, **kwargs):
            raise FileNotFoundError

    mock_get = mock_failed_get_func(HTTPStatus.NOT_FOUND.value)

    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "head", mock_get)
    monkeypatch.setattr(v3io.dataplane, "Client", MockV3ioClient)


def mock_failed_get_func(status_code: int):
    def mock_get(*args, **kwargs):
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.raise_for_status = Mock(
            side_effect=requests.HTTPError("Error", response=mock_response)
        )
        return mock_response

    return mock_get


# Mock class used for client-side runtime tests. This mocks the rundb interface, for running/deploying runtimes
class RunDBMock:
    def __init__(self):
        self.kind = "http"
        self._pipeline = None
        self._functions = {}
        self._artifacts = {}
        self._project_name = None
        self._project = None
        self._runs = {}

    def reset(self):
        self._functions = {}
        self._pipeline = None
        self._project_name = None
        self._project = None
        self._artifacts = {}

    # Expected to return a hash-key
    def store_function(self, function, name, project="", tag=None, versioned=False):
        hash_key = mlrun.utils.fill_function_hash(function, tag)
        self._functions[name] = function
        return hash_key

    def store_artifact(self, key, artifact, uid, iter=None, tag="", project=""):
        self._artifacts[key] = artifact
        return artifact

    def read_artifact(self, key, tag=None, iter=None, project=""):
        return self._artifacts.get(key, None)

    def list_artifacts(
        self,
        name="",
        project="",
        tag="",
        labels=None,
        since=None,
        until=None,
        kind=None,
        category=None,
        iter: int = None,
        best_iteration: bool = False,
        as_records: bool = False,
        use_tag_as_uid: bool = None,
    ):
        def filter_artifact(artifact):
            if artifact["metadata"].get("tag", None) == tag:
                return True

        return ArtifactList(filter(filter_artifact, self._artifacts.values()))

    def store_run(self, struct, uid, project="", iter=0):
        if hasattr(struct, "to_dict"):
            struct = struct.to_dict()

        if project:
            struct["metadata"]["project"] = project

        if iter:
            struct["status"]["iteration"] = iter

        self._runs[uid] = struct

    def read_run(self, uid, project, iter=0):
        return self._runs.get(uid, {})

    def list_runs(
        self,
        name: Optional[str] = None,
        uid: Optional[Union[str, List[str]]] = None,
        project: Optional[str] = None,
        labels: Optional[Union[str, List[str]]] = None,
        state: Optional[str] = None,
        sort: bool = True,
        last: int = 0,
        iter: bool = False,
        start_time_from: Optional[datetime] = None,
        start_time_to: Optional[datetime] = None,
        last_update_time_from: Optional[datetime] = None,
        last_update_time_to: Optional[datetime] = None,
        partition_by: Optional[
            Union[mlrun.common.schemas.RunPartitionByField, str]
        ] = None,
        rows_per_partition: int = 1,
        partition_sort_by: Optional[Union[mlrun.common.schemas.SortField, str]] = None,
        partition_order: Union[
            mlrun.common.schemas.OrderType, str
        ] = mlrun.common.schemas.OrderType.desc,
        max_partitions: int = 0,
        with_notifications: bool = False,
    ) -> mlrun.lists.RunList:
        return mlrun.lists.RunList(self._runs.values())

    def get_function(self, function, project, tag, hash_key=None):
        if function not in self._functions:
            raise mlrun.errors.MLRunNotFoundError("Function not found")
        return self._functions[function]

    def submit_job(self, runspec, schedule=None):
        return {"status": {"status_text": "just a status"}}

    def watch_log(self, uid, project="", watch=True, offset=0):
        # mock API updated the run status to completed
        self._runs[uid]["status"] = {"state": "completed"}
        return "completed", 0

    def submit_pipeline(
        self,
        project,
        pipeline,
        arguments,
        experiment,
        run,
        namespace,
        ops,
        artifact_path,
    ):
        self._pipeline = pipeline
        return True

    def store_project(self, name, project):
        return self.create_project(project)

    def create_project(self, project):
        if isinstance(project, dict):
            project = mlrun.projects.MlrunProject.from_dict(project)
        self._project = project
        self._project_name = project.name
        return self._project

    def get_project(self, name):
        if self._project_name and name == self._project_name:
            return self._project

        elif name == config.default_project and not self._project:
            project = mlrun.projects.MlrunProject(name)
            self.store_project(name, project)
            return project

        raise mlrun.errors.MLRunNotFoundError(f"Project '{name}' not found")

    def remote_builder(
        self,
        func,
        with_mlrun,
        mlrun_version_specifier=None,
        skip_deployed=False,
        builder_env=None,
    ):
        function = func.to_dict()
        status = NuclioStatus(
            state="ready",
            nuclio_name="test-nuclio-name",
        )
        self._functions[function["metadata"]["name"]] = function
        return {
            "data": {
                "status": status.to_dict(),
                "metadata": function.get("metadata"),
                "spec": function.get("spec"),
            }
        }

    def get_builder_status(
        self,
        func: BaseRuntime,
        offset=0,
        logs=True,
        last_log_timestamp=0,
        verbose=False,
    ):
        return "ready", last_log_timestamp

    def update_run(self, updates: dict, uid, project="", iter=0):
        for key, value in updates.items():
            update_in(self._runs[uid], key, value)

    def assert_no_mount_or_creds_configured(self, function_name=None):
        function = self._get_function_internal(function_name)

        env_list = function["spec"]["env"]
        env_params = [item["name"] for item in env_list]
        for env_variable in [
            "V3IO_USERNAME",
            "V3IO_ACCESS_KEY",
            "V3IO_FRAMESD",
            "V3IO_API",
        ]:
            assert env_variable not in env_params

        volume_mounts = function["spec"]["volume_mounts"]
        volumes = function["spec"]["volumes"]
        assert len(volumes) == 0
        assert len(volume_mounts) == 0

    def assert_v3io_mount_or_creds_configured(
        self, v3io_user, v3io_access_key, cred_only=False, function_name=None
    ):
        function = self._get_function_internal(function_name)
        env_list = function["spec"]["env"]
        env_dict = {item["name"]: item["value"] for item in env_list}
        expected_env = {
            "V3IO_USERNAME": v3io_user,
            "V3IO_ACCESS_KEY": v3io_access_key,
        }
        result = deepdiff.DeepDiff(env_dict, expected_env, ignore_order=True)
        # We allow extra env parameters
        result.pop("dictionary_item_removed")
        assert result == {}

        volume_mounts = function["spec"]["volume_mounts"]
        volumes = function["spec"]["volumes"]

        if cred_only:
            assert len(volumes) == 0
            assert len(volume_mounts) == 0
            return

        expected_mounts = [
            {"mountPath": "/v3io", "name": "v3io", "subPath": ""},
            {"mountPath": "/User", "name": "v3io", "subPath": f"users/{v3io_user}"},
        ]
        expected_volumes = [
            {
                "flexVolume": {
                    "driver": "v3io/fuse",
                    "options": {
                        "accessKey": v3io_access_key,
                        "dirsToCreate": f'[{{"name": "users//{v3io_user}", "permissions": 488}}]',
                    },
                },
                "name": "v3io",
            }
        ]

        assert deepdiff.DeepDiff(volumes, expected_volumes) == {}
        assert deepdiff.DeepDiff(volume_mounts, expected_mounts) == {}

    def assert_pvc_mount_configured(self, pvc_params, function_name=None):
        function_spec = self._get_function_internal(function_name)["spec"]

        expected_volumes = [
            {
                "name": pvc_params["volume_name"],
                "persistentVolumeClaim": {"claimName": pvc_params["pvc_name"]},
            }
        ]
        expected_mounts = [
            {
                "mountPath": pvc_params["volume_mount_path"],
                "name": pvc_params["volume_name"],
            }
        ]

        assert deepdiff.DeepDiff(function_spec["volumes"], expected_volumes) == {}
        assert deepdiff.DeepDiff(function_spec["volume_mounts"], expected_mounts) == {}

    def assert_s3_mount_configured(self, s3_params, function_name=None):
        function = self._get_function_internal(function_name)
        env_list = function["spec"]["env"]
        param_names = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        secret_name = s3_params.get("secret_name")
        non_anonymous = s3_params.get("non_anonymous")

        env_dict = {
            item["name"]: item["valueFrom"] if "valueFrom" in item else item["value"]
            for item in env_list
            if item["name"] in param_names + ["S3_NON_ANONYMOUS"]
        }

        if secret_name:
            expected_envs = {
                name: {"secretKeyRef": {"key": name, "name": secret_name}}
                for name in param_names
            }
        else:
            expected_envs = {
                "AWS_ACCESS_KEY_ID": s3_params["aws_access_key"],
                "AWS_SECRET_ACCESS_KEY": s3_params["aws_secret_key"],
            }
        if non_anonymous:
            expected_envs["S3_NON_ANONYMOUS"] = "true"
        assert expected_envs == env_dict

    def assert_env_variables(self, expected_env_dict, function_name=None):
        function = self._get_function_internal(function_name)
        env_list = function["spec"]["env"]
        env_dict = {item["name"]: item["value"] for item in env_list}

        for key, value in expected_env_dict.items():
            assert env_dict[key] == value

    def verify_authorization(
        self,
        authorization_verification_input: mlrun.common.schemas.AuthorizationVerificationInput,
    ):
        pass

    def _get_function_internal(self, function_name: str = None):
        if function_name:
            return self._functions[function_name]

        return list(self._functions.values())[0]

    def store_metric(self, uid, project="", keyvals=None, timestamp=None, labels=None):
        pass

    def list_hub_sources(self, *args, **kwargs):
        return [self._create_dummy_indexed_hub_source()]

    def get_hub_source(self, *args, **kwargs):
        return self._create_dummy_indexed_hub_source()

    def _create_dummy_indexed_hub_source(self):
        return mlrun.common.schemas.IndexedHubSource(
            index=1,
            source=mlrun.common.schemas.HubSource(
                metadata=mlrun.common.schemas.HubObjectMetadata(
                    name="default", description="some description"
                ),
                spec=mlrun.common.schemas.HubSourceSpec(
                    path=mlrun.mlconf.hub.default_source.url,
                    channel="master",
                    object_type="functions",
                ),
            ),
        )


@pytest.fixture()
def rundb_mock() -> RunDBMock:
    mock_object = RunDBMock()

    orig_get_run_db = mlrun.db.get_run_db
    mlrun.db.get_run_db = unittest.mock.Mock(return_value=mock_object)
    mlrun.get_run_db = unittest.mock.Mock(return_value=mock_object)

    orig_get_db = BaseRuntime._get_db
    BaseRuntime._get_db = unittest.mock.Mock(return_value=mock_object)

    orig_db_path = config.dbpath
    config.dbpath = "http://localhost:12345"

    # Create the default project to mimic real MLRun DB (the default project is always available for use):
    mlrun.get_or_create_project("default")

    yield mock_object

    # Have to revert the mocks, otherwise scheduling tests (and possibly others) are failing
    mlrun.db.get_run_db = orig_get_run_db
    mlrun.get_run_db = orig_get_run_db
    BaseRuntime._get_db = orig_get_db
    config.dbpath = orig_db_path
