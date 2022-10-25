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
import unittest
from http import HTTPStatus
from os import environ
from typing import Callable, Generator
from unittest.mock import Mock

import deepdiff
import pytest
import requests
import v3io.dataplane

import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.k8s
import mlrun.api.utils.singletons.logs_dir
import mlrun.api.utils.singletons.project_member
import mlrun.api.utils.singletons.scheduler
import mlrun.config
import mlrun.datastore
import mlrun.db
import mlrun.k8s_utils
import mlrun.utils
import mlrun.utils.singleton
from mlrun.api.db.sqldb.db import SQLDB
from mlrun.api.db.sqldb.session import _init_engine, create_session
from mlrun.api.initial_data import init_data
from mlrun.api.utils.singletons.db import initialize_db
from mlrun.config import config
from mlrun.runtimes import BaseRuntime
from mlrun.runtimes.function import NuclioStatus
from mlrun.runtimes.utils import global_context
from tests.conftest import logs_path, root_path, rundb_path

session_maker: Callable


@pytest.fixture(autouse=True)
# if we'll just call it config it may be overridden by other fixtures with the same name
def config_test_base():
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

    # remove the run db cache so it won't pass between tests
    mlrun.db._run_db = None
    mlrun.db._last_db_url = None
    mlrun.datastore.store_manager._db = None
    mlrun.datastore.store_manager._stores = {}

    # remove singletons in case they were changed (we don't want changes to pass between tests)
    mlrun.utils.singleton.Singleton._instances = {}

    mlrun.api.utils.singletons.db.db = None
    mlrun.api.utils.singletons.project_member.project_member = None
    mlrun.api.utils.singletons.scheduler.scheduler = None
    mlrun.api.utils.singletons.k8s._k8s = None
    mlrun.api.utils.singletons.logs_dir.logs_dir = None

    mlrun.k8s_utils._k8s = None
    mlrun.runtimes.runtime_handler_instances_cache = {}
    mlrun.runtimes.utils.cached_mpijob_crd_version = None
    mlrun.runtimes.utils.cached_nuclio_version = None


@pytest.fixture
def db():
    global session_maker
    dsn = "sqlite:///:memory:?check_same_thread=false"
    db_session = None
    try:
        config.httpdb.dsn = dsn
        _init_engine(dsn)
        init_data()
        initialize_db()
        db_session = create_session()
        db = SQLDB(dsn)
        db.initialize(db_session)
    finally:
        if db_session is not None:
            db_session.close()
    mlrun.api.utils.singletons.db.initialize_db(db)
    mlrun.api.utils.singletons.project_member.initialize_project_member()
    return db


@pytest.fixture()
def db_session() -> Generator:
    db_session = None
    try:
        db_session = create_session()
        yield db_session
    finally:
        if db_session is not None:
            db_session.close()


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
        self._function = None

    def reset(self):
        self._function = None
        self._pipeline = None

    # Expected to return a hash-key
    def store_function(self, function, name, project="", tag=None, versioned=False):
        self._function = function
        return "1234-1234-1234-1234"

    def store_run(self, struct, uid, project="", iter=0):
        self._run = {
            uid: {
                "struct": struct,
                "projct": project,
                "iter": iter,
            }
        }

    def get_function(self, function, project, tag):
        return {
            "name": function,
            "metadata": "bla",
            "uid": "1234-1234-1234-1234",
            "project": project,
            "tag": tag,
        }

    def submit_job(self, runspec, schedule=None):
        return {"status": {"status_text": "just a status"}}

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
        self._project_name = name
        self._project = project

    def get_project(self, name):
        if name == self._project_name:
            return self._project

    def remote_builder(
        self,
        func,
        with_mlrun,
        mlrun_version_specifier=None,
        skip_deployed=False,
        builder_env=None,
    ):
        self._function = func.to_dict()
        status = NuclioStatus(
            state="ready",
            nuclio_name="test-nuclio-name",
        )
        return {
            "data": {
                "status": status.to_dict(),
                "metadata": self._function.get("metadata"),
                "spec": self._function.get("spec"),
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

    def assert_no_mount_or_creds_configured(self):
        env_list = self._function["spec"]["env"]
        env_params = [item["name"] for item in env_list]
        for env_variable in [
            "V3IO_USERNAME",
            "V3IO_ACCESS_KEY",
            "V3IO_FRAMESD",
            "V3IO_API",
        ]:
            assert env_variable not in env_params

        volume_mounts = self._function["spec"]["volume_mounts"]
        volumes = self._function["spec"]["volumes"]
        assert len(volumes) == 0
        assert len(volume_mounts) == 0

    def assert_v3io_mount_or_creds_configured(
        self, v3io_user, v3io_access_key, cred_only=False
    ):
        env_list = self._function["spec"]["env"]
        env_dict = {item["name"]: item["value"] for item in env_list}
        expected_env = {
            "V3IO_USERNAME": v3io_user,
            "V3IO_ACCESS_KEY": v3io_access_key,
        }
        result = deepdiff.DeepDiff(env_dict, expected_env, ignore_order=True)
        # We allow extra env parameters
        result.pop("dictionary_item_removed")
        assert result == {}

        volume_mounts = self._function["spec"]["volume_mounts"]
        volumes = self._function["spec"]["volumes"]

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
                    "options": {"accessKey": v3io_access_key},
                },
                "name": "v3io",
            }
        ]

        assert deepdiff.DeepDiff(volumes, expected_volumes) == {}
        assert deepdiff.DeepDiff(volume_mounts, expected_mounts) == {}

    def assert_pvc_mount_configured(self, pvc_params):
        function_spec = self._function["spec"]

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

    def assert_s3_mount_configured(self, s3_params):
        env_list = self._function["spec"]["env"]
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

    def assert_env_variables(self, expected_env_dict):
        env_list = self._function["spec"]["env"]
        env_dict = {item["name"]: item["value"] for item in env_list}

        for key, value in expected_env_dict.items():
            assert env_dict[key] == value

    def verify_authorization(
        self,
        authorization_verification_input: mlrun.api.schemas.AuthorizationVerificationInput,
    ):
        pass


@pytest.fixture()
def rundb_mock() -> RunDBMock:
    mock_object = RunDBMock()

    orig_get_run_db = mlrun.db.get_run_db
    mlrun.db.get_run_db = unittest.mock.Mock(return_value=mock_object)
    mlrun.get_run_db = unittest.mock.Mock(return_value=mock_object)

    orig_use_remote_api = BaseRuntime._use_remote_api
    orig_get_db = BaseRuntime._get_db
    BaseRuntime._get_db = unittest.mock.Mock(return_value=mock_object)

    orig_db_path = config.dbpath
    config.dbpath = "http://localhost:12345"
    yield mock_object

    # Have to revert the mocks, otherwise scheduling tests (and possibly others) are failing
    mlrun.db.get_run_db = orig_get_run_db
    mlrun.get_run_db = orig_get_run_db
    BaseRuntime._use_remote_api = orig_use_remote_api
    BaseRuntime._get_db = orig_get_db
    config.dbpath = orig_db_path
