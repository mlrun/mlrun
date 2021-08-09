import unittest.mock
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Generator

import deepdiff
import pytest
from fastapi.testclient import TestClient

import mlrun.api.utils.singletons.k8s
from mlrun import mlconf
from mlrun.api.db.sqldb.session import _init_engine, create_session
from mlrun.api.initial_data import init_data
from mlrun.api.main import app
from mlrun.api.utils.singletons.db import initialize_db
from mlrun.api.utils.singletons.project_member import initialize_project_member
from mlrun.config import config
from mlrun.runtimes import BaseRuntime
from mlrun.runtimes.function import NuclioStatus
from mlrun.utils import logger


@pytest.fixture()
def db() -> Generator:
    """
    This fixture initialize the db singleton (so it will be accessible using mlrun.api.singletons.get_db()
    and generates a db session that can be used by the test
    """
    db_file = NamedTemporaryFile(suffix="-mlrun.db")
    logger.info(f"Created temp db file: {db_file.name}")
    config.httpdb.db_type = "sqldb"
    dsn = f"sqlite:///{db_file.name}?check_same_thread=false"
    config.httpdb.dsn = dsn

    # TODO: make it simpler - doesn't make sense to call 3 different functions to initialize the db
    # we need to force re-init the engine cause otherwise it is cached between tests
    _init_engine(config.httpdb.dsn)

    # forcing from scratch because we created an empty file for the db
    init_data(from_scratch=True)
    initialize_db()
    initialize_project_member()

    # we're also running client code in tests so set dbpath as well
    # note that setting this attribute triggers connection to the run db therefore must happen after the initialization
    config.dbpath = dsn
    yield create_session()
    logger.info(f"Removing temp db file: {db_file.name}")
    db_file.close()


@pytest.fixture()
def client(db) -> Generator:
    with TemporaryDirectory(suffix="mlrun-logs") as log_dir:
        mlconf.httpdb.logs_path = log_dir
        mlconf.runs_monitoring_interval = 0
        mlconf.runtimes_cleanup_interval = 0
        mlconf.httpdb.projects.periodic_sync_interval = "0 seconds"

        with TestClient(app) as c:
            yield c


class K8sSecretsMock:
    def __init__(self):
        # project -> secret_key -> secret_value
        self.project_secrets_map = {}

    def store_project_secrets(self, project, secrets, namespace=""):
        self.project_secrets_map.setdefault(project, {}).update(secrets)

    def delete_project_secrets(self, project, secrets, namespace=""):
        if not secrets:
            self.project_secrets_map.pop(project, None)
        else:
            for key in secrets:
                self.project_secrets_map.get(project, {}).pop(key, None)

    def get_project_secret_keys(self, project, namespace=""):
        return list(self.project_secrets_map.get(project, {}).keys())

    def get_project_secret_data(self, project, secret_keys=None, namespace=""):
        secrets_data = self.project_secrets_map.get(project, {})
        return {
            key: value
            for key, value in secrets_data.items()
            if (secret_keys and key in secret_keys) or not secret_keys
        }

    def assert_project_secrets(self, project: str, secrets: dict):
        assert (
            deepdiff.DeepDiff(
                self.project_secrets_map[project], secrets, ignore_order=True,
            )
            == {}
        )


@pytest.fixture()
def k8s_secrets_mock(client: TestClient) -> K8sSecretsMock:
    logger.info("Creating k8s secrets mock")
    k8s_secrets_mock = K8sSecretsMock()
    config.namespace = "default-tenant"

    mlrun.api.utils.singletons.k8s.get_k8s().is_running_inside_kubernetes_cluster = unittest.mock.Mock(
        return_value=True
    )
    mlrun.api.utils.singletons.k8s.get_k8s().get_project_secret_keys = unittest.mock.Mock(
        side_effect=k8s_secrets_mock.get_project_secret_keys
    )
    mlrun.api.utils.singletons.k8s.get_k8s().get_project_secret_data = unittest.mock.Mock(
        side_effect=k8s_secrets_mock.get_project_secret_data
    )
    mlrun.api.utils.singletons.k8s.get_k8s().store_project_secrets = unittest.mock.Mock(
        side_effect=k8s_secrets_mock.store_project_secrets
    )
    mlrun.api.utils.singletons.k8s.get_k8s().delete_project_secrets = unittest.mock.Mock(
        side_effect=k8s_secrets_mock.delete_project_secrets
    )

    return k8s_secrets_mock


# Mock class used for client-side runtime tests. This mocks the rundb interface, for running/deploying runtimes
class RunDBMock:
    def __init__(self):
        self._function = None
        self._runspec = None

    # Expected to return a hash-key
    def store_function(self, function, name, project="", tag=None, versioned=False):
        self._function = function
        return "1234-1234-1234-1234"

    def submit_job(self, runspec, schedule=None):
        self._runspec = runspec
        return {"status": {"status_text": "just a status"}}

    def remote_builder(
        self, func, with_mlrun, mlrun_version_specifier=None, skip_deployed=False
    ):
        self._function = func.to_dict()
        status = NuclioStatus(
            state="ready",
            nuclio_name="test-nuclio-name",
        )
        return {"data": {"status": status.to_dict()}}

    def get_builder_status(
        self,
        func: BaseRuntime,
        offset=0,
        logs=True,
        last_log_timestamp=0,
        verbose=False,
    ):
        return "ready", last_log_timestamp

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


@pytest.fixture()
def rundb_mock(client: TestClient) -> RunDBMock:
    logger.info("Creating rundb mock")
    rundb_mock = RunDBMock()

    mlrun.db.get_run_db = unittest.mock.Mock(return_value=rundb_mock)

    BaseRuntime._use_remote_api = unittest.mock.Mock(return_value=True)
    BaseRuntime._get_db = unittest.mock.Mock(return_value=rundb_mock)

    mlconf.dbpath = "http://localhost:12345"
    return rundb_mock