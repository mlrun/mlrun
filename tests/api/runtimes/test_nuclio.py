import os
import pytest
from tests.api.runtimes.base import TestRuntimeBase
from mlrun.runtimes.function import deploy_nuclio_function
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from mlrun import code_to_function
import unittest.mock
import nuclio
import deepdiff
import base64
from mlrun.platforms.iguazio import split_path


class TestNuclioRuntime(TestRuntimeBase):
    @pytest.fixture(autouse=True)
    def setup_method_fixture(self, db: Session, client: TestClient):
        # We want this mock for every test, ideally we would have simply put it in the custom_setup
        # but this function is called by the base class's setup_method which is happening before the fixtures
        # initialization. We need the client fixture (which needs the db one) in order to be able to mock k8s stuff
        self._mock_nuclio_deploy_config()

    def custom_setup(self):
        self.image_name = "test/image:latest"
        self.code_handler = "test_func"
        os.environ["V3IO_ACCESS_KEY"] = self.v3io_access_key = "1111-2222-3333-4444"
        os.environ["V3IO_USERNAME"] = self.v3io_user = "test-user"

    @staticmethod
    def _mock_nuclio_deploy_config():
        nuclio.deploy.deploy_config = unittest.mock.Mock()

    @staticmethod
    def _get_expected_struct_for_http_trigger(parameters):
        expected_struct = {
            "kind": "http",
            "maxWorkers": parameters["workers"],
            "attributes": {
                "ingresses": {
                    "0": {
                        "host": parameters["host"],
                        "paths": parameters["paths"],
                        "secretName": parameters["secret"],
                    }
                },
                "port": parameters["port"],
            },
        }
        if "canary" in parameters:
            expected_struct["annotations"] = {
                "nginx.ingress.kubernetes.io/canary": "true",
                "nginx.ingress.kubernetes.io/canary-weight": parameters["canary"],
            }
        return expected_struct

    def _get_expected_struct_for_v3io_trigger(self, parameters):
        container, path = split_path(parameters["stream_path"])
        # Remove leading / in the path
        path = path[1:]

        # TODO - Not sure what happens to the "shards" parameter. Seems to be dropped along the way?

        return {
            "kind": "v3ioStream",
            "name": parameters["name"],
            "password": self.v3io_access_key,
            "attributes": {
                "containerName": container,
                "streamPath": path,
                "consumerGroup": parameters["group"],
                "seekTo": parameters["seek_to"],
            },
        }

    def _generate_runtime(self, kind="nuclio"):
        runtime = code_to_function(
            name=self.name,
            project=self.project,
            filename=self.code_filename,
            handler=self.code_handler,
            kind=kind,
            image=self.image_name,
            description="test function",
        )
        return runtime

    def _assert_deploy_called_basic_config(self):
        deploy_mock = nuclio.deploy.deploy_config
        deploy_mock.assert_called_once()

        full_function_name = f"{self.project}-{self.name}"

        args, kwargs = deploy_mock.call_args

        assert kwargs["name"] == full_function_name
        assert kwargs["project"] == self.project

        deploy_config = args[0]
        function_metadata = deploy_config["metadata"]
        assert function_metadata["name"] == full_function_name
        expected_labels = {"mlrun/class": "remote"}
        assert deepdiff.DeepDiff(function_metadata["labels"], expected_labels) == {}

        code_base64 = base64.b64encode(open(self.code_filename, "rb").read()).decode(
            "utf-8"
        )
        build_info = deploy_config["spec"]["build"]
        assert build_info["functionSourceCode"] == code_base64
        assert build_info["baseImage"] == self.image_name

    def _assert_triggers(self, http_trigger=None, v3io_trigger=None):
        args, _ = nuclio.deploy.deploy_config.call_args
        triggers_config = args[0]["spec"]["triggers"]

        if http_trigger:
            expected_struct = self._get_expected_struct_for_http_trigger(http_trigger)
            assert (
                deepdiff.DeepDiff(
                    triggers_config["http"],
                    expected_struct,
                    ignore_order=True,
                    # TODO - (in Nuclio) There is a bug with canary configuration:
                    #        the nginx.ingress.kubernetes.io/canary-weight annotation gets assigned the host name
                    #        rather than the actual weight. Remove this once bug is fixed.
                    exclude_paths=[
                        "root['annotations']['nginx.ingress.kubernetes.io/canary-weight']"
                    ],
                )
                == {}
            )

        if v3io_trigger:
            expected_struct = self._get_expected_struct_for_v3io_trigger(v3io_trigger)
            diff_result = deepdiff.DeepDiff(
                triggers_config[v3io_trigger["name"]],
                expected_struct,
                ignore_order=True,
            )
            # It's ok if the Nuclio trigger has additional parameters, these are constants that we don't care
            # about. We just care that the values we look for are fully there.
            diff_result.pop("dictionary_item_removed", None)
            assert diff_result == {}

    def _assert_nuclio_v3io_mount(self, local_path, remote_path):
        args, _ = nuclio.deploy.deploy_config.call_args
        deploy_spec = args[0]["spec"]
        container, path = split_path(remote_path)

        expected_volume = {
            "volume": {
                "flexVolume": {
                    "driver": "v3io/fuse",
                    "options": {
                        "accessKey": self.v3io_access_key,
                        "container": container,
                        "subPath": path,
                    },
                },
                "name": "v3io",
            },
            "volumeMount": {"mountPath": local_path, "name": "v3io", "subPath": ""},
        }
        assert (
            deepdiff.DeepDiff(
                deploy_spec["volumes"], [expected_volume], ignore_order=True
            )
            == {}
        )

        env_config = deploy_spec["env"]
        expected_env = {
            "V3IO_ACCESS_KEY": self.v3io_access_key,
            "V3IO_USERNAME": self.v3io_user,
            "V3IO_API": None,
            "MLRUN_NAMESPACE": self.namespace,
        }
        self._assert_pod_env(env_config, expected_env)

    def test_deploy_basic_function(self, db: Session, client: TestClient):
        function = self._generate_runtime("nuclio")

        deploy_nuclio_function(function)
        self._assert_deploy_called_basic_config()

    def test_deploy_with_triggers(self, db: Session, client: TestClient):
        function = self._generate_runtime("nuclio")

        http_trigger = {
            "workers": 2,
            "port": 12345,
            "host": "http://my.host",
            "paths": ["/path/1", "/path/2"],
            "secret": "my little secret",
            "canary": 50,
        }

        v3io_trigger = {
            "stream_path": "/container/and/path",
            "name": "test_stream",
            "group": "beatles",
            "seek_to": "latest",
            "shards": 42,
        }

        function.with_http(**http_trigger)
        function.add_v3io_stream_trigger(**v3io_trigger)

        deploy_nuclio_function(function)
        self._assert_deploy_called_basic_config()
        self._assert_triggers(http_trigger, v3io_trigger)

    def test_deploy_with_v3io(self, db: Session, client: TestClient):
        function = self._generate_runtime("nuclio")
        local_path = "/local/path"
        remote_path = "/container/and/path"
        function.with_v3io(local_path, remote_path)

        deploy_nuclio_function(function)
        self._assert_deploy_called_basic_config()
        self._assert_nuclio_v3io_mount(local_path, remote_path)
