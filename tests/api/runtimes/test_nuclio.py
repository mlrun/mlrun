import base64
import json
import os
import unittest.mock

import deepdiff
import nuclio
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.errors
from mlrun import code_to_function
from mlrun.platforms.iguazio import split_path
from mlrun.runtimes.constants import NuclioIngressAddTemplatedIngressModes
from mlrun.runtimes.function import (
    compile_function_config,
    deploy_nuclio_function,
    enrich_function_with_ingress,
    resolve_function_ingresses,
)
from tests.api.runtimes.base import TestRuntimeBase


class TestNuclioRuntime(TestRuntimeBase):
    def custom_setup_after_fixtures(self):
        self._mock_nuclio_deploy_config()

    def custom_setup(self):
        self.image_name = "test/image:latest"
        self.code_handler = "test_func"

        os.environ["V3IO_ACCESS_KEY"] = self.v3io_access_key = "1111-2222-3333-4444"
        os.environ["V3IO_USERNAME"] = self.v3io_user = "test-user"

    @staticmethod
    def _mock_nuclio_deploy_config():
        nuclio.deploy.deploy_config = unittest.mock.Mock(return_value="some-server")

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

    def _assert_deploy_called_basic_config(
        self, expected_class="remote", call_count=1, expected_params=[]
    ):
        deploy_mock = nuclio.deploy.deploy_config
        assert deploy_mock.call_count == call_count

        call_args_list = deploy_mock.call_args_list
        for single_call_args in call_args_list:
            args, kwargs = single_call_args
            if expected_params:
                current_parameters = expected_params.pop(0)
                expected_function_name = current_parameters["function_name"]
                source_filename = current_parameters["file_name"]
            else:
                expected_function_name = f"{self.project}-{self.name}"
                source_filename = self.code_filename

            assert kwargs["name"] == expected_function_name
            assert kwargs["project"] == self.project

            deploy_config = args[0]
            function_metadata = deploy_config["metadata"]
            assert function_metadata["name"] == expected_function_name
            expected_labels = {"mlrun/class": expected_class}
            assert deepdiff.DeepDiff(function_metadata["labels"], expected_labels) == {}

            build_info = deploy_config["spec"]["build"]

            # Nuclio source code in some cases adds a suffix to the code, initializing nuclio context.
            # We just verify that the code provided starts with our code.
            original_source_code = open(source_filename, "r").read()
            spec_source_code = base64.b64decode(
                build_info["functionSourceCode"]
            ).decode("utf-8")
            assert spec_source_code.startswith(original_source_code)

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

    def _assert_node_selections(
        self,
        expected_node_name=None,
        expected_node_selector=None,
        expected_affinity=None,
    ):
        args, _ = nuclio.deploy.deploy_config.call_args
        deploy_spec = args[0]["spec"]

        if expected_node_name:
            assert deploy_spec["nodeName"] == expected_node_name

        if expected_node_selector:
            assert (
                deepdiff.DeepDiff(
                    deploy_spec["nodeSelector"],
                    expected_node_selector,
                    ignore_order=True,
                )
                == {}
            )
        if expected_affinity:
            assert (
                deepdiff.DeepDiff(
                    deploy_spec["affinity"].to_dict(),
                    expected_affinity.to_dict(),
                    ignore_order=True,
                )
                == {}
            )

    def test_enrich_with_ingress_no_overriding(self, db: Session, client: TestClient):
        """
        Expect no ingress template to be created, thought its mode is "always",
        since the function already have a pre-configured ingress
        """
        function = self._generate_runtime("nuclio")

        # both ingress and node port
        ingress_host = "something.com"
        function.with_http(host=ingress_host, paths=["/"], port=30030)
        function_name, project_name, config = compile_function_config(function)
        service_type = "NodePort"
        enrich_function_with_ingress(
            config, NuclioIngressAddTemplatedIngressModes.always, service_type
        )
        ingresses = resolve_function_ingresses(config["spec"])
        assert len(ingresses) > 0, "Expected one ingress to be created"
        for ingress in ingresses:
            assert "hostTemplate" not in ingress, "No host template should be added"
            assert ingress["host"] == ingress_host

    def test_enrich_with_ingress_always(self, db: Session, client: TestClient):
        """
        Expect ingress template to be created as the configuration templated ingress mode is "always"
        """
        function = self._generate_runtime("nuclio")
        function_name, project_name, config = compile_function_config(function)
        service_type = "NodePort"
        enrich_function_with_ingress(
            config, NuclioIngressAddTemplatedIngressModes.always, service_type
        )
        ingresses = resolve_function_ingresses(config["spec"])
        assert ingresses[0]["hostTemplate"] != ""

    def test_enrich_with_ingress_on_cluster_ip(self, db: Session, client: TestClient):
        """
        Expect ingress template to be created as the configuration templated ingress mode is "onClusterIP" while the
        function service type is ClusterIP
        """
        function = self._generate_runtime("nuclio")
        function_name, project_name, config = compile_function_config(function)
        service_type = "ClusterIP"
        enrich_function_with_ingress(
            config, NuclioIngressAddTemplatedIngressModes.on_cluster_ip, service_type,
        )
        ingresses = resolve_function_ingresses(config["spec"])
        assert ingresses[0]["hostTemplate"] != ""

    def test_enrich_with_ingress_never(self, db: Session, client: TestClient):
        """
        Expect no ingress to be created automatically as the configuration templated ingress mode is "never"
        """
        function = self._generate_runtime("nuclio")
        function_name, project_name, config = compile_function_config(function)
        service_type = "DoesNotMatter"
        enrich_function_with_ingress(
            config, NuclioIngressAddTemplatedIngressModes.never, service_type
        )
        ingresses = resolve_function_ingresses(config["spec"])
        assert ingresses == []

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

    def test_deploy_with_node_selection(self, db: Session, client: TestClient):
        function = self._generate_runtime("nuclio")

        node_name = "some-node-name"
        function.with_node_selection(node_name=node_name)

        deploy_nuclio_function(function)
        self._assert_deploy_called_basic_config()
        self._assert_node_selections(expected_node_name=node_name)

        function = self._generate_runtime("nuclio")

        node_selector = {
            "label-1": "val1",
            "label-2": "val2",
        }
        mlrun.mlconf.default_function_node_selector = base64.b64encode(
            json.dumps(node_selector).encode("utf-8")
        )
        function.with_node_selection(node_selector=node_selector)
        deploy_nuclio_function(function)
        self._assert_deploy_called_basic_config(call_count=2)
        self._assert_node_selections(expected_node_selector=node_selector)

        function = self._generate_runtime("nuclio")

        node_selector = {
            "label-3": "val3",
            "label-4": "val4",
        }
        function.with_node_selection(node_selector=node_selector)
        deploy_nuclio_function(function)
        self._assert_deploy_called_basic_config(call_count=3)
        self._assert_node_selections(expected_node_selector=node_selector)

        function = self._generate_runtime("nuclio")
        affinity = self._generate_affinity()
        function.with_node_selection(affinity=affinity)
        deploy_nuclio_function(function)
        self._assert_deploy_called_basic_config(call_count=4)
        self._assert_node_selections(expected_affinity=affinity)

        function = self._generate_runtime("nuclio")
        function.with_node_selection(node_name, node_selector, affinity)
        deploy_nuclio_function(function)
        self._assert_deploy_called_basic_config(call_count=5)
        self._assert_node_selections(
            expected_node_name=node_name,
            expected_node_selector=node_selector,
            expected_affinity=affinity,
        )

    def test_load_function_with_source_archive_git(self):
        fn = self._generate_runtime("nuclio")
        fn.with_source_archive(
            "git://github.com/org/repo#my-branch",
            handler="path/inside/repo#main:handler",
            secrets={"GIT_PASSWORD": "my-access-token"},
        )

        assert fn.spec.base_spec == {
            "apiVersion": "nuclio.io/v1",
            "kind": "Function",
            "metadata": {"name": "notebook", "labels": {}, "annotations": {}},
            "spec": {
                "runtime": "python:3.7",
                "handler": "main:handler",
                "env": [],
                "volumes": [],
                "build": {
                    "commands": [],
                    "noBaseImagesPull": True,
                    "path": "https://github.com/org/repo",
                    "codeEntryType": "git",
                    "codeEntryAttributes": {
                        "workDir": "path/inside/repo",
                        "reference": "refs/heads/my-branch",
                        "username": "",
                        "password": "my-access-token",
                    },
                },
            },
        }

    def test_load_function_with_source_archive_s3(self):
        fn = self._generate_runtime("nuclio")
        fn.with_source_archive(
            "s3://my-bucket/path/in/bucket/my-functions-archive",
            handler="path/inside/functions/archive#main:Handler",
            runtime="golang",
            secrets={
                "AWS_ACCESS_KEY_ID": "some-id",
                "AWS_SECRET_ACCESS_KEY": "some-secret",
            },
        )

        assert fn.spec.base_spec == {
            "apiVersion": "nuclio.io/v1",
            "kind": "Function",
            "metadata": {"name": "notebook", "labels": {}, "annotations": {}},
            "spec": {
                "runtime": "golang",
                "handler": "main:Handler",
                "env": [],
                "volumes": [],
                "build": {
                    "commands": [],
                    "noBaseImagesPull": True,
                    "path": "s3://my-bucket/path/in/bucket/my-functions-archive",
                    "codeEntryType": "s3",
                    "codeEntryAttributes": {
                        "workDir": "path/inside/functions/archive",
                        "s3Bucket": "my-bucket",
                        "s3ItemKey": "path/in/bucket/my-functions-archive",
                        "s3AccessKeyId": "some-id",
                        "s3SecretAccessKey": "some-secret",
                        "s3SessionToken": "",
                    },
                },
            },
        }

    def test_load_function_with_source_archive_v3io(self):
        fn = self._generate_runtime("nuclio")
        fn.with_source_archive(
            "v3ios://host.com/container/my-functions-archive.zip",
            handler="path/inside/functions/archive#main:handler",
            secrets={"V3IO_ACCESS_KEY": "ma-access-key"},
        )

        assert fn.spec.base_spec == {
            "apiVersion": "nuclio.io/v1",
            "kind": "Function",
            "metadata": {"name": "notebook", "labels": {}, "annotations": {}},
            "spec": {
                "runtime": "python:3.7",
                "handler": "main:handler",
                "env": [],
                "volumes": [],
                "build": {
                    "commands": [],
                    "noBaseImagesPull": True,
                    "path": "https://host.com/container/my-functions-archive.zip",
                    "codeEntryType": "archive",
                    "codeEntryAttributes": {
                        "workDir": "path/inside/functions/archive",
                        "headers": {"headers": {"X-V3io-Session-Key": "ma-access-key"}},
                    },
                },
            },
        }
