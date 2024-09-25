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
import base64
import json
import os
import typing
import unittest.mock

import deepdiff
import kubernetes
import nuclio
import nuclio.utils
import pytest
import requests
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.runtimes.nuclio.function
import mlrun.runtimes.pod
import server.api.crud.runtimes.nuclio.function
import server.api.crud.runtimes.nuclio.helpers
import server.api.utils.runtimes.nuclio
from mlrun import code_to_function, mlconf
from mlrun.common.runtimes.constants import NuclioIngressAddTemplatedIngressModes
from mlrun.platforms.iguazio import split_path
from mlrun.utils import logger
from server.api.utils.functions import build_function
from tests.api.conftest import K8sSecretsMock
from tests.api.runtimes.base import TestRuntimeBase


class TestNuclioRuntime(TestRuntimeBase):
    @property
    def runtime_kind(self):
        # enables extending classes to run the same tests with different runtime
        return "nuclio"

    @property
    def class_name(self):
        # enables extending classes to run the same tests with different class
        return "remote"

    def custom_setup_after_fixtures(self):
        self._mock_nuclio_deploy_config()

    def custom_setup(self):
        self.image_name = "test/image:latest"
        self.code_handler = "test_func"

        os.environ["V3IO_ACCESS_KEY"] = self.v3io_access_key = "1111-2222-3333-4444"
        os.environ["V3IO_USERNAME"] = self.v3io_user = "test-user"

    @staticmethod
    def _get_deployed_config():
        args, _ = nuclio.deploy.deploy_config.call_args
        return args[0]

    @staticmethod
    def _mock_nuclio_deploy_config():
        nuclio.deploy.deploy_config = unittest.mock.Mock(return_value="some-server")

    @staticmethod
    def _get_expected_struct_for_http_trigger(parameters):
        expected_struct = {
            "kind": "http",
            "name": "http",
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

    def _execute_run(self, runtime, **kwargs):
        # deploy_nuclio_function doesn't accept watch, so we need to remove it
        kwargs.pop("watch", None)
        server.api.crud.runtimes.nuclio.function.deploy_nuclio_function(
            runtime, **kwargs
        )

    def _generate_runtime(
        self, kind=None, labels=None
    ) -> typing.Union[mlrun.runtimes.RemoteRuntime, mlrun.runtimes.ServingRuntime]:
        runtime = code_to_function(
            name=self.name,
            project=self.project,
            filename=self.code_filename,
            handler=self.code_handler,
            kind=kind or self.runtime_kind,
            image=self.image_name,
            description="test function",
            labels=labels,
        )
        return runtime

    def _reset_mock(self):
        nuclio.deploy.deploy_config.reset_mock()

    def _assert_deploy_called_basic_config(
        self,
        expected_class="remote",
        call_count=1,
        expected_params=None,
        expected_labels=None,
        expected_env_from_secrets=None,
        expected_service_account=None,
        expected_build_base_image=None,
        expected_nuclio_runtime=None,
        expected_env=None,
        expected_build_commands=None,
        expected_build_args=None,
    ):
        if expected_labels is None:
            expected_labels = {}
        deploy_mock = nuclio.deploy.deploy_config
        assert deploy_mock.call_count == call_count

        deploy_configs = []

        call_args_list = deploy_mock.call_args_list
        for single_call_args in call_args_list:
            args, kwargs = single_call_args
            parent_function = None
            if expected_params:
                current_parameters = expected_params.pop(0)
                expected_function_name = current_parameters["function_name"]
                source_filename = current_parameters["file_name"]
                parent_function = current_parameters.get("parent_function")
            else:
                expected_function_name = f"{self.project}-{self.name}"
                source_filename = self.code_filename

            assert kwargs["name"] == expected_function_name
            assert kwargs["project"] == self.project

            deploy_config = args[0]
            deploy_configs.append(deploy_config)
            function_metadata = deploy_config["metadata"]
            assert function_metadata["name"] == expected_function_name
            labels_for_diff = expected_labels.copy()
            labels_for_diff.update(
                {mlrun_constants.MLRunInternalLabels.mlrun_class: expected_class}
            )
            if parent_function:
                labels_for_diff.update({"mlrun/parent-function": parent_function})
            assert deepdiff.DeepDiff(function_metadata["labels"], labels_for_diff) == {}

            build_info = deploy_config["spec"]["build"]

            # Nuclio source code in some cases adds a suffix to the code, initializing nuclio context.
            # We just verify that the code provided starts with our code.
            original_source_code = open(source_filename).read()
            spec_source_code = base64.b64decode(
                build_info["functionSourceCode"]
            ).decode("utf-8")
            assert spec_source_code.startswith(original_source_code)

            if self.image_name or expected_build_base_image:
                assert (
                    build_info["baseImage"] == self.image_name
                    or expected_build_base_image
                )

            if expected_env:
                env_vars = deploy_config["spec"]["env"]
                self._assert_pod_env(env_vars, expected_env)

            if expected_env_from_secrets:
                env_vars = deploy_config["spec"]["env"]
                self._assert_pod_env_from_secrets(env_vars, expected_env_from_secrets)

            if expected_service_account:
                assert (
                    deploy_config["spec"]["serviceAccount"] == expected_service_account
                )

            if expected_nuclio_runtime:
                assert deploy_config["spec"]["runtime"] == expected_nuclio_runtime

            if expected_build_commands:
                assert (
                    deploy_config["spec"]["build"]["commands"]
                    == expected_build_commands
                )
            if expected_build_args:
                assert deploy_config["spec"]["build"]["flags"] == expected_build_args

        return deploy_configs

    def _assert_http_trigger(self, http_trigger):
        args, _ = nuclio.deploy.deploy_config.call_args
        triggers_config = args[0]["spec"]["triggers"]

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

    def _assert_v3io_trigger(self, v3io_trigger):
        args, _ = nuclio.deploy.deploy_config.call_args
        triggers_config = args[0]["spec"]["triggers"]

        expected_struct = self._get_expected_struct_for_v3io_trigger(v3io_trigger)

        if mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
            "1.13.11"
        ):
            expected_struct["password"] = mlrun.model.Credentials.generate_access_key

        diff_result = deepdiff.DeepDiff(
            triggers_config[v3io_trigger["name"]],
            expected_struct,
            ignore_order=True,
        )
        # It's ok if the Nuclio trigger has additional parameters, these are constants that we don't care
        # about. We just care that the values we look for are fully there.
        diff_result.pop("dictionary_item_removed", None)
        assert diff_result == {}

    def _assert_nuclio_v3io_mount(self, local_path="", remote_path="", cred_only=False):
        args, _ = nuclio.deploy.deploy_config.call_args
        deploy_spec = args[0]["spec"]

        env_config = deploy_spec["env"]
        expected_env = {
            "V3IO_ACCESS_KEY": self.v3io_access_key,
            "V3IO_USERNAME": self.v3io_user,
            "V3IO_API": None,
            "MLRUN_NAMESPACE": self.namespace,
        }
        self._assert_pod_env(env_config, expected_env)
        if cred_only:
            assert len(deploy_spec["volumes"]) == 0
            return

        container, path = split_path(remote_path)

        expected_volume = {
            "volume": {
                "flexVolume": {
                    "driver": "v3io/fuse",
                    "options": {
                        "accessKey": self.v3io_access_key,
                        "container": container,
                        "subPath": path,
                        "dirsToCreate": f'[{{"name": "users//{self.v3io_user}", "permissions": 488}}]',
                    },
                },
                "name": "v3io",
            },
            "volumeMount": {"mountPath": local_path, "name": "v3io", "subPath": ""},
        }

        expected_cm_volume = {
            "volume": {
                "name": "serving-conf",
                "configMap": {"name": "serving-conf-test-project-test-function"},
            },
            "volumeMount": {
                "name": "serving-conf",
                "mountPath": "/tmp/mlrun/serving-conf",
                "readOnly": True,
            },
        }
        expected = (
            [expected_volume, expected_cm_volume]
            if self.runtime_kind == "serving"
            else [expected_volume]
        )

        assert (
            deepdiff.DeepDiff(
                deploy_spec["volumes"],
                expected,
                ignore_order=True,
            )
            == {}
        )

    def assert_node_selection(
        self,
        node_name=None,
        node_selector=None,
        affinity=None,
        tolerations=None,
    ):
        args, _ = nuclio.deploy.deploy_config.call_args
        deploy_spec = args[0]["spec"]

        if node_selector:
            assert deploy_spec.get("nodeSelector") == node_selector
        else:
            assert deploy_spec.get("nodeSelector") is None

        if node_name:
            assert deploy_spec.get("nodeName") == node_name
        else:
            assert deploy_spec.get("nodeName") is None

        if affinity:
            # deploy_spec returns affinity in CamelCase, V1Affinity is in snake_case
            assert (
                mlrun.runtimes.pod.transform_attribute_to_k8s_class_instance(
                    "affinity", deploy_spec.get("affinity")
                )
                == affinity
            )
        else:
            assert deploy_spec.get("affinity") is None

        if tolerations:
            # deploy_spec returns tolerations in CamelCase, [V1Toleration] is in snake_case
            assert (
                mlrun.runtimes.pod.transform_attribute_to_k8s_class_instance(
                    "tolerations", deploy_spec.get("tolerations")
                )
                == tolerations
            )
        else:
            assert deploy_spec.get("tolerations") is None

    def assert_security_context(
        self,
        security_context=None,
    ):
        args, _ = nuclio.deploy.deploy_config.call_args
        deploy_spec = args[0]["spec"]

        if security_context:
            assert (
                mlrun.runtimes.pod.transform_attribute_to_k8s_class_instance(
                    "security_context", deploy_spec.get("securityContext")
                )
                == security_context
            )
        else:
            assert deploy_spec.get("securityContext") is None

    def test_compile_function_config_with_special_character_labels(
        self, db: Session, client: TestClient
    ):
        """
        Test that compiling function configuration with labels containing special characters correctly sets them
        """
        function = self._generate_runtime(self.runtime_kind)
        key, val = "test.label.com/env", "test"
        function.set_label(key, val)
        (
            _,
            _,
            config,
        ) = server.api.crud.runtimes.nuclio.function._compile_function_config(function)
        assert config["metadata"]["labels"].get(key) == val

    def test_enrich_with_ingress_no_overriding(self, db: Session, client: TestClient):
        """
        Expect no ingress template to be created, thought its mode is "always",
        since the function already have a pre-configured ingress
        """
        function = self._generate_runtime(self.runtime_kind)

        # both ingress and node port
        ingress_host = "something.com"
        function.with_http(host=ingress_host, paths=["/"], port=30030)
        (
            function_name,
            project_name,
            config,
        ) = server.api.crud.runtimes.nuclio.function._compile_function_config(function)
        service_type = "NodePort"
        server.api.crud.runtimes.nuclio.helpers.enrich_function_with_ingress(
            config, NuclioIngressAddTemplatedIngressModes.always, service_type
        )
        ingresses = server.api.crud.runtimes.nuclio.helpers.resolve_function_ingresses(
            config["spec"]
        )
        assert len(ingresses) > 0, "Expected one ingress to be created"
        for ingress in ingresses:
            assert "hostTemplate" not in ingress, "No host template should be added"
            assert ingress["host"] == ingress_host

    def test_enrich_with_ingress_always(self, db: Session, client: TestClient):
        """
        Expect ingress template to be created as the configuration templated ingress mode is "always"
        """
        function = self._generate_runtime(self.runtime_kind)
        (
            function_name,
            project_name,
            config,
        ) = server.api.crud.runtimes.nuclio.function._compile_function_config(function)
        service_type = "NodePort"
        server.api.crud.runtimes.nuclio.helpers.enrich_function_with_ingress(
            config, NuclioIngressAddTemplatedIngressModes.always, service_type
        )
        ingresses = server.api.crud.runtimes.nuclio.helpers.resolve_function_ingresses(
            config["spec"]
        )
        assert ingresses[0]["hostTemplate"] != ""

    def test_enrich_with_ingress_on_cluster_ip(self, db: Session, client: TestClient):
        """
        Expect ingress template to be created as the configuration templated ingress mode is "onClusterIP" while the
        function service type is ClusterIP
        """
        function = self._generate_runtime(self.runtime_kind)
        (
            function_name,
            project_name,
            config,
        ) = server.api.crud.runtimes.nuclio.function._compile_function_config(function)
        service_type = "ClusterIP"
        server.api.crud.runtimes.nuclio.helpers.enrich_function_with_ingress(
            config,
            NuclioIngressAddTemplatedIngressModes.on_cluster_ip,
            service_type,
        )
        ingresses = server.api.crud.runtimes.nuclio.helpers.resolve_function_ingresses(
            config["spec"]
        )
        assert ingresses[0]["hostTemplate"] != ""

    def test_enrich_with_ingress_never(self, db: Session, client: TestClient):
        """
        Expect no ingress to be created automatically as the configuration templated ingress mode is "never"
        """
        function = self._generate_runtime(self.runtime_kind)
        (
            function_name,
            project_name,
            config,
        ) = server.api.crud.runtimes.nuclio.function._compile_function_config(function)
        service_type = "DoesNotMatter"
        server.api.crud.runtimes.nuclio.helpers.enrich_function_with_ingress(
            config, NuclioIngressAddTemplatedIngressModes.never, service_type
        )
        ingresses = server.api.crud.runtimes.nuclio.helpers.resolve_function_ingresses(
            config["spec"]
        )
        assert ingresses == []

    def test_nuclio_config_spec_env(self, db: Session, client: TestClient):
        function = self._generate_runtime(self.runtime_kind)

        name = "env1"
        secret = "shh"
        secret_key = "open sesame"
        function.set_env_from_secret(name, secret=secret, secret_key=secret_key)

        name2 = "env2"
        value2 = "value2"
        function.set_env(name2, value2)

        expected_env_vars = [
            {
                "name": name,
                "valueFrom": {"secretKeyRef": {"key": secret_key, "name": secret}},
            },
            {"name": name2, "value": value2},
        ]

        (
            function_name,
            project_name,
            config,
        ) = server.api.crud.runtimes.nuclio.function._compile_function_config(function)
        for expected_env_var in expected_env_vars:
            assert expected_env_var in config["spec"]["env"]
        env_var_names = []
        for envvar in function.spec.env:
            if isinstance(envvar, kubernetes.client.V1EnvVar):
                env_var_names.append(envvar.name)
        assert env_var_names == ["env1", "env2"]

        # simulating sending to API - serialization through dict
        function = function.from_dict(function.to_dict())
        (
            function_name,
            project_name,
            config,
        ) = server.api.crud.runtimes.nuclio.function._compile_function_config(function)
        for expected_env_var in expected_env_vars:
            assert expected_env_var in config["spec"]["env"]

    def test_deploy_with_project_secrets(
        self, db: Session, k8s_secrets_mock: K8sSecretsMock
    ):
        secret_keys = ["secret1", "secret2", "secret3"]
        secrets = {key: "some-secret-value" for key in secret_keys}

        k8s_secrets_mock.store_project_secrets(self.project, secrets)

        function = self._generate_runtime(self.runtime_kind)
        self.execute_function(function)

        # This test runs in serving, nuclio:mlrun as well, with different secret names encoding
        expected_secrets = k8s_secrets_mock.get_expected_env_variables_from_secrets(
            self.project, encode_key_names=(self.class_name != "remote")
        )
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name, expected_env_from_secrets=expected_secrets
        )

    def test_deploy_with_project_service_accounts(
        self, db: Session, k8s_secrets_mock: K8sSecretsMock
    ):
        k8s_secrets_mock.set_service_account_keys(self.project, "sa1", ["sa1", "sa2"])
        auth_info = mlrun.common.schemas.AuthInfo()
        function = self._generate_runtime(self.runtime_kind)
        # Need to call build_function, since service-account enrichment is happening only on server side, before the
        # call to deploy_nuclio_function
        build_function(db, auth_info, function)
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name, expected_service_account="sa1"
        )
        nuclio.deploy.deploy_config.reset_mock()

        function.spec.service_account = "bad-sa"
        with pytest.raises(HTTPException):
            build_function(db, auth_info, function)

        # verify that project SA overrides the global SA
        mlconf.function.spec.service_account.default = "some-other-sa"
        function.spec.service_account = "sa2"
        build_function(db, auth_info, function)
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name, expected_service_account="sa2"
        )
        mlconf.function.spec.service_account.default = None

    def test_deploy_with_security_context_enrichment(
        self, db: Session, k8s_secrets_mock: K8sSecretsMock
    ):
        user_unix_id = 1000
        auth_info = mlrun.common.schemas.AuthInfo(user_unix_id=user_unix_id)
        mlrun.mlconf.igz_version = "3.6"
        mlrun.mlconf.function.spec.security_context.enrichment_mode = (
            mlrun.common.schemas.function.SecurityContextEnrichmentModes.disabled.value
        )
        function = self._generate_runtime(self.runtime_kind)
        build_function(db, auth_info, function)
        self.assert_security_context({})

        mlrun.mlconf.function.spec.security_context.enrichment_mode = (
            mlrun.common.schemas.function.SecurityContextEnrichmentModes.override.value
        )
        function = self._generate_runtime(self.runtime_kind)
        build_function(db, auth_info, function)
        self.assert_security_context(
            self._generate_security_context(
                run_as_group=mlrun.mlconf.function.spec.security_context.enrichment_group_id,
                run_as_user=user_unix_id,
            )
        )

    def test_deploy_mlrun_requirements(
        self, db: Session, k8s_secrets_mock: K8sSecretsMock
    ):
        auth_info = mlrun.common.schemas.AuthInfo()
        mlrun.mlconf.function.spec.security_context.enrichment_mode = (
            mlrun.common.schemas.function.SecurityContextEnrichmentModes.disabled.value
        )
        function = self._generate_runtime(self.runtime_kind)
        mlrun.utils.update_in(
            function.spec.config,
            "spec.build.baseImage",
            "mlrun/mlrun:0.6.0",
        )
        function.spec.build.requirements = ["some-requirements"]
        build_function(db, auth_info, function)
        assert "mlrun[complete]==0.6.0" in function.spec.build.requirements

    def test_deploy_with_global_service_account(
        self, db: Session, k8s_secrets_mock: K8sSecretsMock
    ):
        service_account_name = "default-sa"
        mlconf.function.spec.service_account.default = service_account_name
        auth_info = mlrun.common.schemas.AuthInfo()
        function = self._generate_runtime(self.runtime_kind)
        # Need to call build_function, since service-account enrichment is happening only on server side, before the
        # call to deploy_nuclio_function
        build_function(db, auth_info, function)
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name,
            expected_service_account=service_account_name,
        )
        mlconf.function.spec.service_account.default = None

    def test_deploy_basic_function(self, db: Session, client: TestClient):
        function = self._generate_runtime(self.runtime_kind)

        self.execute_function(function)
        self._assert_deploy_called_basic_config(expected_class=self.class_name)

    def test_deploy_build_base_image(
        self, db: Session, k8s_secrets_mock: K8sSecretsMock
    ):
        expected_build_base_image = "mlrun/base_mlrun:latest"
        self.image_name = None

        function = self._generate_runtime(self.runtime_kind)
        function.spec.build.base_image = expected_build_base_image

        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name,
            expected_build_base_image=expected_build_base_image,
        )

    def test_deploy_populate_nuclio_errors(
        self, db: Session, k8s_secrets_mock: K8sSecretsMock
    ):
        function = self._generate_runtime(self.runtime_kind)

        # simulate a nuclio deploy error
        response = requests.Response()
        response._content = (
            b'{"error": "Something bad happened - custom message from nuclio"}'
        )
        response.reason = "Bad Request"
        response.status_code = 400

        nuclio.deploy.deploy_config.side_effect = [
            nuclio.utils.DeployError("Deployment failed", response)
        ]
        with pytest.raises(mlrun.errors.MLRunBadRequestError) as exc:
            self.execute_function(function)
        assert "custom message from nuclio" in str(exc.value)

    def test_deploy_image_name_and_build_base_image(
        self, db: Session, k8s_secrets_mock: K8sSecretsMock
    ):
        """When spec.image and also spec.build.base_image are both defined the spec.image should be applied
        to spec.baseImage in nuclio."""

        function = self._generate_runtime(self.runtime_kind)
        function.spec.build.base_image = "mlrun/base_mlrun:latest"

        self.execute_function(function)
        self._assert_deploy_called_basic_config(expected_class=self.class_name)

    def test_deploy_without_image_and_build_base_image(
        self, db: Session, k8s_secrets_mock: K8sSecretsMock
    ):
        self.image_name = None

        function = self._generate_runtime(self.runtime_kind)
        self.execute_function(function)

        self._assert_deploy_called_basic_config(expected_class=self.class_name)

    @pytest.mark.parametrize(
        "extra_args,expected_build_flags",
        [
            ("--skip-tls-verify --cleanup", ["--skip-tls-verify", "--cleanup"]),
            ("--skip-tls-verify    --cleanup", ["--skip-tls-verify", "--cleanup"]),
            (
                "--skip-tls-verify  --build-arg LABEL=SL --cleanup --memory=100",
                [
                    "--skip-tls-verify",
                    "--build-arg LABEL=SL",
                    "--cleanup",
                    "--memory=100",
                ],
            ),
        ],
    )
    def test_deploy_with_build_flags(
        self,
        extra_args: str,
        expected_build_flags: list,
        db: Session,
        client: TestClient,
    ):
        function = self._generate_runtime(self.runtime_kind)
        function.spec.build.extra_args = extra_args
        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name, expected_build_args=expected_build_flags
        )

    def test_deploy_image_with_enrich_registry_prefix(self):
        function = self._generate_runtime(self.runtime_kind)
        function.spec.image = ".my/image:latest"

        with unittest.mock.patch(
            "mlrun.utils.get_parsed_docker_registry",
            return_value=["some.registry", "some-repository"],
        ):
            self.execute_function(function)
            self._assert_deploy_called_basic_config(
                expected_class=self.class_name,
                expected_build_base_image="some.registry/some-repository/my/image:latest",
            )

    @pytest.mark.parametrize(
        "requirements,expected_commands",
        [
            (["pandas", "numpy"], ["python -m pip install pandas numpy"]),
            (
                ["-r requirements.txt", "numpy"],
                ["python -m pip install -r requirements.txt numpy"],
            ),
            (["pandas>=1.0.0, <2"], ["python -m pip install 'pandas>=1.0.0, <2'"]),
            (["pandas>=1.0.0,<2"], ["python -m pip install 'pandas>=1.0.0,<2'"]),
            (
                ["-r somewhere/requirements.txt"],
                ["python -m pip install -r somewhere/requirements.txt"],
            ),
            (
                ["something @ git+https://somewhere.com/a/b.git@v0.0.0#egg=something"],
                [
                    "python -m pip install 'something @ git+https://somewhere.com/a/b.git@v0.0.0#egg=something'"
                ],
            ),
        ],
    )
    def test_deploy_function_with_requirements(
        self,
        requirements: list,
        expected_commands: list,
        db: Session,
        client: TestClient,
    ):
        function = self._generate_runtime(self.runtime_kind)
        function.with_requirements(requirements)
        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name, expected_build_commands=expected_commands
        )

    def test_deploy_function_with_commands_and_requirements(
        self, db: Session, client: TestClient
    ):
        function = self._generate_runtime(self.runtime_kind)
        function.with_commands(["python -m pip install scikit-learn"])
        function.with_requirements(["pandas", "numpy"])
        self.execute_function(function)
        expected_commands = [
            "python -m pip install scikit-learn",
            "python -m pip install pandas numpy",
        ]
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name, expected_build_commands=expected_commands
        )

    def test_deploy_function_with_labels(self, db: Session, client: TestClient):
        labels = {
            "key": "value",
            "key-2": "value-2",
        }
        function = self._generate_runtime(self.runtime_kind, labels)

        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            expected_labels=labels, expected_class=self.class_name
        )

    @pytest.mark.parametrize(
        "nuclio_version",
        ["1.12.1", "1.13.1", "1.13.11", "1.14.3"],
    )
    def test_deploy_with_triggers(
        self, db: Session, client: TestClient, nuclio_version
    ):
        mlconf.nuclio_version = nuclio_version

        function = self._generate_runtime(self.runtime_kind)

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

        self.execute_function(function)
        self._assert_deploy_called_basic_config(expected_class=self.class_name)
        self._assert_http_trigger(http_trigger)
        self._assert_v3io_trigger(v3io_trigger)

    def test_deploy_with_v3io(self, db: Session, client: TestClient):
        function = self._generate_runtime(self.runtime_kind)
        local_path = "/local/path"
        remote_path = "/container/and/path"
        function.with_v3io(local_path, remote_path)

        self.execute_function(function)
        self._assert_deploy_called_basic_config(expected_class=self.class_name)
        self._assert_nuclio_v3io_mount(local_path, remote_path)

    def test_deploy_with_node_selection(self, db: Session, client: TestClient):
        function = self._generate_runtime(self.runtime_kind)
        node_name = "some-node-name"
        mlconf.nuclio_version = "1.6.3"
        with pytest.raises(mlrun.errors.MLRunIncompatibleVersionError):
            function.with_node_selection(node_name=node_name)

        mlconf.nuclio_version = "1.5.21"
        function.with_node_selection(node_name=node_name)

        self.execute_function(function)
        self._assert_deploy_called_basic_config(expected_class=self.class_name)
        self.assert_node_selection(node_name=node_name)

        function = self._generate_runtime(self.runtime_kind)

        mlconf.nuclio_version = "1.6.10"
        config_node_selector = {
            "label-1": "val1",
            "label-2": "val2",
        }
        mlconf.default_function_node_selector = base64.b64encode(
            json.dumps(config_node_selector).encode("utf-8")
        )
        function.with_node_selection(node_selector=config_node_selector)
        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            call_count=2, expected_class=self.class_name
        )
        self.assert_node_selection(node_selector=config_node_selector)

        function = self._generate_runtime(self.runtime_kind)

        invalid_node_selector = {"label-3": "val=3"}
        with pytest.warns(
            Warning,
            match="The node selector you’ve set does not meet the validation rules for the current Kubernetes version",
        ):
            function.with_node_selection(node_selector=invalid_node_selector)

        node_selector = {
            "label-3": "val3",
            "label-4": "val4",
        }
        function.with_node_selection(node_selector=node_selector)
        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            call_count=3, expected_class=self.class_name
        )
        self.assert_node_selection(
            node_selector={**config_node_selector, **node_selector}
        )

        function = self._generate_runtime(self.runtime_kind)
        affinity = self._generate_affinity()

        function.with_node_selection(affinity=affinity)
        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            call_count=4, expected_class=self.class_name
        )
        # The node selector is specific to the service configuration, not the function itself.
        # It is applied only to the run object on other run kinds. In case of a Nuclio function,
        # since there is no run object, the node selector is included in the created config.
        self.assert_node_selection(
            affinity=affinity, node_selector=config_node_selector
        )

        function = self._generate_runtime(self.runtime_kind)
        function.with_node_selection(node_name, node_selector, affinity)
        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            call_count=5, expected_class=self.class_name
        )
        self.assert_node_selection(
            node_name=node_name,
            node_selector={**config_node_selector, **node_selector},
            affinity=affinity,
        )

        tolerations = self._generate_tolerations()
        function = self._generate_runtime(self.runtime_kind)
        with pytest.raises(mlrun.errors.MLRunIncompatibleVersionError):
            function.with_node_selection(tolerations=tolerations)

        mlconf.nuclio_version = "1.7.6"
        function = self._generate_runtime(self.runtime_kind)
        function.with_node_selection(tolerations=tolerations)
        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            call_count=6, expected_class=self.class_name
        )
        self.assert_node_selection(
            tolerations=tolerations,
            node_selector=config_node_selector,
        )

    @pytest.mark.parametrize(
        "config_node_selector, project_node_selector",
        [({}, {}), ({"kubernetes.io/arch": "amd64"}, {"kubernetes.io/os": "linux"})],
    )
    def test_compile_function_config_node_selector_enriched_from_project(
        self,
        db: Session,
        client: TestClient,
        project_node_selector,
        config_node_selector,
    ):
        config_node_selector = config_node_selector
        mlconf.default_function_node_selector = base64.b64encode(
            json.dumps(config_node_selector).encode("utf-8")
        )

        run_db = mlrun.get_run_db()
        project = run_db.get_project(self.project)
        project.spec.default_function_node_selector = project_node_selector
        run_db.store_project(self.project, project)

        function = self._generate_runtime(self.runtime_kind)
        function_node_selector = {"kubernetes.io/hostname": "k8s-node1"}
        function.spec.node_selector = function_node_selector

        (
            _,
            _,
            config,
        ) = server.api.crud.runtimes.nuclio.function._compile_function_config(function)
        assert config["spec"]["nodeSelector"] == {
            **config_node_selector,
            **project.spec.default_function_node_selector,
            **function_node_selector,
        }

    def test_deploy_with_priority_class_name(self, db: Session, client: TestClient):
        mlconf.nuclio_version = "1.5.20"
        default_priority_class_name = "default-priority"
        mlrun.mlconf.default_function_priority_class_name = default_priority_class_name
        mlrun.mlconf.valid_function_priority_class_names = default_priority_class_name
        function = self._generate_runtime(self.runtime_kind)

        self.execute_function(function)
        self._assert_deploy_called_basic_config(expected_class=self.class_name)
        args, _ = nuclio.deploy.deploy_config.call_args
        deploy_spec = args[0]["spec"]

        assert "priorityClassName" not in deploy_spec

        mlconf.nuclio_version = "1.6.18"
        mlrun.mlconf.valid_function_priority_class_names = ""
        function = self._generate_runtime(self.runtime_kind)

        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            call_count=2, expected_class=self.class_name
        )
        args, _ = nuclio.deploy.deploy_config.call_args
        deploy_spec = args[0]["spec"]

        assert "priorityClassName" not in deploy_spec

        mlrun.mlconf.valid_function_priority_class_names = default_priority_class_name
        function = self._generate_runtime(self.runtime_kind)

        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            call_count=3, expected_class=self.class_name
        )
        args, _ = nuclio.deploy.deploy_config.call_args
        deploy_spec = args[0]["spec"]

        assert deploy_spec["priorityClassName"] == default_priority_class_name

        function = self._generate_runtime(self.runtime_kind)
        medium_priority_class_name = "medium-priority"
        mlrun.mlconf.valid_function_priority_class_names = medium_priority_class_name
        mlconf.nuclio_version = "1.5.20"
        with pytest.raises(mlrun.errors.MLRunIncompatibleVersionError):
            function.with_priority_class(medium_priority_class_name)

        mlconf.nuclio_version = "1.6.10"
        with pytest.raises(mlrun.errors.MLRunIncompatibleVersionError):
            function.with_priority_class(medium_priority_class_name)

        mlconf.nuclio_version = "1.6.18"
        function.with_priority_class(medium_priority_class_name)

        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            call_count=4, expected_class=self.class_name
        )
        args, _ = nuclio.deploy.deploy_config.call_args
        deploy_spec = args[0]["spec"]

        assert deploy_spec["priorityClassName"] == medium_priority_class_name

    def test_set_metadata_annotations(self, db: Session, client: TestClient):
        function = self._generate_runtime(self.runtime_kind)
        function.with_annotations({"annotation-key": "annotation-value"})

        self.execute_function(function)
        args, _ = nuclio.deploy.deploy_config.call_args
        deploy_metadata = args[0]["metadata"]

        if deploy_metadata.get("annotations"):
            assert (
                deploy_metadata["annotations"].get("annotation-key")
                == "annotation-value"
            )

    @pytest.mark.parametrize(
        "client_version,client_python_version,nuclio_version,expected_nuclio_runtime",
        [
            ("1.2.0", None, "1.5.9", "python:3.6"),
            ("1.2.0", None, "1.9.15", mlrun.mlconf.default_nuclio_runtime),
            (None, None, "1.5.9", "python:3.6"),
            (None, None, "1.9.15", mlrun.mlconf.default_nuclio_runtime),
            ("1.3.0", "3.7", "1.11.9", "python:3.7"),
            ("1.3.0", "3.9", "1.11.9", "python:3.9"),
            ("1.3.0", "3.9", "1.5.9", "python:3.6"),
            ("1.3.0-rc1", "3.9", "1.11.9", "python:3.9"),
            ("1.3.0-rc1", "3.7", "1.11.9", "python:3.7"),
            ("0.0.0-unstable", "3.7", "1.11.9", "python:3.7"),
            ("0.0.0-unstable", "3.9", "1.11.9", "python:3.9"),
        ],
    )
    def test_deploy_with_runtime(
        self,
        db: Session,
        client: TestClient,
        client_version,
        client_python_version,
        nuclio_version,
        expected_nuclio_runtime,
    ):
        mlconf.nuclio_version = nuclio_version
        function = self._generate_runtime(self.runtime_kind)
        self.execute_function(
            function,
            client_version=client_version,
            client_python_version=client_python_version,
        )
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name,
            expected_nuclio_runtime=expected_nuclio_runtime,
        )

    def test_deploy_python_decode_string_env_var_enrichment(
        self, db: Session, client: TestClient
    ):
        mlconf.default_nuclio_runtime = "python:3.7"
        decode_event_strings_env_var_name = "NUCLIO_PYTHON_DECODE_EVENT_STRINGS"

        logger.info("Function runtime is golang - do nothing")
        function = self._generate_runtime(self.runtime_kind)
        function.spec.nuclio_runtime = "golang"
        self.execute_function(function)
        deploy_configs = self._assert_deploy_called_basic_config(
            expected_class=self.class_name,
            expected_nuclio_runtime=function.spec.nuclio_runtime,
        )
        assert decode_event_strings_env_var_name not in deploy_configs[0]["spec"]["env"]

        logger.info(
            "Function runtime is configured to python:3.7, nuclio version <1.6.0 - explode"
        )
        function = self._generate_runtime(self.runtime_kind)
        function.spec.nuclio_runtime = "python:3.7"
        mlconf.nuclio_version = "1.5.13"
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match=r"(.*)Nuclio version does not support(.*)",
        ):
            self.execute_function(function)

        logger.info(
            "Function runtime is default to python:3.7, nuclio is <1.6.0 - change to 3.6"
        )
        self._reset_mock()
        function = self._generate_runtime(self.runtime_kind)
        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name,
            expected_nuclio_runtime="python:3.6",
        )
        assert decode_event_strings_env_var_name not in deploy_configs[0]["spec"]["env"]

        logger.info("Function runtime is python, but nuclio is >=1.8.0 - do nothing")
        self._reset_mock()
        mlconf.nuclio_version = "1.8.5"
        function = self._generate_runtime(self.runtime_kind)
        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name,
            expected_nuclio_runtime=mlconf.default_nuclio_runtime,
        )
        assert decode_event_strings_env_var_name not in deploy_configs[0]["spec"]["env"]

        logger.info(
            "Function runtime is python, nuclio version in range, but already has the env var set - do nothing"
        )
        self._reset_mock()
        mlconf.nuclio_version = "1.7.5"
        function = self._generate_runtime(self.runtime_kind)
        function.set_env(decode_event_strings_env_var_name, "false")
        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name,
            expected_nuclio_runtime=mlconf.default_nuclio_runtime,
            expected_env={decode_event_strings_env_var_name: "false"},
        )

        logger.info(
            "Function runtime is python, nuclio version in range, env var not set - add it"
        )
        self._reset_mock()
        mlconf.nuclio_version = "1.7.5"
        function = self._generate_runtime(self.runtime_kind)
        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name,
            expected_nuclio_runtime=mlconf.default_nuclio_runtime,
            expected_env={decode_event_strings_env_var_name: "true"},
        )

    def test_is_nuclio_version_in_range(self):
        mlconf.nuclio_version = "1.7.2"

        assert not server.api.crud.runtimes.nuclio.helpers.is_nuclio_version_in_range(
            "1.6.11", "1.7.2"
        )
        assert not server.api.crud.runtimes.nuclio.helpers.is_nuclio_version_in_range(
            "1.7.0", "1.3.1"
        )
        assert not server.api.crud.runtimes.nuclio.helpers.is_nuclio_version_in_range(
            "1.7.3", "1.8.5"
        )
        assert not server.api.crud.runtimes.nuclio.helpers.is_nuclio_version_in_range(
            "1.7.2", "1.7.2"
        )
        assert server.api.crud.runtimes.nuclio.helpers.is_nuclio_version_in_range(
            "1.7.2", "1.7.3"
        )
        assert server.api.crud.runtimes.nuclio.helpers.is_nuclio_version_in_range(
            "1.7.0", "1.7.3"
        )
        assert server.api.crud.runtimes.nuclio.helpers.is_nuclio_version_in_range(
            "1.5.5", "1.7.3"
        )
        assert server.api.crud.runtimes.nuclio.helpers.is_nuclio_version_in_range(
            "1.5.5", "2.3.4"
        )

        # best effort - assumes compatibility
        mlconf.nuclio_version = ""
        assert server.api.crud.runtimes.nuclio.helpers.is_nuclio_version_in_range(
            "1.5.5", "2.3.4"
        )
        assert server.api.crud.runtimes.nuclio.helpers.is_nuclio_version_in_range(
            "1.7.2", "1.7.2"
        )

    def test_validate_nuclio_version_compatibility(self):
        # nuclio version we have
        mlconf.nuclio_version = "1.6.10"

        # mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility receives the min nuclio version required
        assert not mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
            "1.6.11"
        )
        assert not mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
            "1.5.9", "1.6.11"
        )
        assert not mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
            "1.6.11", "1.5.9"
        )
        assert mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
            "1.6.9", "1.7.0"
        )
        assert not mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
            "2.0.0"
        )
        assert mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
            "1.6.9"
        )
        assert mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
            "1.5.9"
        )

        mlconf.nuclio_version = "2.0.0"
        assert mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
            "1.6.11"
        )
        assert mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
            "1.5.9", "1.6.11"
        )

        # best effort - assumes compatibility
        mlconf.nuclio_version = ""
        assert mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
            "1.6.11"
        )
        assert mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility(
            "1.5.9", "1.6.11"
        )

        with pytest.raises(ValueError):
            mlrun.runtimes.nuclio.function.validate_nuclio_version_compatibility("")

    def test_min_nuclio_versions_decorator_failure(self):
        mlconf.nuclio_version = "1.6.10"

        for case in [
            ["1.6.11"],
            ["2.6.11"],
            ["1.5.9", "1.6.11"],
        ]:

            @mlrun.runtimes.nuclio.function.min_nuclio_versions(*case)
            def fail():
                pytest.fail("Should not enter this function")

            with pytest.raises(mlrun.errors.MLRunIncompatibleVersionError):
                fail()

    def test_min_nuclio_versions_decorator_success(self):
        for nuclio_version in ["1.6.10", "2.2.1", "", "Gibberish"]:
            mlconf.nuclio_version = nuclio_version

            for case in [
                ["1.6.9"],
                ["1.5.9", "1.6.9"],
                ["1.0.0", "0.9.81", "1.4.1"],
            ]:

                @mlrun.runtimes.nuclio.function.min_nuclio_versions(*case)
                def success():
                    pass

                success()

    def test_load_function_with_source_archive_git(self):
        fn = self._generate_runtime(self.runtime_kind)
        fn.with_source_archive(
            "git://github.com/org/repo#my-branch",
            handler="main:handler",
            workdir="path/inside/repo",
        )
        secrets = {"GIT_PASSWORD": "my-access-token"}

        get_archive_spec(fn, secrets)
        assert get_archive_spec(fn, secrets) == {
            "spec": {
                "handler": "main:handler",
                "build": {
                    "path": "https://github.com/org/repo",
                    "codeEntryType": "git",
                    "codeEntryAttributes": {
                        "workDir": "path/inside/repo",
                        "branch": "my-branch",
                        "username": "",
                        "password": "my-access-token",
                    },
                },
            },
        }

        fn = self._generate_runtime(self.runtime_kind)
        fn.with_source_archive(
            "git://github.com/org/repo#refs/heads/my-branch",
            handler="main:handler",
            workdir="path/inside/repo",
        )

        assert get_archive_spec(fn, secrets) == {
            "spec": {
                "handler": "main:handler",
                "build": {
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

    def test_nuclio_run_without_specifying_resources(
        self, db: Session, client: TestClient
    ):
        self.assert_run_without_specifying_resources()

    def test_load_function_with_source_archive_s3(self):
        fn = self._generate_runtime(self.runtime_kind)
        fn.with_source_archive(
            "s3://my-bucket/path/in/bucket/my-functions-archive.tar.gz",
            handler="main:Handler",
            workdir="path/inside/functions/archive",
            runtime="golang",
        )
        secrets = {
            "AWS_ACCESS_KEY_ID": "some-id",
            "AWS_SECRET_ACCESS_KEY": "some-secret",
        }

        assert fn.spec.nuclio_runtime == "golang"
        assert get_archive_spec(fn, secrets) == {
            "spec": {
                "handler": "main:Handler",
                "build": {
                    "path": "s3://my-bucket/path/in/bucket/my-functions-archive.tar.gz",
                    "codeEntryType": "s3",
                    "codeEntryAttributes": {
                        "workDir": "path/inside/functions/archive",
                        "s3Bucket": "my-bucket",
                        "s3ItemKey": "path/in/bucket/my-functions-archive.tar.gz",
                        "s3AccessKeyId": "some-id",
                        "s3SecretAccessKey": "some-secret",
                        "s3SessionToken": "",
                    },
                },
            },
        }

    def test_load_function_with_source_archive_v3io(self):
        fn = self._generate_runtime(self.runtime_kind)
        fn.with_source_archive(
            "v3ios://host.com/container/my-functions-archive.zip",
            handler="main:handler",
            workdir="path/inside/functions/archive",
        )
        secrets = {"V3IO_ACCESS_KEY": "ma-access-key"}

        assert get_archive_spec(fn, secrets) == {
            "spec": {
                "handler": "main:handler",
                "build": {
                    "path": "https://host.com/container/my-functions-archive.zip",
                    "codeEntryType": "archive",
                    "codeEntryAttributes": {
                        "workDir": "path/inside/functions/archive",
                        "headers": {"X-V3io-Session-Key": "ma-access-key"},
                    },
                },
            },
        }

    @pytest.mark.parametrize(
        "image_pull_secret_name,build_secret_name,default_image_pull_secret_name,"
        "default_build_secret_name,expected_secret_name",
        [
            ("", "", "", "", None),
            ("my-secret", "", "", "", "my-secret"),
            ("my-secret", None, "", "", "my-secret"),
            ("my-secret", None, None, None, "my-secret"),
            ("my-secret", "my-secret", "", "", "my-secret"),
            (None, "my-secret", "", "", "my-secret"),
            (None, "my-secret", None, None, "my-secret"),
            ("my-image-pull-secret", "my-build-secret", "", "", "my-image-pull-secret"),
            (
                None,
                None,
                "my-default-image-pull-secret",
                "",
                "my-default-image-pull-secret",
            ),
            (None, None, "", "my-default-builder-secret", "my-default-builder-secret"),
            (
                None,
                None,
                "my-default-image-pull-secret",
                "my-default-builder-secret",
                "my-default-image-pull-secret",
            ),
            (
                "my-other-image-pull-secret",
                None,
                "my-default-image-pull-secret",
                "",
                "my-other-image-pull-secret",
            ),
            (
                None,
                "my-other-builder-secret",
                "",
                "my-default-builder-secret",
                "my-other-builder-secret",
            ),
            (
                "my-other-image-pull-secret",
                "my-other-builder-secret",
                "",
                "my-default-builder-secret",
                "my-other-image-pull-secret",
            ),
            (
                "my-other-image-pull-secret",
                "my-other-builder-secret",
                "my-default-image-pull-secret",
                "my-default-builder-secret",
                "my-other-image-pull-secret",
            ),
            (
                "my-default-image-pull-secret",
                "my-other-builder-secret",
                "my-default-image-pull-secret",
                "my-default-builder-secret",
                "my-other-builder-secret",
            ),
            (
                "my-default-image-pull-secret",
                "my-default-builder-secret",
                "my-default-image-pull-secret",
                "my-default-builder-secret",
                "my-default-image-pull-secret",
            ),
            (
                None,
                "my-other-builder-secret",
                "my-default-image-pull-secret",
                "my-default-builder-secret",
                "my-other-builder-secret",
            ),
            (
                "",
                "my-other-builder-secret",
                "my-default-image-pull-secret",
                "my-default-builder-secret",
                None,
            ),
            (
                "",
                "",
                "my-default-image-pull-secret",
                "my-default-builder-secret",
                None,
            ),
            (
                "my-default-image-pull-secret",
                "",
                "my-default-image-pull-secret",
                "my-default-builder-secret",
                None,
            ),
        ],
    )
    def test_deploy_function_with_image_pull_secret(
        self,
        image_pull_secret_name,
        build_secret_name,
        default_image_pull_secret_name,
        default_build_secret_name,
        expected_secret_name,
    ):
        mlrun.mlconf.function.spec.image_pull_secret.default = (
            default_image_pull_secret_name
        )
        mlrun.mlconf.httpdb.builder.docker_registry_secret = default_build_secret_name
        fn = self._generate_runtime()

        if image_pull_secret_name is not None:
            fn.set_image_pull_configuration(
                image_pull_secret_name=image_pull_secret_name
            )

        if build_secret_name is not None:
            fn.spec.build.secret = build_secret_name

        (
            _,
            _,
            deployed_config,
        ) = server.api.crud.runtimes.nuclio.function._compile_function_config(fn)
        assert deployed_config["spec"].get("imagePullSecrets") == expected_secret_name

    def test_nuclio_with_preemption_mode(self):
        fn = self._generate_runtime(self.runtime_kind)
        assert fn.spec.preemption_mode == "prevent"
        fn.with_preemption_mode(mlrun.common.schemas.PreemptionModes.allow.value)
        assert fn.spec.preemption_mode == "allow"
        fn.with_preemption_mode(mlrun.common.schemas.PreemptionModes.constrain.value)
        assert fn.spec.preemption_mode == "constrain"

        fn.with_preemption_mode(mlrun.common.schemas.PreemptionModes.allow.value)
        assert fn.spec.preemption_mode == "allow"

        mlconf.nuclio_version = "1.7.5"
        with pytest.raises(mlrun.errors.MLRunIncompatibleVersionError):
            fn.with_preemption_mode(mlrun.common.schemas.PreemptionModes.allow.value)

        mlconf.nuclio_version = "1.8.6"
        fn.with_preemption_mode(mlrun.common.schemas.PreemptionModes.allow.value)
        assert fn.spec.preemption_mode == "allow"

    def test_preemption_mode_without_preemptible_configuration(
        self, db: Session, client: TestClient
    ):
        self.assert_run_with_preemption_mode_without_preemptible_configuration()

    def test_preemption_mode_with_preemptible_node_selector_without_tolerations(
        self, db: Session, client: TestClient
    ):
        self.assert_run_preemption_mode_with_preemptible_node_selector_without_preemptible_tolerations()

    def test_preemption_mode_with_preemptible_node_selector_and_tolerations(
        self, db: Session, client: TestClient
    ):
        self.assert_run_preemption_mode_with_preemptible_node_selector_and_tolerations()

    def test_preemption_mode_with_preemptible_node_selector_and_tolerations_with_extra_settings(
        self, db: Session, client: TestClient
    ):
        self.assert_run_preemption_mode_with_preemptible_node_selector_and_tolerations_with_extra_settings()

    def test_with_preemption_mode_none_transitions(
        self, db: Session, client: TestClient
    ):
        self.assert_run_with_preemption_mode_none_transitions()

    def test_preemption_mode_with_preemptible_node_selector_without_preemptible_tolerations_with_extra_settings(
        self, db: Session, client: TestClient
    ):
        self.assert_run_preemption_mode_with_preemptible_node_selector_without_preemptible_tolerations_with_extra_settings()  # noqa: E501

    def test_deploy_with_security_context(self, db: Session, client: TestClient):
        function = self._generate_runtime(self.runtime_kind)

        self.execute_function(function)
        self._assert_deploy_called_basic_config(expected_class=self.class_name)
        self.assert_security_context()

        default_security_context_dict = {
            "runAsUser": 1000,
            "runAsGroup": 3000,
        }
        mlrun.mlconf.function.spec.security_context.default = base64.b64encode(
            json.dumps(default_security_context_dict).encode("utf-8")
        )
        default_security_context = self._generate_security_context(
            default_security_context_dict["runAsUser"],
            default_security_context_dict["runAsGroup"],
        )
        function = self._generate_runtime(self.runtime_kind)
        self.execute_function(function)

        self._assert_deploy_called_basic_config(
            call_count=2, expected_class=self.class_name
        )
        self.assert_security_context(default_security_context)

        function = self._generate_runtime(self.runtime_kind)
        other_security_context = self._generate_security_context(
            2000,
            2000,
        )

        function.with_security_context(other_security_context)
        self.execute_function(function)

        self._assert_deploy_called_basic_config(
            call_count=3, expected_class=self.class_name
        )
        self.assert_security_context(other_security_context)

    @pytest.mark.parametrize(
        "service_type, default_service_type, expected_service_type, "
        "add_templated_ingress_host_mode, default_add_templated_ingress_host_mode, expected_ingress_host_template",
        [
            (
                "NodePort",
                "ClusterIP",
                "NodePort",
                NuclioIngressAddTemplatedIngressModes.never,
                NuclioIngressAddTemplatedIngressModes.always,
                None,
            ),
            (
                "NodePort",
                "ClusterIP",
                "NodePort",
                NuclioIngressAddTemplatedIngressModes.always,
                NuclioIngressAddTemplatedIngressModes.never,
                "@nuclio.fromDefault",
            ),
            (
                "",
                "ClusterIP",
                "ClusterIP",
                NuclioIngressAddTemplatedIngressModes.never,
                NuclioIngressAddTemplatedIngressModes.always,
                None,
            ),
            (
                "NodePort",
                "ClusterIP",
                "NodePort",
                "",
                NuclioIngressAddTemplatedIngressModes.on_cluster_ip,
                None,
            ),
            (
                "ClusterIP",
                "NodePort",
                "ClusterIP",
                "",
                NuclioIngressAddTemplatedIngressModes.on_cluster_ip,
                "@nuclio.fromDefault",
            ),
            (
                "ClusterIP",
                "NodePort",
                "ClusterIP",
                NuclioIngressAddTemplatedIngressModes.never,
                NuclioIngressAddTemplatedIngressModes.on_cluster_ip,
                None,
            ),
            (
                "ClusterIP",
                "NodePort",
                "ClusterIP",
                NuclioIngressAddTemplatedIngressModes.on_cluster_ip,
                NuclioIngressAddTemplatedIngressModes.never,
                "@nuclio.fromDefault",
            ),
        ],
    )
    def test_deploy_with_service_type(
        self,
        db: Session,
        client: TestClient,
        service_type,
        default_service_type,
        expected_service_type,
        add_templated_ingress_host_mode,
        default_add_templated_ingress_host_mode,
        expected_ingress_host_template,
    ):
        mlconf.httpdb.nuclio.default_service_type = default_service_type
        mlconf.httpdb.nuclio.add_templated_ingress_host_mode = (
            default_add_templated_ingress_host_mode
        )
        function = self._generate_runtime(self.runtime_kind)
        function.with_service_type(service_type, add_templated_ingress_host_mode)

        self.execute_function(function)
        args, _ = nuclio.deploy.deploy_config.call_args
        deploy_spec = args[0]["spec"]

        assert deploy_spec["serviceType"] == expected_service_type

        if expected_ingress_host_template is None:
            # never
            ingresses = (
                server.api.crud.runtimes.nuclio.helpers.resolve_function_ingresses(
                    deploy_spec
                )
            )
            assert ingresses == []

        else:
            ingresses = (
                server.api.crud.runtimes.nuclio.helpers.resolve_function_ingresses(
                    deploy_spec
                )
            )
            assert ingresses[0]["hostTemplate"] == expected_ingress_host_template

    def test_deploy_with_readiness_timeout_params(
        self, db: Session, client: TestClient
    ):
        function = self._generate_runtime(self.runtime_kind)
        function.spec.readiness_timeout = 501
        function.spec.readiness_timeout_before_failure = True

        self.execute_function(function)
        args, _ = nuclio.deploy.deploy_config.call_args
        deploy_spec = args[0]["spec"]

        assert deploy_spec["readinessTimeoutSeconds"] == 501
        assert deploy_spec["waitReadinessTimeoutBeforeFailure"]

    def test_deploy_with_disabled_http_trigger_creation(
        self, db: Session, client: TestClient
    ):
        # TODO: delete version mocking as soon as we release it in nuclio
        mlconf.nuclio_version = "1.13.1"
        function = self._generate_runtime(self.runtime_kind)
        function.disable_default_http_trigger()

        self.execute_function(function)
        args, _ = nuclio.deploy.deploy_config.call_args
        deploy_spec = args[0]["spec"]

        assert deploy_spec["disableDefaultHTTPTrigger"]

    def test_deploy_with_enabled_http_trigger_creation(
        self, db: Session, client: TestClient
    ):
        # TODO: delete version mocking as soon as we release it in nuclio
        mlconf.nuclio_version = "1.13.1"
        function = self._generate_runtime(self.runtime_kind)
        function.enable_default_http_trigger()

        self.execute_function(function)
        args, _ = nuclio.deploy.deploy_config.call_args
        deploy_spec = args[0]["spec"]

        assert not deploy_spec["disableDefaultHTTPTrigger"]

    def test_invoke_with_disabled_http_trigger_creation(
        self, db: Session, client: TestClient
    ):
        # TODO: delete version mocking as soon as we release it in nuclio
        mlconf.nuclio_version = "1.13.1"
        function = self._generate_runtime(self.runtime_kind)
        function.disable_default_http_trigger()

        self.execute_function(function)
        args, _ = nuclio.deploy.deploy_config.call_args

        with pytest.raises(mlrun.errors.MLRunPreconditionFailedError):
            function.invoke("/")

    def test_error_on_multiple_stream_triggers_old_nuclio_explicit_ack(self):
        mlconf.nuclio_version = "1.13.11"
        function = self._generate_runtime(self.runtime_kind)
        function.add_trigger(
            "stream1",
            nuclio.triggers.V3IOStreamTrigger(explicit_ack_mode="explicitOnly"),
        )
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Multiple triggers cannot be used in conjunction with explicit ack. "
            "Please upgrade to nuclio 1.13.12 or newer.",
        ):
            function.add_trigger(
                "stream2",
                nuclio.triggers.V3IOStreamTrigger(explicit_ack_mode="explicitOnly"),
            )

    def test_multiple_stream_triggers_new_nuclio_explicit_ack(self):
        mlconf.nuclio_version = "1.13.12"
        function = self._generate_runtime(self.runtime_kind)
        function.add_trigger(
            "stream1",
            nuclio.triggers.V3IOStreamTrigger(explicit_ack_mode="explicitOnly"),
        )
        function.add_trigger(
            "stream2",
            nuclio.triggers.V3IOStreamTrigger(explicit_ack_mode="explicitOnly"),
        )


# Kind of "nuclio:mlrun" is a special case of nuclio functions. Run the same suite of tests here as well
class TestNuclioMLRunRuntime(TestNuclioRuntime):
    @property
    def runtime_kind(self):
        # enables extending classes to run the same tests with different runtime
        return "nuclio:mlrun"


def get_archive_spec(function, secrets):
    spec = nuclio.ConfigSpec()
    config = {}
    server.api.crud.runtimes.nuclio.helpers.compile_nuclio_archive_config(
        spec, function, secrets
    )
    spec.merge(config)
    return config
