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

import base64
import json
import os
import typing
import unittest.mock
from http import HTTPStatus

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
import mlrun.common.types
import mlrun.errors
import mlrun.k8s_utils
import mlrun.runtimes.nuclio.function
import mlrun.runtimes.pod
from mlrun import code_to_function, mlconf
from mlrun.common.runtimes.constants import NuclioIngressAddTemplatedIngressModes
from mlrun.platforms.iguazio import split_path
from mlrun.utils import logger

import services.api.crud.runtimes.nuclio.function
import services.api.crud.runtimes.nuclio.helpers
from services.api.api.endpoints.nuclio import _validate_sidecar_probes
from services.api.tests.unit.conftest import APIK8sSecretsMock
from services.api.tests.unit.runtimes.base import TestRuntimeBase
from services.api.utils.functions import build_function


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
        services.api.crud.runtimes.nuclio.function.deploy_nuclio_function(
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

    def _assert_batching_spec(
        self,
        function,
        enabled,
        expected_size=None,
        expected_timeout=None,
    ):
        batch_info = function.spec.config["spec.triggers.http"].get("batch")
        if enabled:
            assert batch_info["mode"] == "enable"
        else:
            assert batch_info is None
            return
        assert batch_info.get("batchSize") == expected_size
        assert batch_info.get("timeout") == expected_timeout

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

        assert (
            deepdiff.DeepDiff(
                deploy_spec["volumes"],
                [expected_volume],
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
        ) = services.api.crud.runtimes.nuclio.function._compile_function_config(
            function
        )
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
        ) = services.api.crud.runtimes.nuclio.function._compile_function_config(
            function
        )
        service_type = "NodePort"
        services.api.crud.runtimes.nuclio.helpers.enrich_function_with_ingress(
            config, NuclioIngressAddTemplatedIngressModes.always, service_type
        )
        ingresses = (
            services.api.crud.runtimes.nuclio.helpers.resolve_function_ingresses(
                config["spec"]
            )
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
        ) = services.api.crud.runtimes.nuclio.function._compile_function_config(
            function
        )
        service_type = "NodePort"
        services.api.crud.runtimes.nuclio.helpers.enrich_function_with_ingress(
            config, NuclioIngressAddTemplatedIngressModes.always, service_type
        )
        ingresses = (
            services.api.crud.runtimes.nuclio.helpers.resolve_function_ingresses(
                config["spec"]
            )
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
        ) = services.api.crud.runtimes.nuclio.function._compile_function_config(
            function
        )
        service_type = "ClusterIP"
        services.api.crud.runtimes.nuclio.helpers.enrich_function_with_ingress(
            config,
            NuclioIngressAddTemplatedIngressModes.on_cluster_ip,
            service_type,
        )
        ingresses = (
            services.api.crud.runtimes.nuclio.helpers.resolve_function_ingresses(
                config["spec"]
            )
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
        ) = services.api.crud.runtimes.nuclio.function._compile_function_config(
            function
        )
        service_type = "DoesNotMatter"
        services.api.crud.runtimes.nuclio.helpers.enrich_function_with_ingress(
            config, NuclioIngressAddTemplatedIngressModes.never, service_type
        )
        ingresses = (
            services.api.crud.runtimes.nuclio.helpers.resolve_function_ingresses(
                config["spec"]
            )
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
            # TODO: Remove this in 1.12.0 — deprecated MLRUN_DEFAULT_PROJECT injected for backward compatibility
            {"name": "MLRUN_DEFAULT_PROJECT", "value": self.project},
        ]

        (
            function_name,
            project_name,
            config,
        ) = services.api.crud.runtimes.nuclio.function._compile_function_config(
            function
        )
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
        ) = services.api.crud.runtimes.nuclio.function._compile_function_config(
            function
        )
        for expected_env_var in expected_env_vars:
            assert expected_env_var in config["spec"]["env"]

    def test_deploy_with_project_secrets(
        self, db: Session, k8s_secrets_mock: APIK8sSecretsMock
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
        self, db: Session, k8s_secrets_mock: APIK8sSecretsMock
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
        self, db: Session, k8s_secrets_mock: APIK8sSecretsMock
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
        self, db: Session, k8s_secrets_mock: APIK8sSecretsMock
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
        self, db: Session, k8s_secrets_mock: APIK8sSecretsMock
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
        self, db: Session, k8s_secrets_mock: APIK8sSecretsMock
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
        self, db: Session, k8s_secrets_mock: APIK8sSecretsMock
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
        self, db: Session, k8s_secrets_mock: APIK8sSecretsMock
    ):
        """When spec.image and also spec.build.base_image are both defined the spec.image should be applied
        to spec.baseImage in nuclio."""

        function = self._generate_runtime(self.runtime_kind)
        function.spec.build.base_image = "mlrun/base_mlrun:latest"

        self.execute_function(function)
        self._assert_deploy_called_basic_config(expected_class=self.class_name)

    def test_deploy_without_image_and_build_base_image(
        self, db: Session, k8s_secrets_mock: APIK8sSecretsMock
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

    def test_deploy_with_batching(self, db: Session, client: TestClient):
        mlconf.nuclio_version = "1.14.0"
        function = self._generate_runtime(self.runtime_kind)

        http_trigger = {
            "batching_spec": mlrun.common.schemas.BatchingSpec(
                enabled=True, batch_size=2, timeout="1s"
            ),
        }

        # create http trigger with full batching spec
        function.with_http(**http_trigger)
        self._assert_batching_spec(
            function, enabled=True, expected_size=2, expected_timeout="1s"
        )

        # disable batching
        function.with_http(batching_spec=None)
        self._assert_batching_spec(function, enabled=False)

        # enable batching again, but without setting size/timeout (will be set to Nuclio's defaults)
        function.with_http(
            batching_spec=mlrun.common.schemas.BatchingSpec(enabled=True)
        )
        self._assert_batching_spec(function, enabled=True)

        # disable again
        function.with_http(batching_spec=None)
        self._assert_batching_spec(function, enabled=False)

        mlconf.nuclio_version = "1.13.9"
        with pytest.raises(mlrun.errors.MLRunValueError):
            function.with_http(
                batching_spec=mlrun.common.schemas.BatchingSpec(enabled=True)
            )

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
        ) = services.api.crud.runtimes.nuclio.function._compile_function_config(
            function
        )
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
            # explicit python version
            ("1.11.0", "3.9", "1.14.14", "python:3.9"),
            ("1.11.0", "3.11", "1.14.14", "python:3.11"),
            # no explicit python version defaults to config
            (None, None, "1.14.14", mlrun.mlconf.default_nuclio_runtime),
            ("1.11.0", None, "1.14.14", mlrun.mlconf.default_nuclio_runtime),
            # mlrun is known, not forcing any python version
            ("0.0.0-unstable", "3.9", "1.14.14", "python:3.9"),
            ("0.0.0-unstable", "3.11", "1.14.14", "python:3.11"),
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

    def test_deploy_python_version_validations(self, db: Session, client: TestClient):
        mlconf.default_nuclio_runtime = "python:3.7"

        logger.info("Function runtime is golang - do nothing")
        function = self._generate_runtime(self.runtime_kind)
        function.spec.nuclio_runtime = "golang"
        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name,
            expected_nuclio_runtime=function.spec.nuclio_runtime,
        )

        logger.info(
            "Function runtime is configured to python:3.7, nuclio version > 1.14.0 and no base image - explode"
        )
        function = self._generate_runtime(self.runtime_kind)
        function.spec.nuclio_runtime = "python:3.7"
        mlconf.nuclio_version = "1.14.1"
        function.spec.image = None
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match=r"(.*)Nuclio version does not support(.*)",
        ):
            self.execute_function(function)

        logger.info("Function runtime is python, but nuclio is >=1.8.0 - do nothing")
        self._reset_mock()
        mlconf.nuclio_version = "1.8.5"
        function = self._generate_runtime(self.runtime_kind)
        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name,
            expected_nuclio_runtime=mlconf.default_nuclio_runtime,
        )

        logger.info(
            "Function runtime is python, nuclio version in range, but already has the env var set - do nothing"
        )
        self._reset_mock()
        mlconf.nuclio_version = "1.14.14"
        function = self._generate_runtime(self.runtime_kind)
        function.set_env("something", "false")
        self.execute_function(function)
        self._assert_deploy_called_basic_config(
            expected_class=self.class_name,
            expected_nuclio_runtime=mlconf.default_nuclio_runtime,
            expected_env={"something": "false"},
        )

    @pytest.mark.parametrize(
        "nuclio_version,min_version,max_version,expected_result",
        [
            ("1.7.2", "1.6.11", "1.7.2", False),
            ("1.7.2", "1.7.0", "1.3.1", False),
            ("1.7.2", "1.7.3", "1.8.5", False),
            ("1.7.2", "1.7.2", "1.7.2", False),
            ("1.7.2", "1.7.2", "1.7.3", True),
            ("1.7.2", "1.7.0", "1.7.3", True),
            ("1.7.2", "1.5.5", "1.7.3", True),
            ("1.7.2", "1.5.5", "2.3.4", True),
            # best effort - assumes compatibility
            ("", "1.5.5", "2.3.4", True),
            ("", "1.7.2", "1.7.2", True),
        ],
    )
    def test_is_nuclio_version_in_range(
        self, nuclio_version, min_version, max_version, expected_result
    ):
        mlconf.nuclio_version = nuclio_version
        assert (
            services.api.crud.runtimes.nuclio.helpers.is_nuclio_version_in_range(
                min_version, max_version
            )
            is expected_result
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

    @pytest.mark.parametrize(
        "case",
        [
            ["1.6.11"],
            ["2.6.11"],
            ["1.5.9", "1.6.11"],
        ],
    )
    def test_min_nuclio_versions_decorator_failure(self, case):
        mlconf.nuclio_version = "1.6.10"

        @mlrun.runtimes.nuclio.function.min_nuclio_versions(*case)
        def fail():
            pytest.fail("Should not enter this function")

        with pytest.raises(mlrun.errors.MLRunIncompatibleVersionError):
            fail()

    @pytest.mark.parametrize(
        "nuclio_version",
        ["1.6.10", "2.2.1", "", "Gibberish"],
    )
    @pytest.mark.parametrize(
        "min_nuclio_versions_args",
        [
            ["1.6.9"],
            ["1.5.9", "1.6.9"],
            ["1.0.0", "0.9.81", "1.4.1"],
        ],
    )
    def test_min_nuclio_versions_decorator_success(
        self, nuclio_version, min_nuclio_versions_args
    ):
        mlconf.nuclio_version = nuclio_version

        @mlrun.runtimes.nuclio.function.min_nuclio_versions(*min_nuclio_versions_args)
        def success():
            pass

        success()

    def test_load_function_with_source_archive_git(self):
        fn = self._generate_runtime(self.runtime_kind)
        handler = "main:handler"
        fn.with_source_archive(
            "git://github.com/org/repo#my-branch",
            handler=handler,
            workdir="path/inside/repo",
        )
        secrets = {"GIT_PASSWORD": "my-access-token"}

        get_archive_spec(fn, secrets)
        assert get_archive_spec(fn, secrets) == {
            "spec": {
                "handler": handler,
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
            handler=handler,
            workdir="path/inside/repo",
        )

        assert get_archive_spec(fn, secrets) == {
            "spec": {
                "handler": handler,
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

        # ensure handler is not overridden if not passed
        fn.with_source_archive(
            "git://github.com/org/repo#refs/heads/my-other-branch",
        )
        assert fn.spec.function_handler == handler

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
                        "headers": {
                            mlrun.common.schemas.HeaderNames.v3io_session_key: "ma-access-key"
                        },
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
        ) = services.api.crud.runtimes.nuclio.function._compile_function_config(fn)
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
                services.api.crud.runtimes.nuclio.helpers.resolve_function_ingresses(
                    deploy_spec
                )
            )
            assert ingresses == []

        else:
            ingresses = (
                services.api.crud.runtimes.nuclio.helpers.resolve_function_ingresses(
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
        with unittest.mock.patch.object(
            function, "_get_state", return_value=("ready", "", None)
        ):
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

    @pytest.mark.parametrize(
        "nuclio_version",
        [
            "1.13.12",
            "unstable",
        ],
    )
    def test_multiple_stream_triggers_new_nuclio_explicit_ack(self, nuclio_version):
        mlconf.nuclio_version = nuclio_version
        function = self._generate_runtime(self.runtime_kind)
        function.add_trigger(
            "stream1",
            nuclio.triggers.V3IOStreamTrigger(explicit_ack_mode="explicitOnly"),
        )
        function.add_trigger(
            "stream2",
            nuclio.triggers.V3IOStreamTrigger(explicit_ack_mode="explicitOnly"),
        )

    def test_masking_sensitive_fields(self):
        function = self._generate_runtime(self.runtime_kind)

        raw_password = "raw_password"
        function.add_v3io_stream_trigger(
            stream_path="test",
            access_key=raw_password,
            extra_attributes={"password": raw_password},
        )
        raw_config = function.mask_sensitive_data_in_config()

        assert (
            function.spec.config.get("spec.triggers.stream").get("password")
            == "$ref:/spec/triggers/stream/password"
        )
        assert raw_config.get("spec.triggers.stream").get("password") == raw_password

        assert (
            function.spec.config.get("spec.triggers.stream")
            .get("attributes", {})
            .get("password")
            == "$ref:/spec/triggers/stream/attributes/password"
        )
        assert (
            raw_config.get("spec.triggers.stream").get("attributes", {}).get("password")
            == raw_password
        )

    @pytest.mark.parametrize(
        "nuclio_version",
        [
            "1.14.15",
            "1.14.11",
        ],
    )
    def test_masking_rabbitmq_url(self, nuclio_version):
        """Test that RabbitMQ URL credentials are extracted and masked properly."""
        password = "rabbit123"
        url = (
            f"amqp://user:{password}@my-rabbitmq.default-tenant.svc.cluster.local:5672"
        )

        def _validate_masked_trigger(masked_trigger):
            if nuclio_version == "1.14.15":
                assert (
                    masked_trigger["url"]
                    == "amqp://my-rabbitmq.default-tenant.svc.cluster.local:5672"
                )

                # Assert: Password is masked in attributes
                assert masked_trigger["password"] != password

                # Assert: Username is in attributes
                assert masked_trigger["username"] == "user"
            else:
                # should not mask for versions less than 1.14.15
                assert masked_trigger["url"] == url

            # should not be masked in raw_config (this is config we send to nuclio)
            if "spec.triggers" in raw_config:
                assert (
                    raw_config.get("spec.triggers").get("rabbit-trigger").get("url")
                    == url
                )
            else:
                assert raw_config.get("spec.triggers.rabbit-trigger").get("url") == url

        mlconf.nuclio_version = nuclio_version
        function = self._generate_runtime(self.runtime_kind)
        attributes = {
            "exchangeName": "input_ex",
            "queueName": "input_queue",
        }

        # Option 1: set trigger via add_trigger
        function.add_trigger(
            "rabbit-trigger",
            {"kind": "rabbit-mq", "url": url, "attributes": attributes},
        )

        raw_config = function.mask_sensitive_data_in_config()
        masked_trigger = function.spec.config.get("spec.triggers.rabbit-trigger")
        _validate_masked_trigger(masked_trigger)

        # Option 2: set trigger via set_config
        triggers = {
            "rabbit-trigger": {
                "kind": "rabbit-mq",
                "url": url,
                "attributes": attributes,
            },
        }
        function.set_config("spec.triggers", triggers)
        raw_config = function.mask_sensitive_data_in_config()
        masked_trigger = function.spec.config.get("spec.triggers").get("rabbit-trigger")
        _validate_masked_trigger(masked_trigger)

    @pytest.mark.parametrize(
        "inside_k8s,force_external,internal_urls,external_urls,address,expected_url,expected_exception,disable_default_http_trigger",
        [
            # Prefer internal when inside k8s and not forcing external
            (
                True,
                False,
                ["internal-url:1234"],
                ["external-url:5678"],
                "legacy-address:4321",
                "internal-url:1234",
                None,
                False,
            ),
            # Use external when forcing external
            (
                True,
                True,
                ["internal-url:1234"],
                ["external-url:5678"],
                "legacy-address:4321",
                "external-url:5678",
                None,
                False,
            ),
            # Use external when not inside k8s
            (
                False,
                False,
                ["internal-url:1234"],
                ["external-url:5678"],
                "legacy-address:4321",
                "external-url:5678",
                None,
                False,
            ),
            # Fallback to address if no invocation urls
            (
                True,
                False,
                [],
                [],
                "legacy-address:4321",
                "legacy-address:4321",
                None,
                False,
            ),
            # Error if no address and no triggers, default http trigger disabled
            (
                True,
                False,
                [],
                [],
                "",
                None,
                mlrun.errors.MLRunPreconditionFailedError,
                True,
            ),
            # Error if no address and no triggers, default http trigger enabled
            (True, False, [], [], "", None, ValueError, False),
            (False, True, [], [], "", None, ValueError, False),
        ],
    )
    def test_get_url(
        self,
        inside_k8s,
        force_external,
        internal_urls,
        external_urls,
        address,
        expected_url,
        expected_exception,
        disable_default_http_trigger,
    ):
        function = self._generate_runtime(self.runtime_kind)
        ingress_host = "something.com"
        function = function.with_http(host=ingress_host, paths=["/"], port=30030)

        function.status.internal_invocation_urls = internal_urls
        function.status.external_invocation_urls = external_urls
        function.status.address = address
        function.spec.disable_default_http_trigger = disable_default_http_trigger

        for state in ["ready", "error", "building"]:
            with unittest.mock.patch.object(
                function, "_get_state", return_value=(state, "", None)
            ):
                with unittest.mock.patch.object(
                    mlrun.k8s_utils,
                    "is_running_inside_kubernetes_cluster",
                    return_value=inside_k8s,
                ):
                    if expected_exception:
                        function.spec.config = (
                            {}
                            if expected_exception
                            == mlrun.errors.MLRunPreconditionFailedError
                            else function.spec.config
                        )
                        with pytest.raises(expected_exception):
                            function.get_url(force_external_address=force_external)
                    else:
                        url = function.get_url(force_external_address=force_external)
                        assert isinstance(url, str)
                        assert (
                            expected_url in url
                            if expected_url
                            else url.startswith("http")
                        )

    def test_compile_function_config_with_auth_secret(self):
        function = self._generate_runtime(self.runtime_kind)

        # minimal auth spec
        function.spec.auth = {"token_name": "default"}

        auth_info = unittest.mock.Mock()
        auth_info.username = "test-user"
        mlrun.mlconf.httpdb.authentication.mode = (
            mlrun.common.types.AuthenticationMode.IGUAZIO_V4
        )

        with unittest.mock.patch(
            "framework.utils.singletons.k8s.get_k8s_helper"
        ) as k8s_helper_mock:
            # fake k8s secret
            secret = unittest.mock.Mock()
            secret.metadata.name = "mlrun-auth-secrets.123456"
            k8s_helper_mock.return_value._get_user_token_secret.return_value = secret

            _, _, config = (
                services.api.crud.runtimes.nuclio.function._compile_function_config(
                    function=function,
                    auth_info=auth_info,
                )
            )

        volumes = mlrun.utils.get_in(config, "spec.volumes", [])

        auth_volumes = [
            volume
            for volume in volumes
            if volume.get("volume", {})
            .get("secret", {})
            .get("secretName", "")
            .startswith("mlrun-auth-secrets")
        ]

        assert len(auth_volumes) == 1

        auth_volume = auth_volumes[0]

        assert auth_volume["volume"]["secret"]["items"] == [
            {
                "key": "tokensFile",
                "path": mlrun.common.constants.MLRUN_JOB_AUTH_SECRET_FILE,
            }
        ]

        assert auth_volume["volumeMount"]["mountPath"] == (
            mlrun.common.constants.MLRUN_JOB_AUTH_SECRET_PATH
        )

    def test_compile_function_config_non_iguazio_v4(self):
        function = self._generate_runtime(self.runtime_kind)

        auth_info = unittest.mock.Mock()
        auth_info.username = "test-user"
        mlrun.mlconf.httpdb.authentication.mode = (
            mlrun.common.types.AuthenticationMode.IGUAZIO
        )

        _, _, config = (
            services.api.crud.runtimes.nuclio.function._compile_function_config(
                function=function,
                auth_info=auth_info,
            )
        )

        volumes = mlrun.utils.get_in(config, "spec.volumes", [])

        assert not any(
            volume.get("secret", {})
            .get("secretName", "")
            .startswith("mlrun-auth-secrets")
            for volume in volumes
        )

    def test_validate_sidecar_probes_positive_flow(self):
        # Test 3 sidecars, each with a different health check method
        sidecars = [
            {
                "name": "sidecar-http",
                "readinessProbe": {
                    "httpGet": {
                        "path": "/ready",
                        "port": 8080,
                    },
                    "initialDelaySeconds": 5,
                    "periodSeconds": 3,
                },
            },
            {
                "name": "sidecar-exec",
                "livenessProbe": {
                    "exec": {
                        "command": ["/bin/sh", "-c", "cat /tmp/healthy"],
                    },
                    "initialDelaySeconds": 10,
                    "periodSeconds": 5,
                },
            },
            {
                "name": "sidecar-tcp",
                "startupProbe": {
                    "tcpSocket": {
                        "port": 9090,
                    },
                    "initialDelaySeconds": 15,
                    "periodSeconds": 10,
                },
                "livenessProbe": {
                    "grpc": {
                        "port": 9091,
                        "service": "health",
                    },
                    "initialDelaySeconds": 20,
                    "periodSeconds": 5,
                },
            },
        ]

        _validate_sidecar_probes(sidecars)

    def test_validate_sidecar_probes_invalid_configurations(self):
        # Test various invalid probe configurations - should raise HTTPException
        invalid_sidecar_configs = [
            [
                {
                    "name": "test-sidecar-missing",
                    "readinessProbe": {
                        "initialDelaySeconds": 5,
                        "periodSeconds": 3,
                    },
                }
            ],
            [
                {
                    "name": "test-sidecar-more-than-one_health-check",
                    "readinessProbe": {
                        "initialDelaySeconds": 5,
                        "periodSeconds": 3,
                        "tcpSocket": {
                            "port": 9090,
                        },
                        "httpGet": {
                            "path": "/ready",
                            "port": 8080,
                        },
                    },
                }
            ],
        ]

        for sidecars in invalid_sidecar_configs:
            with pytest.raises(HTTPException) as exception_result:
                _validate_sidecar_probes(sidecars)

            assert exception_result.value.status_code == HTTPStatus.BAD_REQUEST.value
            assert "must have exactly one of" in str(
                exception_result.value.detail.get("reason", "")
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
    services.api.crud.runtimes.nuclio.helpers.compile_nuclio_archive_config(
        function,
        spec,
        secrets,
    )
    spec.merge(config)
    return config
