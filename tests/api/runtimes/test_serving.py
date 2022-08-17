import json
import os
import unittest
import unittest.mock
from http import HTTPStatus

import deepdiff
import nuclio
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.api.utils
import tests.api.api.utils
from mlrun import mlconf, new_function
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.db import SQLDB
from mlrun.runtimes.function import (
    NuclioStatus,
    compile_function_config,
    deploy_nuclio_function,
)

from .assets.serving_child_functions import *  # noqa

# Needed for the serving test
from .assets.serving_functions import *  # noqa
from .test_nuclio import TestNuclioRuntime


class TestServingRuntime(TestNuclioRuntime):
    @property
    def runtime_kind(self):
        # enables extending classes to run the same tests with different runtime
        return "serving"

    @property
    def class_name(self):
        # enables extending classes to run the same tests with different class
        return "serving"

    def custom_setup_after_fixtures(self):
        self._mock_nuclio_deploy_config()
        self._mock_vault_functionality()
        # Since most of the Serving runtime handling is done client-side, we'll mock the calls to remote-build
        # and instead just call the deploy_nuclio_function() API which actually performs the
        # deployment in this case. This will keep the tests' code mostly client-side oriented, but validations
        # will be performed against the Nuclio spec created on the server side.
        self._mock_db_remote_deploy_functions()

    def custom_setup(self):
        super().custom_setup()
        self.inline_secrets = {
            "inline_secret1": "very secret",
            "inline_secret2": "terribly secret",
        }
        os.environ["ENV_SECRET1"] = "ENV SECRET!!!!"

        self.code_filename = str(self.assets_path / "serving_functions.py")

    @staticmethod
    def _mock_db_remote_deploy_functions():
        def _remote_db_mock_function(func, with_mlrun, builder_env=None):
            deploy_nuclio_function(func)
            return {
                "data": {
                    "status": NuclioStatus(
                        state="ready",
                        nuclio_name=f"nuclio-{func.metadata.name}",
                        address="http://127.0.0.1:1234",
                        external_invocation_urls=["http://somewhere-far-away.com"],
                        internal_invocation_urls=["http://127.0.0.1:1234"],
                    )
                }
            }

        # Since we're in a test, the RunDB is of type SQLDB, not HTTPDB as it would usually be.
        SQLDB.remote_builder = unittest.mock.Mock(side_effect=_remote_db_mock_function)
        SQLDB.get_builder_status = unittest.mock.Mock(return_value=("text", "last_log"))

    def _create_serving_function(self):
        function = self._generate_runtime(self.runtime_kind)
        graph = function.set_topology("flow", exist_ok=True, engine="sync")

        graph.add_step(name="s1", class_name="Chain", secret="inline_secret1")
        graph.add_step(name="s3", class_name="Chain", after="$prev", secret="AWS_KEY")
        graph.add_step(
            name="s2", class_name="Chain", after="s1", before="s3", secret="ENV_SECRET1"
        )

        function.with_secrets("inline", self.inline_secrets)
        function.with_secrets("env", "ENV_SECRET1")
        function.with_secrets("vault", self.vault_secrets)
        function.with_secrets(
            "azure_vault",
            {
                "name": "azure-key-vault",
                "k8s_secret": self.azure_vault_secret_name,
                "secrets": [],
            },
        )
        return function

    def _assert_deploy_spec_has_secrets_config(self, expected_secret_sources):
        call_args_list = nuclio.deploy.deploy_config.call_args_list
        for single_call_args in call_args_list:
            args, _ = single_call_args
            deploy_spec = args[0]["spec"]

            token_path = mlconf.secret_stores.vault.token_path.replace("~", "/root")
            azure_secret_path = mlconf.secret_stores.azure_vault.secret_path.replace(
                "~", "/root"
            )
            expected_volumes = [
                {
                    "volume": {
                        "name": "vault-secret",
                        "secret": {
                            "defaultMode": 420,
                            "secretName": self.vault_secret_name,
                        },
                    },
                    "volumeMount": {"name": "vault-secret", "mountPath": token_path},
                },
                {
                    "volume": {
                        "name": "azure-vault-secret",
                        "secret": {
                            "defaultMode": 420,
                            "secretName": self.azure_vault_secret_name,
                        },
                    },
                    "volumeMount": {
                        "name": "azure-vault-secret",
                        "mountPath": azure_secret_path,
                    },
                },
            ]
            assert (
                deepdiff.DeepDiff(
                    deploy_spec["volumes"], expected_volumes, ignore_order=True
                )
                == {}
            )

            expected_env = {
                "MLRUN_SECRET_STORES__VAULT__ROLE": f"project:{self.project}",
                "MLRUN_SECRET_STORES__VAULT__URL": mlconf.secret_stores.vault.url,
                # For now, just checking the variable exists, later we check specific contents
                "SERVING_SPEC_ENV": None,
            }
            self._assert_pod_env(deploy_spec["env"], expected_env)

            for env_variable in deploy_spec["env"]:
                if env_variable["name"] == "SERVING_SPEC_ENV":
                    serving_spec = json.loads(env_variable["value"])
                    assert (
                        deepdiff.DeepDiff(
                            serving_spec["secret_sources"],
                            expected_secret_sources,
                            ignore_order=True,
                        )
                        == {}
                    )

    def _generate_expected_secret_sources(self):
        full_inline_secrets = self.inline_secrets.copy()
        full_inline_secrets["ENV_SECRET1"] = os.environ["ENV_SECRET1"]
        expected_secret_sources = [
            {"kind": "inline", "source": full_inline_secrets},
            {
                "kind": "vault",
                "source": {"project": self.project, "secrets": self.vault_secrets},
            },
            {
                "kind": "azure_vault",
                "source": {
                    "name": "azure-key-vault",
                    "k8s_secret": self.azure_vault_secret_name,
                    "secrets": [],
                },
            },
        ]
        return expected_secret_sources

    def test_remote_deploy_with_secrets(self, db: Session, client: TestClient):
        function = self._create_serving_function()

        function.deploy(verbose=True)
        self._assert_deploy_called_basic_config(expected_class=self.class_name)

        self._assert_deploy_spec_has_secrets_config(
            expected_secret_sources=self._generate_expected_secret_sources()
        )

    def test_mock_server_secrets(self, db: Session, client: TestClient):
        function = self._create_serving_function()

        server = function.to_mock_server()

        # Verify all secrets are in the context
        for secret_key in self.vault_secrets:
            assert server.context.get_secret(secret_key) == self.vault_secret_value
        for secret_key in self.inline_secrets:
            assert (
                server.context.get_secret(secret_key) == self.inline_secrets[secret_key]
            )
        assert server.context.get_secret("ENV_SECRET1") == os.environ["ENV_SECRET1"]

        resp = server.test(body=[])

        expected_response = [
            {"inline_secret1": self.inline_secrets["inline_secret1"]},
            {"ENV_SECRET1": os.environ["ENV_SECRET1"]},
            {"AWS_KEY": self.vault_secret_value},
        ]

        assert deepdiff.DeepDiff(resp, expected_response) == {}

    def test_mock_bad_step(self, db: Session, client: TestClient):
        function = self._generate_runtime(self.runtime_kind)
        graph = function.set_topology("flow", exist_ok=True, engine="sync")

        graph.add_step(
            name="extend", class_name="storey.Extend", _fn='({"tag": "something"})'
        )

        server = function.to_mock_server()
        with pytest.raises(RuntimeError):
            server.test()

    def test_serving_with_secrets_remote_build(self, db: Session, client: TestClient):
        orig_function = get_k8s()._get_project_secrets_raw_data
        get_k8s()._get_project_secrets_raw_data = unittest.mock.Mock(return_value={})
        mlrun.api.api.utils.mask_function_sensitive_data = unittest.mock.Mock()

        function = self._create_serving_function()
        tests.api.api.utils.create_project(client, self.project)

        # Simulate a remote build by issuing client's API. Code below is taken from httpdb.
        req = {
            "function": function.to_dict(),
            "with_mlrun": "no",
            "mlrun_version_specifier": "0.6.0",
        }
        response = client.post("build/function", json=req)

        assert response.status_code == HTTPStatus.OK.value

        self._assert_deploy_called_basic_config(expected_class=self.class_name)

        get_k8s()._get_project_secrets_raw_data = orig_function

    def test_child_functions_with_secrets(self, db: Session, client: TestClient):
        function = self._create_serving_function()
        graph = function.spec.graph
        graph.add_step(
            name="s4",
            class_name="ChildChain",
            after="s3",
            function="child_function",
            secret="inline_secret2",
        )
        graph.add_step(
            name="s5",
            class_name="ChildChain",
            after="s4",
            function="child_function",
            secret="AWS_KEY",
        )
        child_function_path = str(self.assets_path / "serving_child_functions.py")
        function.add_child_function(
            "child_function", child_function_path, self.image_name
        )

        function.deploy(verbose=True)
        # Child function is deployed before main function
        expected_deploy_params = [
            {
                "function_name": f"{self.project}-{self.name}-child_function",
                "file_name": child_function_path,
                "parent_function": function.metadata.name,
            },
            {
                "function_name": f"{self.project}-{self.name}",
                "file_name": self.code_filename,
            },
        ]

        self._assert_deploy_called_basic_config(
            expected_class=self.class_name,
            call_count=2,
            expected_params=expected_deploy_params,
        )

        self._assert_deploy_spec_has_secrets_config(
            expected_secret_sources=self._generate_expected_secret_sources()
        )

    def test_empty_function(self):
        # test simple function (no source)
        function = new_function("serving", kind="serving", image="mlrun/mlrun")
        function.set_topology("flow")
        _, _, config = compile_function_config(function)
        # verify the code is filled with the mlrun serving wrapper
        assert config["spec"]["build"]["functionSourceCode"]

        # test function built from source repo (set the handler)
        function = new_function(
            "serving", kind="serving", image="mlrun/mlrun", source="git://x/y#z"
        )
        function.set_topology("flow")

        # mock secrets for the source (so it will not fail)
        orig_function = get_k8s()._get_project_secrets_raw_data
        get_k8s()._get_project_secrets_raw_data = unittest.mock.Mock(return_value={})
        _, _, config = compile_function_config(function, builder_env={})
        get_k8s()._get_project_secrets_raw_data = orig_function

        # verify the handler points to mlrun serving wrapper handler
        assert config["spec"]["handler"].startswith("mlrun.serving")
