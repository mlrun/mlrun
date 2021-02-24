import json
import os
from http import HTTPStatus

from .test_nuclio import TestNuclioRuntime
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
import deepdiff
from mlrun import mlconf
import nuclio

# Needed for the serving test
from .assets.serving_functions import *  # noqa


class TestServingRuntime(TestNuclioRuntime):
    def custom_setup(self):
        super().custom_setup()
        self.inline_secrets = {
            "inline_secret1": "very secret",
            "inline_secret2": "terribly secret",
        }
        os.environ["ENV_SECRET1"] = "ENV SECRET!!!!"

        self.code_filename = str(self.assets_path / "serving_functions.py")

        self._mock_vault_functionality()

    def _create_serving_function(self):
        function = self._generate_runtime("serving")
        graph = function.set_topology("flow", exist_ok=True, engine="sync")

        graph.add_step(name="s1", class_name="Chain", secret="inline_secret1")
        graph.add_step(name="s3", class_name="Chain", after="$prev", secret="AWS_KEY")
        graph.add_step(
            name="s2", class_name="Chain", after="s1", before="s3", secret="ENV_SECRET1"
        )

        function.with_secrets("inline", self.inline_secrets)
        function.with_secrets("env", "ENV_SECRET1")
        function.with_secrets("vault", self.vault_secrets)
        return function

    def _assert_deploy_spec_has_secrets_config(self, expected_secret_sources):
        args, _ = nuclio.deploy.deploy_config.call_args
        deploy_spec = args[0]["spec"]

        token_path = mlconf.secret_stores.vault.token_path.replace("~", "/root")
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
            }
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

    def test_remote_deploy_with_secrets(self, db: Session, client: TestClient):
        function = self._create_serving_function()

        function.deploy(dashboard="dashboard", verbose=True)
        self._assert_deploy_called_basic_config(expected_class="serving")

        full_inline_secrets = self.inline_secrets.copy()
        full_inline_secrets["ENV_SECRET1"] = os.environ["ENV_SECRET1"]
        expected_secret_sources = [
            {"kind": "inline", "source": full_inline_secrets},
            {
                "kind": "vault",
                "source": {"project": self.project, "secrets": self.vault_secrets},
            },
        ]
        self._assert_deploy_spec_has_secrets_config(
            expected_secret_sources=expected_secret_sources
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

    def test_serving_with_secrets_remote_build(self, db: Session, client: TestClient):
        function = self._create_serving_function()

        # Simulate a remote build by issuing client's API. Code below is taken from httpdb.
        req = {
            "function": function.to_dict(),
            "with_mlrun": "no",
            "mlrun_version_specifier": "0.6.0",
        }
        response = client.post("/api/build/function", json=req)

        assert response.status_code == HTTPStatus.OK.value

        self._assert_deploy_called_basic_config(expected_class="serving")
