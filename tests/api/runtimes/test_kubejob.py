import os
import unittest.mock

import deepdiff
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.errors
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.config import config as mlconf
from mlrun.platforms import auto_mount
from mlrun.runtimes.kubejob import KubejobRuntime
from mlrun.runtimes.utils import generate_resources
from mlrun.secrets import SecretsStore
from tests.api.runtimes.base import TestRuntimeBase


class TestKubejobRuntime(TestRuntimeBase):
    @pytest.fixture(autouse=True)
    def setup_method_fixture(self, db: Session, client: TestClient):
        # We want this mock for every test, ideally we would have simply put it in the custom_setup
        # but this function is called by the base class's setup_method which is happening before the fixtures
        # initialization. We need the client fixture (which needs the db one) in order to be able to mock k8s stuff
        self._mock_create_namespaced_pod()

    def custom_setup(self):
        self.image_name = "mlrun/mlrun:latest"

    def _generate_runtime(self):
        runtime = KubejobRuntime()
        runtime.spec.image = self.image_name
        return runtime

    def test_run_without_runspec(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()
        self._execute_run(runtime)
        self._assert_pod_creation_config()

        params = {"p1": "v1", "p2": 20}
        inputs = {"input1": f"{self.artifact_path}/input1.txt"}

        self._execute_run(runtime, params=params, inputs=inputs)
        self._assert_pod_creation_config(expected_params=params, expected_inputs=inputs)

    def test_run_with_runspec(self, db: Session, client: TestClient):
        task = self._generate_task()
        params = {"p1": "v1", "p2": 20}
        task.with_params(**params)
        inputs = {
            "input1": f"{self.artifact_path}/input1.txt",
            "input2": f"{self.artifact_path}/input2.csv",
        }
        for key in inputs:
            task.with_input(key, inputs[key])
        hyper_params = {"p2": [1, 2, 3]}
        task.with_hyper_params(hyper_params, "min.loss")
        secret_source = {
            "kind": "inline",
            "source": {"secret1": "password1", "secret2": "password2"},
        }
        task.with_secrets(secret_source["kind"], secret_source["source"])

        runtime = self._generate_runtime()
        self._execute_run(runtime, runspec=task)
        self._assert_pod_creation_config(
            expected_params=params,
            expected_inputs=inputs,
            expected_hyper_params=hyper_params,
            expected_secrets=secret_source,
        )

    def test_run_with_resource_limits_and_requests(
        self, db: Session, client: TestClient
    ):
        runtime = self._generate_runtime()

        gpu_type = "test/gpu"
        expected_limits = generate_resources(2, 4, 4, gpu_type)
        runtime.with_limits(
            mem=expected_limits["memory"],
            cpu=expected_limits["cpu"],
            gpus=expected_limits[gpu_type],
            gpu_type=gpu_type,
        )

        expected_requests = generate_resources(mem=2, cpu=3)
        runtime.with_requests(
            mem=expected_requests["memory"], cpu=expected_requests["cpu"]
        )

        self._execute_run(runtime)
        self._assert_pod_creation_config(
            expected_limits=expected_limits, expected_requests=expected_requests
        )

    def test_run_with_node_selection(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        node_name = "some-node-name"
        runtime.with_node_selection(node_name)
        self._execute_run(runtime)
        self._assert_pod_creation_config(expected_node_name=node_name)

        runtime = self._generate_runtime()

        node_selector = {
            "label-a": "val1",
            "label-2": "val2",
        }
        runtime.with_node_selection(node_selector=node_selector)
        self._execute_run(runtime)
        self._assert_pod_creation_config(expected_node_selector=node_selector)

        runtime = self._generate_runtime()
        affinity = self._generate_affinity()
        runtime.with_node_selection(affinity=affinity)
        self._execute_run(runtime)
        self._assert_pod_creation_config(expected_affinity=affinity)

        runtime = self._generate_runtime()
        runtime.with_node_selection(node_name, node_selector, affinity)
        self._execute_run(runtime)
        self._assert_pod_creation_config(
            expected_node_name=node_name,
            expected_node_selector=node_selector,
            expected_affinity=affinity,
        )

    def test_run_with_mounts(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        # Mount v3io - Set the env variable, so auto_mount() will pick it up and mount v3io
        v3io_access_key = "1111-2222-3333-4444"
        v3io_user = "test-user"
        os.environ["V3IO_ACCESS_KEY"] = v3io_access_key
        os.environ["V3IO_USERNAME"] = v3io_user
        runtime.apply(auto_mount())

        self._execute_run(runtime)
        self._assert_pod_creation_config()
        self._assert_v3io_mount_configured(v3io_user, v3io_access_key)

        # Mount a PVC. Create a new runtime so we don't have both v3io and the PVC mounted
        runtime = self._generate_runtime()
        pvc_name = "test-pvc"
        pvc_mount_path = "/volume/mount/path"
        volume_name = "test-volume-name"
        runtime.apply(auto_mount(pvc_name, pvc_mount_path, volume_name))

        self._execute_run(runtime)
        self._assert_pod_creation_config()
        self._assert_pvc_mount_configured(pvc_name, pvc_mount_path, volume_name)

    def test_run_with_k8s_secrets(self, db: Session, client: TestClient):
        project_secret_name = "dummy_secret_name"
        secret_keys = ["secret1", "secret2", "secret3"]

        # Need to do some mocking, so code thinks that the secret contains these keys. Otherwise it will not add
        # the env. variables to the pod spec.
        get_k8s().get_project_secret_name = unittest.mock.Mock(
            return_value=project_secret_name
        )
        get_k8s().get_project_secret_keys = unittest.mock.Mock(return_value=secret_keys)

        runtime = self._generate_runtime()

        task = self._generate_task()
        task.metadata.project = self.project
        secret_source = {
            "kind": "kubernetes",
            "source": secret_keys,
        }
        task.with_secrets(secret_source["kind"], secret_keys)

        # What we expect in this case is that environment variables will be added to the pod which get their
        # value from the k8s secret, using the correct keys.
        expected_env_from_secrets = {}
        for key in secret_keys:
            env_variable_name = SecretsStore._k8s_env_variable_name_for_secret(key)
            expected_env_from_secrets[env_variable_name] = {project_secret_name: key}

        self._execute_run(runtime, runspec=task)

        self._assert_pod_creation_config(
            expected_secrets=secret_source,
            expected_env_from_secrets=expected_env_from_secrets,
        )

    def test_run_with_vault_secrets(self, db: Session, client: TestClient):
        self._mock_vault_functionality()
        runtime = self._generate_runtime()

        task = self._generate_task()

        task.metadata.project = self.project
        secret_source = {
            "kind": "vault",
            "source": {"project": self.project, "secrets": self.vault_secrets},
        }
        task.with_secrets(secret_source["kind"], self.vault_secrets)
        vault_url = "/url/for/vault"
        mlconf.secret_stores.vault.remote_url = vault_url
        mlconf.secret_stores.vault.token_path = vault_url

        self._execute_run(runtime, runspec=task)

        self._assert_pod_creation_config(
            expected_secrets=secret_source,
            expected_env={
                "MLRUN_SECRET_STORES__VAULT__ROLE": f"project:{self.project}",
                "MLRUN_SECRET_STORES__VAULT__URL": vault_url,
            },
        )

        self._assert_secret_mount(
            "vault-secret", self.vault_secret_name, 420, vault_url
        )

    def test_run_with_code(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        expected_code = """
def my_func(context):
    print("Hello cruel world")
        """
        runtime.with_code(body=expected_code)

        self._execute_run(runtime)
        self._assert_pod_creation_config(expected_code=expected_code)

    def test_set_env(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()
        env = {"MLRUN_LOG_LEVEL": "DEBUG", "IMAGE_HEIGHT": "128"}
        for env_variable in env:
            runtime.set_env(env_variable, env[env_variable])
        self._execute_run(runtime)
        self._assert_pod_creation_config(expected_env=env)

        # set the same env key for a different value and check that the updated one is used
        env2 = {"MLRUN_LOG_LEVEL": "ERROR", "IMAGE_HEIGHT": "128"}
        runtime.set_env("MLRUN_LOG_LEVEL", "ERROR")
        self._execute_run(runtime)
        self._assert_pod_creation_config(expected_env=env2)

    def test_run_with_code_with_file(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        runtime.with_code(from_file=self.code_filename)

        self._execute_run(runtime)
        self._assert_pod_creation_config(expected_code=open(self.code_filename).read())

    def test_run_with_code_and_file(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        expected_code = """
        def my_func(context):
            print("Hello cruel world")
                """

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as excinfo:
            runtime.with_code(from_file=self.code_filename, body=expected_code)
        assert "must provide either body or from_file argument. not both" in str(
            excinfo.value
        )

    def test_run_with_code_empty(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        with pytest.raises(ValueError) as excinfo:
            runtime.with_code()
        assert "please specify" in str(excinfo.value)

    def test_set_label(self, db: Session, client: TestClient):
        task = self._generate_task()
        task.set_label("category", "test")
        labels = {"category": "test"}

        runtime = self._generate_runtime()
        self._execute_run(runtime, runspec=task)
        self._assert_pod_creation_config(expected_labels=labels)

    def test_with_requirements(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()
        runtime.with_requirements(self.requirements_file)
        expected_commands = ["python -m pip install faker python-dotenv"]
        assert (
            deepdiff.DeepDiff(
                expected_commands, runtime.spec.build.commands, ignore_order=True,
            )
            == {}
        )
