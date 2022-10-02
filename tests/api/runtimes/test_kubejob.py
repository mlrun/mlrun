import base64
import json
import os

import deepdiff
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.schemas
import mlrun.errors
import mlrun.k8s_utils
from mlrun.api.schemas import SecurityContextEnrichmentModes
from mlrun.config import config as mlconf
from mlrun.platforms import auto_mount
from mlrun.runtimes.utils import generate_resources
from tests.api.conftest import K8sSecretsMock
from tests.api.runtimes.base import TestRuntimeBase


class TestKubejobRuntime(TestRuntimeBase):
    def custom_setup_after_fixtures(self):
        self._mock_create_namespaced_pod()
        # auto-mount is looking at this to check if we're running on Iguazio
        mlconf.igz_version = "some_version"

    def custom_setup(self):
        self.image_name = "mlrun/mlrun:latest"

    def _generate_runtime(self) -> mlrun.runtimes.KubejobRuntime:
        runtime = mlrun.runtimes.KubejobRuntime()
        runtime.spec.image = self.image_name
        return runtime

    def test_run_without_runspec(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()
        self.execute_function(runtime)
        self._assert_pod_creation_config()

        params = {"p1": "v1", "p2": 20}
        inputs = {"input1": f"{self.artifact_path}/input1.txt"}

        self.execute_function(runtime, params=params, inputs=inputs)
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
        self.execute_function(runtime, runspec=task)
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

        self.execute_function(runtime)
        self._assert_pod_creation_config(
            expected_limits=expected_limits, expected_requests=expected_requests
        )

    def test_run_without_specifying_resources(self, db: Session, client: TestClient):
        self.assert_run_without_specifying_resources()

    def test_run_with_node_selection(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        node_name = "some-node-name"
        runtime.with_node_selection(node_name)
        self.execute_function(runtime)
        self._assert_pod_creation_config(expected_node_name=node_name)

        runtime = self._generate_runtime()

        node_selector = {
            "label-1": "val1",
            "label-2": "val2",
        }
        mlrun.mlconf.default_function_node_selector = base64.b64encode(
            json.dumps(node_selector).encode("utf-8")
        )
        runtime.with_node_selection(node_selector=node_selector)
        self.execute_function(runtime)
        self._assert_pod_creation_config(expected_node_selector=node_selector)

        runtime = self._generate_runtime()

        node_selector = {
            "label-3": "val3",
            "label-4": "val4",
        }
        runtime.with_node_selection(node_selector=node_selector)
        self.execute_function(runtime)
        self._assert_pod_creation_config(expected_node_selector=node_selector)

        runtime = self._generate_runtime()
        affinity = self._generate_affinity()
        runtime.with_node_selection(affinity=affinity)
        self.execute_function(runtime)
        self._assert_pod_creation_config(expected_affinity=affinity)

        runtime = self._generate_runtime()
        runtime.with_node_selection(node_name, node_selector, affinity)
        self.execute_function(runtime)
        self._assert_pod_creation_config(
            expected_node_name=node_name,
            expected_node_selector=node_selector,
            expected_affinity=affinity,
        )

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

    def test_preemption_mode_with_preemptible_node_selector_without_preemptible_tolerations_with_extra_settings(
        self, db: Session, client: TestClient
    ):
        self.assert_run_preemption_mode_with_preemptible_node_selector_without_preemptible_tolerations_with_extra_settings()  # noqa: E501

    def test_with_preemption_mode_none_transitions(
        self, db: Session, client: TestClient
    ):
        self.assert_run_with_preemption_mode_none_transitions()

    def assert_node_selection(
        self, node_name=None, node_selector=None, affinity=None, tolerations=None
    ):
        pod = self._get_pod_creation_args()
        # doesn't need a special case because the default it to be set with default node selector
        assert pod.spec.node_selector == (node_selector or {})

        if node_name:
            assert pod.spec.node_name == node_name
        else:
            assert pod.spec.node_name is None

        if affinity:
            assert pod.spec.affinity == affinity
        else:
            assert pod.spec.affinity is None

        if tolerations:
            assert pod.spec.tolerations == tolerations
        else:
            assert pod.spec.tolerations is None

    def assert_security_context(
        self,
        security_context=None,
    ):
        pod = self._get_pod_creation_args()
        assert pod.spec.security_context == (security_context or {})

    def test_set_annotation(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()
        runtime.set_annotation("annotation-key", "annotation-value")
        self.execute_function(runtime)

        pod = self._get_pod_creation_args()
        assert pod.metadata.annotations.get("annotation-key") == "annotation-value"

    def test_run_with_priority_class_name(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        medium_priority_class_name = "medium-priority"
        mlrun.mlconf.valid_function_priority_class_names = medium_priority_class_name
        runtime.with_priority_class(medium_priority_class_name)
        self.execute_function(runtime)
        self._assert_pod_creation_config(
            expected_priority_class_name=medium_priority_class_name
        )

        default_priority_class_name = "default-priority"
        mlrun.mlconf.default_function_priority_class_name = default_priority_class_name
        mlrun.mlconf.valid_function_priority_class_names = ",".join(
            [default_priority_class_name, medium_priority_class_name]
        )
        runtime = self._generate_runtime()

        self.execute_function(runtime)
        self._assert_pod_creation_config(
            expected_priority_class_name=default_priority_class_name
        )

        runtime = self._generate_runtime()

        mlrun.mlconf.valid_function_priority_class_names = ""
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            runtime.with_priority_class(medium_priority_class_name)

    def test_run_with_security_context(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        self.execute_function(runtime)
        self.assert_security_context()

        default_security_context_dict = {
            "runAsUser": 1000,
            "runAsGroup": 3000,
        }
        default_security_context = self._generate_security_context(
            default_security_context_dict["runAsUser"],
            default_security_context_dict["runAsGroup"],
        )

        mlrun.mlconf.function.spec.security_context.default = base64.b64encode(
            json.dumps(default_security_context_dict).encode("utf-8")
        )
        runtime = self._generate_runtime()

        self.execute_function(runtime)
        self.assert_security_context(default_security_context)

        # override default
        other_security_context = self._generate_security_context(
            run_as_group=2000,
        )
        runtime = self._generate_runtime()

        runtime.with_security_context(other_security_context)
        self.execute_function(runtime)
        self.assert_security_context(other_security_context)

        # when enrichment mode is not 'disabled' security context is internally managed
        mlrun.mlconf.function.spec.security_context.enrichment_mode = (
            SecurityContextEnrichmentModes.override.value
        )
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc:
            runtime.with_security_context(other_security_context)
        assert (
            "Security context is handled internally when enrichment mode is not disabled"
            in str(exc.value)
        )

    def test_run_with_mounts(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        # Mount v3io - Set the env variable, so auto_mount() will pick it up and mount v3io
        v3io_access_key = "1111-2222-3333-4444"
        v3io_user = "test-user"
        os.environ["V3IO_ACCESS_KEY"] = v3io_access_key
        os.environ["V3IO_USERNAME"] = v3io_user
        runtime.apply(auto_mount())

        self.execute_function(runtime)
        self._assert_pod_creation_config()
        self._assert_v3io_mount_or_creds_configured(v3io_user, v3io_access_key)

        # Mount a PVC. Create a new runtime so we don't have both v3io and the PVC mounted
        runtime = self._generate_runtime()
        pvc_name = "test-pvc"
        pvc_mount_path = "/volume/mount/path"
        volume_name = "test-volume-name"
        runtime.apply(auto_mount(pvc_name, pvc_mount_path, volume_name))

        self.execute_function(runtime)
        self._assert_pod_creation_config()
        self._assert_pvc_mount_configured(pvc_name, pvc_mount_path, volume_name)

    def test_run_with_k8s_secrets(self, db: Session, k8s_secrets_mock: K8sSecretsMock):
        secret_keys = ["secret1", "secret2", "secret3", "mlrun.internal_secret"]
        secrets = {key: "some-secret-value" for key in secret_keys}

        k8s_secrets_mock.store_project_secrets(self.project, secrets)

        runtime = self._generate_runtime()

        task = self._generate_task()
        task.metadata.project = self.project
        secret_source = {
            "kind": "kubernetes",
            "source": secret_keys,
        }
        task.with_secrets(secret_source["kind"], secret_keys)

        self.execute_function(runtime, runspec=task)

        # We don't expect the internal secret to be visible - the user cannot mount it to the function
        # even if specifically asking for it in with_secrets()
        expected_env_from_secrets = (
            k8s_secrets_mock.get_expected_env_variables_from_secrets(
                self.project, include_internal=False
            )
        )

        self._assert_pod_creation_config(
            expected_secrets=secret_source,
            expected_env_from_secrets=expected_env_from_secrets,
        )

        # Now do the same with auto-mounting of project-secrets, validate internal secret is not visible
        runtime = self._generate_runtime()
        task = self._generate_task()
        task.metadata.project = self.project

        self.execute_function(runtime, runspec=task)
        self._assert_pod_creation_config(
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

        self.execute_function(runtime, runspec=task)

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

        self.execute_function(runtime)
        self._assert_pod_creation_config(expected_code=expected_code)

    def test_set_env(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()
        env = {"MLRUN_LOG_LEVEL": "DEBUG", "IMAGE_HEIGHT": "128"}
        for env_variable in env:
            runtime.set_env(env_variable, env[env_variable])
        self.execute_function(runtime)
        self._assert_pod_creation_config(expected_env=env)

        # set the same env key for a different value and check that the updated one is used
        env2 = {"MLRUN_LOG_LEVEL": "ERROR", "IMAGE_HEIGHT": "128"}
        runtime.set_env("MLRUN_LOG_LEVEL", "ERROR")
        self.execute_function(runtime)
        self._assert_pod_creation_config(expected_env=env2)

    def test_run_with_code_with_file(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        runtime.with_code(from_file=self.code_filename)

        self.execute_function(runtime)
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
        self.execute_function(runtime, runspec=task)
        self._assert_pod_creation_config(expected_labels=labels)

    def test_with_requirements(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()
        runtime.with_requirements(self.requirements_file)
        expected_commands = ["python -m pip install faker python-dotenv"]
        assert (
            deepdiff.DeepDiff(
                expected_commands,
                runtime.spec.build.commands,
                ignore_order=True,
            )
            == {}
        )
