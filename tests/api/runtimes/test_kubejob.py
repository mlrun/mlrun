import base64
import json
import os

import deepdiff
import kubernetes.client as k8s_client
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import mlrun.api.schemas
import mlrun.errors
import mlrun.k8s_utils
from mlrun.config import config as mlconf
from mlrun.platforms import auto_mount
from mlrun.runtimes.kubejob import KubejobRuntime
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

    def test_run_without_specifying_resources(self, db: Session, client: TestClient):
        self.assert_run_without_specifying_resources()

    def test_run_with_node_selection(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        node_name = "some-node-name"
        runtime.with_node_selection(node_name)
        self._execute_run(runtime)
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
        self._execute_run(runtime)
        self._assert_pod_creation_config(expected_node_selector=node_selector)

        runtime = self._generate_runtime()

        node_selector = {
            "label-3": "val3",
            "label-4": "val4",
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

    def test_preemptible_modes_transitions(self, db: Session, client: TestClient):
        # no preemptible nodes tolerations configured, test modes based on affinity/anti-affinity
        node_selector = self._generate_node_selector()
        mlrun.mlconf.preemptible_nodes.node_selector = base64.b64encode(
            json.dumps(node_selector).encode("utf-8")
        )
        mlrun.mlconf.function_defaults.preemption_mode = (
            mlrun.api.schemas.PreemptionModes.prevent.value
        )
        runtime = self._generate_runtime()
        self._execute_run(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_anti_affinity())

        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.constrain.value)
        self._execute_run(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_affinity())

        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.allow.value)
        self._execute_run(runtime)
        self.assert_node_selection()

        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.constrain.value)
        self._execute_run(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_affinity())

        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.prevent.value)
        self._execute_run(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_anti_affinity())

        # set default preemptible tolerations
        tolerations = self._generate_tolerations()
        serialized_tolerations = self.k8s_api.sanitize_for_serialization(tolerations)
        mlrun.mlconf.preemptible_nodes.tolerations = base64.b64encode(
            json.dumps(serialized_tolerations).encode("utf-8")
        )
        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.constrain.value)
        self._execute_run(runtime)
        self.assert_node_selection(
            affinity=self._generate_preemptible_affinity(), tolerations=tolerations
        )

        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.prevent.value)
        self._execute_run(runtime)
        self.assert_node_selection()

        # expects not preemptible tolerations to stay and anti-affinity not to be enriched
        runtime = self._generate_runtime()
        not_preemptible_tolerations = [
            k8s_client.V1Toleration(
                effect="NoSchedule",
                key="notPreemptible",
                operator="Exists",
                toleration_seconds=3600,
            )
        ]
        runtime.with_node_selection(tolerations=not_preemptible_tolerations)
        self._execute_run(runtime)
        self.assert_node_selection(
            tolerations=not_preemptible_tolerations,
        )
        # in this test case we are checking whether when setting anti-affinity of the preemptible nodes in affinity
        # will remove this, we expect this not to be removed because preemtible toleration configuration is set
        runtime = self._generate_runtime()
        not_preemptible_affinity = self._generate_not_preemptible_affinity()
        runtime.with_node_selection(
            tolerations=not_preemptible_tolerations, affinity=not_preemptible_affinity
        )
        self._execute_run(runtime)
        self.assert_node_selection(
            affinity=not_preemptible_affinity,
            tolerations=not_preemptible_tolerations,
        )
        # unset preemptible nodes tolerations and expect anti-affinity to be merged with not preemptible affinity
        mlrun.mlconf.preemptible_nodes.tolerations = base64.b64encode(
            json.dumps([]).encode("utf-8")
        )
        runtime = self._generate_runtime()
        runtime.with_node_selection(
            tolerations=not_preemptible_tolerations, affinity=not_preemptible_affinity
        )
        expected_affinity = self._generate_not_preemptible_affinity()
        expected_affinity.node_affinity.required_during_scheduling_ignored_during_execution.node_selector_terms.extend(
            mlrun.k8s_utils.generate_preemptible_nodes_anti_affinity_terms()
        )
        self.assert_node_selection(
            affinity=not_preemptible_affinity,
            tolerations=not_preemptible_tolerations,
        )

    def test_run_with_prevent_preemptible_mode(self, db: Session, client: TestClient):
        node_selector = self._generate_node_selector()
        mlrun.mlconf.preemptible_nodes.node_selector = base64.b64encode(
            json.dumps(node_selector).encode("utf-8")
        )
        runtime = self._generate_runtime()
        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.prevent.value)
        self._execute_run(runtime)

        self.assert_node_selection(affinity=self._generate_preemptible_anti_affinity())

        # tolerations are set, but expect to stay because those tolerations aren't configured as preemptible tolerations
        runtime = self._generate_runtime()
        runtime.with_node_selection(tolerations=self._generate_tolerations())
        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.prevent.value)
        self._execute_run(runtime)
        self.assert_node_selection(
            affinity=self._generate_preemptible_anti_affinity(),
            tolerations=self._generate_tolerations(),
        )

        # set default preemptible tolerations
        tolerations = self._generate_tolerations()
        serialized_tolerations = self.k8s_api.sanitize_for_serialization(tolerations)
        mlrun.mlconf.preemptible_nodes.tolerations = base64.b64encode(
            json.dumps(serialized_tolerations).encode("utf-8")
        )
        # tolerations are set, expect preemptible tolerations to be removed and anti-affinity not to be set
        runtime = self._generate_runtime()
        runtime.with_node_selection(
            tolerations=self._generate_preemptible_tolerations()
        )
        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.prevent.value)
        self._execute_run(runtime)
        self.assert_node_selection()

    def test_run_with_constrain_preemptible_mode(self, db: Session, client: TestClient):
        node_selector = self._generate_node_selector()
        mlrun.mlconf.preemptible_nodes.node_selector = base64.b64encode(
            json.dumps(node_selector).encode("utf-8")
        )
        runtime = self._generate_runtime()
        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.constrain.value)
        self._execute_run(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_affinity())
        # set default preemptible tolerations
        tolerations = self._generate_tolerations()
        serialized_tolerations = self.k8s_api.sanitize_for_serialization(tolerations)
        mlrun.mlconf.preemptible_nodes.tolerations = base64.b64encode(
            json.dumps(serialized_tolerations).encode("utf-8")
        )
        runtime = self._generate_runtime()
        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.constrain.value)
        self._execute_run(runtime)
        self.assert_node_selection(
            affinity=self._generate_preemptible_affinity(),
            tolerations=self._generate_preemptible_tolerations(),
        )
        # sets different affinity before, expects to override the required_during_scheduling_ignored_during_execution
        runtime = self._generate_runtime()
        runtime.with_node_selection(affinity=self._generate_affinity())
        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.constrain.value)
        self._execute_run(runtime)
        expected_affinity = self._generate_affinity()
        expected_affinity.node_affinity.required_during_scheduling_ignored_during_execution = k8s_client.V1NodeSelector(
            node_selector_terms=mlrun.k8s_utils.generate_preemptible_nodes_affinity_terms(),
        )
        self.assert_node_selection(
            affinity=expected_affinity,
            tolerations=self._generate_preemptible_tolerations(),
        )

    def test_run_with_allow_preemptible_mode(self, db: Session, client: TestClient):
        node_selector = self._generate_node_selector()
        mlrun.mlconf.preemptible_nodes.node_selector = base64.b64encode(
            json.dumps(node_selector).encode("utf-8")
        )
        # without default preemptible tolerations, expecting default to apply
        runtime = self._generate_runtime()
        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.allow.value)
        self._execute_run(runtime)
        self.assert_node_selection()

        # set default preemptible tolerations
        tolerations = self._generate_tolerations()
        serialized_tolerations = self.k8s_api.sanitize_for_serialization(tolerations)
        mlrun.mlconf.preemptible_nodes.tolerations = base64.b64encode(
            json.dumps(serialized_tolerations).encode("utf-8")
        )
        # when allow, preemptible node selector isn't enough to edit the spec yaml, we also need preemptible tolerations
        self.assert_node_selection()

        runtime = self._generate_runtime()
        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.allow.value)
        self._execute_run(runtime)
        self.assert_node_selection(tolerations=self._generate_preemptible_tolerations())

        # with affinity configured, expecting matching label selector to be removed
        affinity = self._generate_affinity()
        # set preemptible label selector in affinity
        affinity.node_affinity.required_during_scheduling_ignored_during_execution.node_selector_terms = [
            k8s_client.V1NodeSelectorTerm(
                match_expressions=mlrun.k8s_utils.generate_preemptible_node_selector_requirements(
                    mlrun.api.schemas.NodeSelectorOperator.node_selector_op_in.value
                )
            )
        ]
        expected_affinity = self._generate_affinity()
        expected_affinity.node_affinity.required_during_scheduling_ignored_during_execution = (
            None
        )
        runtime = self._generate_runtime()
        runtime.with_node_selection(affinity=affinity)
        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.allow.value)
        self._execute_run(runtime)
        self.assert_node_selection(
            tolerations=self._generate_preemptible_tolerations(),
            affinity=expected_affinity,
        )

        # with affinity configured, contains both preemptible label selector and also unrelated label selector,
        # expecting only the preemptible label selector to be removed
        affinity = self._generate_affinity()
        affinity.node_affinity.required_during_scheduling_ignored_during_execution.node_selector_terms.append(
            k8s_client.V1NodeSelectorTerm(
                match_expressions=mlrun.k8s_utils.generate_preemptible_node_selector_requirements(
                    mlrun.api.schemas.NodeSelectorOperator.node_selector_op_in.value
                )
            )
        )
        expected_affinity = self._generate_affinity()
        runtime = self._generate_runtime()
        runtime.with_node_selection(affinity=affinity)
        runtime.with_preemption_mode(mlrun.api.schemas.PreemptionModes.allow)
        self._execute_run(runtime)
        self.assert_node_selection(
            tolerations=self._generate_preemptible_tolerations(),
            affinity=expected_affinity,
        )

    def test_run_with_preemption_mode_without_preemptible_configuration(
        self, db: Session, client: TestClient
    ):
        for test_case in [
            {
                "affinity": False,
                "node_selector": False,
                "node_name": False,
                "tolerations": False,
            },
            {
                "affinity": True,
                "node_selector": True,
                "node_name": False,
                "tolerations": False,
            },
            {
                "affinity": True,
                "node_selector": True,
                "node_name": True,
                "tolerations": False,
            },
            {
                "affinity": True,
                "node_selector": True,
                "node_name": True,
                "tolerations": True,
            },
        ]:
            affinity = (
                self._generate_affinity() if test_case.get("affinity", False) else None
            )
            node_selector = (
                self._generate_node_selector()
                if test_case.get("node_selector", False)
                else None
            )
            node_name = (
                self._generate_node_name()
                if test_case.get("node_name", False)
                else None
            )
            tolerations = (
                self._generate_tolerations()
                if test_case.get("tolerations", False)
                else None
            )
            for preemption_mode in mlrun.api.schemas.PreemptionModes:
                runtime = self._generate_runtime()
                runtime.with_node_selection(
                    node_name=node_name,
                    node_selector=node_selector,
                    affinity=affinity,
                    tolerations=tolerations,
                )
                runtime.with_preemption_mode(mode=preemption_mode.value)
                self._execute_run(runtime)
                self.assert_node_selection(
                    node_name, node_selector, affinity, tolerations
                )

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

    def test_run_with_priority_class_name(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()

        medium_priority_class_name = "medium-priority"
        mlrun.mlconf.valid_function_priority_class_names = medium_priority_class_name
        runtime.with_priority_class(medium_priority_class_name)
        self._execute_run(runtime)
        self._assert_pod_creation_config(
            expected_priority_class_name=medium_priority_class_name
        )

        default_priority_class_name = "default-priority"
        mlrun.mlconf.default_function_priority_class_name = default_priority_class_name
        mlrun.mlconf.valid_function_priority_class_names = ",".join(
            [default_priority_class_name, medium_priority_class_name]
        )
        runtime = self._generate_runtime()

        self._execute_run(runtime)
        self._assert_pod_creation_config(
            expected_priority_class_name=default_priority_class_name
        )

        runtime = self._generate_runtime()

        mlrun.mlconf.valid_function_priority_class_names = ""
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError):
            runtime.with_priority_class(medium_priority_class_name)

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
        self._assert_v3io_mount_or_creds_configured(v3io_user, v3io_access_key)

        # Mount a PVC. Create a new runtime so we don't have both v3io and the PVC mounted
        runtime = self._generate_runtime()
        pvc_name = "test-pvc"
        pvc_mount_path = "/volume/mount/path"
        volume_name = "test-volume-name"
        runtime.apply(auto_mount(pvc_name, pvc_mount_path, volume_name))

        self._execute_run(runtime)
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

        self._execute_run(runtime, runspec=task)

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

        self._execute_run(runtime, runspec=task)
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
                expected_commands,
                runtime.spec.build.commands,
                ignore_order=True,
            )
            == {}
        )
