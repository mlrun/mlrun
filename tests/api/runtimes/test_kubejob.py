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
import copy
import json
import os
import unittest.mock

import deepdiff
import pytest
from fastapi.testclient import TestClient
from mlrun_pipelines.mounts import auto_mount
from sqlalchemy.orm import Session

import mlrun.common.schemas
import mlrun.errors
import mlrun.k8s_utils
import server.api.utils.builder
from mlrun.common.schemas import SecurityContextEnrichmentModes
from mlrun.config import config as mlconf
from mlrun.runtimes.utils import generate_resources
from server.api.utils.singletons.db import get_db
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
        runtime.metadata.project = self.project
        return runtime

    def test_run_without_runspec(self, db: Session, k8s_secrets_mock):
        runtime = self._generate_runtime()
        self.execute_function(runtime)
        self._assert_pod_creation_config()

        params = {"p1": "v1", "p2": 20}
        inputs = {"input1": f"{self.artifact_path}/input1.txt"}

        self.execute_function(runtime, params=params, inputs=inputs)
        self._assert_pod_creation_config(expected_params=params, expected_inputs=inputs)

    def test_run_with_runspec(self, db: Session, k8s_secrets_mock):
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

    def test_run_with_watch_on_server_side(self, db: Session, k8s_secrets_mock):
        runtime = self._generate_runtime()
        with unittest.mock.patch.object(
            mlrun.model.RunObject,
            "logs",
            side_effect=mlrun.errors.MLRunFatalFailureError("should not reach here"),
        ) as logs_mock:
            self._execute_run(runtime, watch=True)
            assert logs_mock.call_count == 0

    def test_run_with_resource_limits_and_requests(self, db: Session, k8s_secrets_mock):
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

    def test_run_with_node_selection(self, db: Session, k8s_secrets_mock):
        runtime = self._generate_runtime()

        node_name = "some-node-name"
        runtime.with_node_selection(node_name)
        self.execute_function(runtime)
        self._assert_pod_creation_config(expected_node_name=node_name)

        runtime = self._generate_runtime()

        invalid_node_selector = {"label-1": "val=1"}
        with pytest.warns(
            Warning,
            match="The node selector you’ve set does not meet the validation rules for the current Kubernetes version",
        ):
            runtime.with_node_selection(node_selector=invalid_node_selector)
        assert runtime.spec.node_selector == {"label-1": "val=1"}

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
        self, db: Session, k8s_secrets_mock
    ):
        self.assert_run_with_preemption_mode_without_preemptible_configuration()

    def test_preemption_mode_with_preemptible_node_selector_without_tolerations(
        self, db: Session, k8s_secrets_mock
    ):
        self.assert_run_preemption_mode_with_preemptible_node_selector_without_preemptible_tolerations()

    def test_preemption_mode_with_preemptible_node_selector_and_tolerations(
        self, db: Session, k8s_secrets_mock
    ):
        self.assert_run_preemption_mode_with_preemptible_node_selector_and_tolerations()

    def test_preemption_mode_with_preemptible_node_selector_and_tolerations_with_extra_settings(
        self, db: Session, k8s_secrets_mock
    ):
        self.assert_run_preemption_mode_with_preemptible_node_selector_and_tolerations_with_extra_settings()

    def test_preemption_mode_with_preemptible_node_selector_without_preemptible_tolerations_with_extra_settings(
        self, db: Session, k8s_secrets_mock
    ):
        self.assert_run_preemption_mode_with_preemptible_node_selector_without_preemptible_tolerations_with_extra_settings()  # noqa: E501

    def test_with_preemption_mode_none_transitions(self, db: Session, k8s_secrets_mock):
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

    def test_run_with_priority_class_name(self, db: Session, k8s_secrets_mock):
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

    @pytest.mark.parametrize(
        "common_node_selector, project_node_selector, user_node_selector, expected_merged_selector",
        [
            # No node selectors provided at any level
            ({}, {}, {}, {}),
            # Only common node selector provided
            ({"zone": "us-east"}, {}, {}, {"zone": "us-east"}),
            # Only project node selector provided
            ({}, {"gpu": "false"}, {}, {"gpu": "false"}),
            # Only function node selector provided
            ({}, {}, {"test": "user"}, {"test": "user"}),
            # Common and project node selectors provided - the common becomes the function node selector
            (
                {"zone": "us-east"},
                {"gpu": "false"},
                {},
                {"zone": "us-east", "gpu": "false"},
            ),
            # Common and user node selectors provided
            (
                {"zone": "us-east"},
                {},
                {"test": "user"},
                {"zone": "us-east", "test": "user"},
            ),
            # Project and user node selectors provided
            ({}, {"gpu": "false"}, {"test": "user"}, {"gpu": "false", "test": "user"}),
            # All node selectors provided
            (
                {"zone": "us-east"},
                {"gpu": "false"},
                {"test": "user"},
                {"gpu": "false", "test": "user"},
            ),
            # Project and user node selectors overlap
            (
                {},
                {"gpu": "false"},
                {"gpu": "true", "zone": "us-south"},
                {"gpu": "true", "zone": "us-south"},
            ),
            # Common, project, and user node selectors all overlap
            (
                {"zone": "us-east", "gpu": "true"},
                {"gpu": "false", "test": "test"},
                {"test": "", "zone": "us-south"},
                {"gpu": "false", "zone": "us-south"},
            ),
        ],
    )
    def test_merge_node_selector(
        self,
        db: Session,
        k8s_secrets_mock,
        common_node_selector,
        project_node_selector,
        user_node_selector,
        expected_merged_selector,
    ):
        mlrun.mlconf.default_function_node_selector = base64.b64encode(
            json.dumps(common_node_selector).encode("utf-8")
        )

        runtime = self._generate_runtime()
        if user_node_selector:
            runtime.spec.node_selector = user_node_selector

        project = runtime._get_db().get_project(runtime.metadata.project)
        project.spec.default_function_node_selector = project_node_selector
        runtime._get_db().store_project(self.project, project)

        self.execute_function(runtime)
        self._assert_pod_creation_config(
            expected_node_selector=expected_merged_selector
        )

    def test_set_annotation(self, db: Session, k8s_secrets_mock):
        runtime = self._generate_runtime()
        runtime.with_annotations({"annotation-key": "annotation-value"})
        self.execute_function(runtime)

        pod = self._get_pod_creation_args()
        assert pod.metadata.annotations.get("annotation-key") == "annotation-value"

    def test_run_with_security_context(self, db: Session, k8s_secrets_mock):
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

    def test_run_with_mounts(self, db: Session, k8s_secrets_mock):
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

    def test_run_with_global_secrets(
        self, db: Session, k8s_secrets_mock: K8sSecretsMock
    ):
        project_secret_keys = ["secret1", "secret2", "secret3", "mlrun.internal_secret"]
        project_secrets = {key: "some-secret-value" for key in project_secret_keys}
        # secret1 is included both in the global secrets and the project secrets, it should have the value from the
        # project-secret (this is the logic in get_expected_env_variables_from_secrets)
        global_secret_keys = [
            "global_secret1",
            "global_secret2",
            "mlrun.global_secret3",
            "secret1",
        ]
        global_secrets = {key: "some-global-secret-value" for key in global_secret_keys}
        global_secret_name = "global-secret-1"

        k8s_secrets_mock.store_project_secrets(self.project, project_secrets)
        k8s_secrets_mock.store_secret(global_secret_name, global_secrets)

        mlconf.secret_stores.kubernetes.global_function_env_secret_name = (
            global_secret_name
        )
        runtime = self._generate_runtime()

        self.execute_function(runtime)

        mlconf.secret_stores.kubernetes.global_function_env_secret_name = None

        expected_env_from_secrets = (
            k8s_secrets_mock.get_expected_env_variables_from_secrets(
                self.project, include_internal=False, global_secret=global_secret_name
            )
        )

        self._assert_pod_creation_config(
            expected_env_from_secrets=expected_env_from_secrets,
        )

    # TODO: Vault: uncomment when vault returns to be relevant
    # def test_run_with_vault_secrets(self, db: Session, client: TestClient):
    #     self._mock_vault_functionality()
    #     runtime = self._generate_runtime()
    #
    #     task = self._generate_task()
    #
    #     task.metadata.project = self.project
    #     secret_source = {
    #         "kind": "vault",
    #         "source": {"project": self.project, "secrets": self.vault_secrets},
    #     }
    #     task.with_secrets(secret_source["kind"], self.vault_secrets)
    #     vault_url = "/url/for/vault"
    #     mlconf.secret_stores.vault.remote_url = vault_url
    #     mlconf.secret_stores.vault.token_path = vault_url
    #
    #     self.execute_function(runtime, runspec=task)
    #
    #     self._assert_pod_creation_config(
    #         expected_secrets=secret_source,
    #         expected_env={
    #             "MLRUN_SECRET_STORES__VAULT__ROLE": f"project:{self.project}",
    #             "MLRUN_SECRET_STORES__VAULT__URL": vault_url,
    #         },
    #     )
    #
    #     self._assert_secret_mount(
    #         "vault-secret", self.vault_secret_name, 420, vault_url
    #     )

    def test_run_with_code(self, db: Session, k8s_secrets_mock):
        runtime = self._generate_runtime()

        expected_code = """
def my_func(context):
    print("Hello cruel world")
        """
        runtime.with_code(body=expected_code)

        self.execute_function(runtime)
        self._assert_pod_creation_config(
            expected_code=expected_code,
            expected_args=[
                "run",
                "--name",
                "test-function",
                "--from-env",
            ],
        )

    def test_set_env(self, db: Session, k8s_secrets_mock):
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

    def test_run_with_code_with_file(self, db: Session, k8s_secrets_mock):
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

    def test_run_with_args(self, db: Session, k8s_secrets_mock):
        runtime = self._generate_runtime()
        runtime.spec.args = ["--arg1", "value1"]

        self.execute_function(runtime)
        self._assert_pod_creation_config(
            expected_args=[
                "run",
                "--name",
                "test-function",
                "--from-env",
                "*",
                "--arg1",
                "value1",
            ],
        )

    def test_set_label(self, db: Session, k8s_secrets_mock):
        task = self._generate_task()
        task.set_label("category", "test")
        labels = {"category": "test"}

        runtime = self._generate_runtime()
        self.execute_function(runtime, runspec=task)
        self._assert_pod_creation_config(expected_labels=labels)

    def test_with_image_pull_configuration(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()
        policy = "IfNotPresent"
        secret = "some_secret"
        runtime.set_image_pull_configuration(
            image_pull_policy=policy, image_pull_secret_name=secret
        )
        assert (
            runtime.spec.image_pull_policy == policy
            and runtime.spec.image_pull_secret == secret
        )

        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Image pull policy must be one of",
        ):
            runtime.set_image_pull_configuration(image_pull_policy="invalidPolicy")

    def test_with_requirements(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()
        runtime.with_requirements(requirements_file=self.requirements_file)
        expected_requirements = ["faker", "python-dotenv", "chardet>=3.0.2, <4.0"]
        assert (
            deepdiff.DeepDiff(
                expected_requirements,
                runtime.spec.build.requirements,
                ignore_order=True,
            )
            == {}
        )

    def test_with_commands(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()
        runtime.with_commands(["pip install pandas", "pip install numpy"])
        expected_commands = ["pip install pandas", "pip install numpy"]
        assert (
            deepdiff.DeepDiff(
                expected_commands,
                runtime.spec.build.commands,
                ignore_order=False,
            )
            == {}
        )
        runtime.with_commands(["pip install scikit-learn"])
        expected_commands = [
            "pip install pandas",
            "pip install numpy",
            "pip install scikit-learn",
        ]
        assert (
            deepdiff.DeepDiff(
                expected_commands,
                runtime.spec.build.commands,
                ignore_order=False,
            )
            == {}
        )
        runtime.with_commands(["pip install tensorflow", "pip install pandas"])
        expected_commands = [
            "pip install pandas",
            "pip install numpy",
            "pip install scikit-learn",
            "pip install tensorflow",
        ]
        assert (
            deepdiff.DeepDiff(
                expected_commands,
                runtime.spec.build.commands,
                ignore_order=False,
            )
            == {}
        )
        runtime.with_commands(
            ["pip install tensorflow", "pip install pandas"], overwrite=True
        )
        expected_commands = ["pip install tensorflow", "pip install pandas"]
        assert (
            deepdiff.DeepDiff(
                expected_commands,
                runtime.spec.build.commands,
                ignore_order=False,
            )
            == {}
        )

    def test_build_config(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()
        runtime.build_config(
            base_image="mlrun/mlrun", commands=["python -m pip install pandas"]
        )
        expected_commands = ["python -m pip install pandas"]
        assert (
            deepdiff.DeepDiff(
                expected_commands,
                runtime.spec.build.commands,
                ignore_order=True,
            )
            == {}
        )

        assert runtime.spec.build.base_image == "mlrun/mlrun"

        runtime.build_config(commands=["python -m pip install numpy"])
        expected_commands = [
            "python -m pip install pandas",
            "python -m pip install numpy",
        ]
        assert (
            deepdiff.DeepDiff(
                expected_commands,
                runtime.spec.build.commands,
                ignore_order=False,
            )
            == {}
        )

        runtime.build_config(
            commands=["python -m pip install scikit-learn"], overwrite=True
        )
        expected_commands = ["python -m pip install scikit-learn"]
        assert (
            deepdiff.DeepDiff(
                expected_commands,
                runtime.spec.build.commands,
                ignore_order=True,
            )
            == {}
        )

        runtime.build_config(requirements=["pandas", "numpy"])
        expected_requirements = [
            "pandas",
            "numpy",
        ]
        assert (
            deepdiff.DeepDiff(
                expected_requirements,
                runtime.spec.build.requirements,
                ignore_order=False,
            )
            == {}
        )
        expected_commands = ["python -m pip install scikit-learn"]
        assert (
            deepdiff.DeepDiff(
                expected_commands,
                runtime.spec.build.commands,
                ignore_order=True,
            )
            == {}
        )

        runtime.build_config(requirements=["scikit-learn"], overwrite=True)
        expected_requirements = ["scikit-learn"]
        assert (
            deepdiff.DeepDiff(
                expected_requirements,
                runtime.spec.build.requirements,
                ignore_order=True,
            )
            == {}
        )

    def test_build_config_commands_and_requirements_order(
        self, db: Session, client: TestClient
    ):
        runtime = self._generate_runtime()
        runtime.build_config(commands=["apt-get update"], requirements=["scikit-learn"])
        expected_commands = ["apt-get update"]
        expected_requirements = ["scikit-learn"]
        assert (
            deepdiff.DeepDiff(
                expected_commands,
                runtime.spec.build.commands,
                ignore_order=False,
            )
            == {}
        )
        assert (
            deepdiff.DeepDiff(
                expected_requirements,
                runtime.spec.build.requirements,
                ignore_order=False,
            )
            == {}
        )

    def test_build_config_with_images(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()
        runtime.build_config(base_image="mlrun/mlrun", image="target/mlrun")
        assert runtime.spec.build.base_image == "mlrun/mlrun"
        assert runtime.spec.build.image == "target/mlrun"

        runtime = self._generate_runtime()
        runtime.build_config(image="target/mlrun")
        assert runtime.spec.build.image == "target/mlrun"

    @pytest.mark.parametrize(
        "with_mlrun, commands, expected_to_upgrade",
        [
            (True, [], True),
            (False, ["some command"], False),
            (False, ["python -m pip install pip"], False),
            (True, ["python -m pip install --upgrade pip~=22.0"], False),
            (True, ["python -m pip install --upgrade pandas"], True),
        ],
    )
    def test_deploy_upgrade_pip(
        self,
        db: Session,
        client: TestClient,
        with_mlrun,
        commands,
        expected_to_upgrade,
    ):
        mlrun.mlconf.httpdb.builder.docker_registry = "localhost:5000"
        with unittest.mock.patch(
            "server.api.utils.builder.make_kaniko_pod", unittest.mock.MagicMock()
        ):
            runtime = self._generate_runtime()
            runtime.spec.build.base_image = "some/image"
            runtime.spec.build.commands = copy.deepcopy(commands)
            self.deploy(db, runtime, with_mlrun=with_mlrun)
            dockerfile = server.api.utils.builder.make_kaniko_pod.call_args[1][
                "dockertext"
            ]
            if expected_to_upgrade:
                expected_str = ""
                if commands:
                    expected_str += "\nRUN "
                    expected_str += "\nRUN ".join(commands)
                expected_str += (
                    f"\nRUN python -m pip install "
                    f"--upgrade pip{mlrun.mlconf.httpdb.builder.pip_version}"
                )

                # assert that mlrun was added to the requirements file
                if with_mlrun:
                    expected_str += (
                        "\nRUN echo 'Installing /empty/requirements.txt...'; cat /empty/requirements.txt"
                        "\nRUN python -m pip install -r /empty/requirements.txt"
                    )
                    kaniko_pod_requirements = (
                        server.api.utils.builder.make_kaniko_pod.call_args[1][
                            "requirements"
                        ]
                    )
                    assert kaniko_pod_requirements == [
                        "mlrun[complete] @ git+https://github.com/mlrun/mlrun@development"
                    ]
                assert expected_str in dockerfile
            else:
                assert (
                    f"pip install --upgrade pip{mlrun.mlconf.httpdb.builder.pip_version}"
                    not in dockerfile
                )

    @pytest.mark.parametrize(
        "with_mlrun, requirements, with_requirements_file, expected_requirements",
        [
            (
                True,
                [],
                False,
                ["mlrun[complete] @ git+https://github.com/mlrun/mlrun@development"],
            ),
            (
                True,
                ["pandas"],
                False,
                [
                    "mlrun[complete] @ git+https://github.com/mlrun/mlrun@development",
                    "pandas",
                ],
            ),
            (
                True,
                ["pandas", "tensorflow"],
                False,
                [
                    "mlrun[complete] @ git+https://github.com/mlrun/mlrun@development",
                    "pandas",
                    "tensorflow",
                ],
            ),
            (False, [], True, ["faker", "python-dotenv", "chardet>=3.0.2, <4.0"]),
            (False, ["pandas", "tensorflow"], False, ["pandas", "tensorflow"]),
            (
                False,
                ["pandas", "tensorflow"],
                True,
                [
                    "faker",
                    "python-dotenv",
                    "chardet>=3.0.2, <4.0",
                    "pandas",
                    "tensorflow",
                ],
            ),
            (
                True,
                ["pandas", "tensorflow"],
                True,
                [
                    "mlrun[complete] @ git+https://github.com/mlrun/mlrun@development",
                    "faker",
                    "python-dotenv",
                    "chardet>=3.0.2, <4.0",
                    "pandas",
                    "tensorflow",
                ],
            ),
            (
                True,
                [],
                True,
                [
                    "mlrun[complete] @ git+https://github.com/mlrun/mlrun@development",
                    "faker",
                    "python-dotenv",
                    "chardet>=3.0.2, <4.0",
                ],
            ),
        ],
    )
    def test_deploy_with_mlrun(
        self,
        db: Session,
        client: TestClient,
        with_mlrun,
        requirements,
        with_requirements_file,
        expected_requirements,
    ):
        mlrun.mlconf.httpdb.builder.docker_registry = "localhost:5000"
        with unittest.mock.patch(
            "server.api.utils.builder.make_kaniko_pod", unittest.mock.MagicMock()
        ):
            runtime = self._generate_runtime()
            runtime.spec.build.base_image = "some/image"

            requirements_file = (
                "" if not with_requirements_file else self.requirements_file
            )
            runtime.with_requirements(
                requirements=requirements, requirements_file=requirements_file
            )

            self.deploy(db, runtime, with_mlrun=with_mlrun)
            dockerfile = server.api.utils.builder.make_kaniko_pod.call_args[1][
                "dockertext"
            ]

            install_requirements_commands = (
                "\nRUN echo 'Installing /empty/requirements.txt...'; cat /empty/requirements.txt"
                "\nRUN python -m pip install -r /empty/requirements.txt"
            )
            kaniko_pod_requirements = (
                server.api.utils.builder.make_kaniko_pod.call_args[1]["requirements"]
            )
            if with_mlrun:
                expected_str = f"\nRUN python -m pip install --upgrade pip{mlrun.mlconf.httpdb.builder.pip_version}"
                expected_str += install_requirements_commands
                assert kaniko_pod_requirements == expected_requirements
                assert expected_str in dockerfile

            else:
                assert (
                    f"pip install --upgrade pip{mlrun.mlconf.httpdb.builder.pip_version}"
                    not in dockerfile
                )

                # assert that install requirements commands are in the dockerfile
                if with_requirements_file or requirements:
                    expected_str = install_requirements_commands
                    assert expected_str in dockerfile

                # assert mlrun is not in the requirements
                for requirement in kaniko_pod_requirements:
                    assert "mlrun" not in requirement

                assert kaniko_pod_requirements == expected_requirements

    @pytest.mark.parametrize(
        "workdir, source, pull_at_runtime, target_dir, expected_workdir",
        [
            ("", "git://bla", True, None, None),
            ("", "git://bla", False, None, None),
            ("", "git://bla", False, "/a/b/c", "/a/b/c/"),
            ("subdir", "git://bla", False, "/a/b/c", "/a/b/c/subdir"),
            ("./subdir", "git://bla", False, "/a/b/c", "/a/b/c/subdir"),
            ("./subdir", "git://bla", True, "/a/b/c", None),
            ("/abs/subdir", "git://bla", False, "/a/b/c", "/abs/subdir"),
            ("/abs/subdir", "git://bla", False, None, "/abs/subdir"),
        ],
    )
    def test_resolve_workdir(
        self,
        workdir,
        source,
        pull_at_runtime,
        target_dir,
        expected_workdir,
        k8s_secrets_mock,
    ):
        runtime = self._generate_runtime()
        runtime.with_source_archive(
            source, workdir, pull_at_runtime=pull_at_runtime, target_dir=target_dir
        )

        # mock the build
        runtime.spec.image = "some/image"
        self.execute_function(runtime)
        pod = self._get_pod_creation_args()
        assert pod.spec.containers[0].working_dir == expected_workdir

    def test_with_source_archive_validation(self):
        runtime = self._generate_runtime()
        source = "./some/relative/path"
        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as e:
            runtime.with_source_archive(source, pull_at_runtime=False)
        assert (
            f"Source '{source}' must be a valid URL or absolute path when 'pull_at_runtime' is False "
            "set 'source' to a remote URL to clone/copy the source to the base image, "
            "or set 'pull_at_runtime' to True to pull the source at runtime."
            in str(e.value)
        )

    @pytest.mark.parametrize(
        "mode, command, args, params, expected_command, expected_args",
        [
            ("pass", None, None, None, None, []),
            ("pass", "echo", ["1"], None, ["echo"], ["1"]),
            (
                "pass",
                "echo {param_1}",
                ["1"],
                {"param_1": "param"},
                ["echo param"],
                ["1"],
            ),
            (
                "",
                "echo {param_1}",
                ["1"],
                {"param_1": "param"},
                ["mlrun"],
                ["run", "--name", "test-function", "--from-env", "echo {param_1}", "1"],
            ),
        ],
    )
    def test_run_mode(
        self,
        db: Session,
        k8s_secrets_mock,
        mode,
        command,
        args,
        params,
        expected_command,
        expected_args,
    ):
        runtime = self._generate_runtime()
        runtime.spec.mode = mode
        runtime.spec.command = command
        runtime.spec.args = args

        self.execute_function(runtime, params=params)

        pod = self._get_pod_creation_args()
        container_spec = pod.spec.containers[0]
        assert container_spec.command == expected_command
        assert container_spec.args == expected_args

    def test_state_thresholds_defaults(self, db: Session, k8s_secrets_mock):
        runtime = self._generate_runtime()
        self.execute_function(runtime)
        run = get_db().list_runs(db, project=self.project)[0]
        assert (
            run["spec"]["state_thresholds"]
            == mlrun.mlconf.function.spec.state_thresholds.default.to_dict()
        )

    def test_set_state_thresholds_success(self, db: Session, k8s_secrets_mock):
        state_thresholds = {
            "pending_not_scheduled": "1000s",
            "pending_scheduled": "1day 20m",
            "executing": "30h 19 min",
            "image_pull_backoff": "-1",
        }

        runtime = self._generate_runtime()
        runtime.set_state_thresholds(
            state_thresholds=state_thresholds,
        )
        self.execute_function(runtime)
        run = get_db().list_runs(db, project=self.project)[0]
        assert run["spec"]["state_thresholds"] == state_thresholds

        override_state_thresholds = {
            "pending_not_scheduled": "250s",
            "executing": "40h 19 min",
        }
        runtime.set_state_thresholds(
            state_thresholds=override_state_thresholds,
            patch=False,
        )
        self.execute_function(runtime)
        run = get_db().list_runs(db, project=self.project)[0]
        expected_state_thresholds = override_state_thresholds
        expected_state_thresholds["image_pull_backoff"] = (
            mlconf.function.spec.state_thresholds.default.image_pull_backoff
        )
        expected_state_thresholds["pending_scheduled"] = (
            mlconf.function.spec.state_thresholds.default.pending_scheduled
        )
        assert run["spec"]["state_thresholds"] == expected_state_thresholds

        patch_state_thresholds = {
            "pending_not_scheduled": "370s",
            "image_pull_backoff": "50h 19 min",
        }
        runtime.set_state_thresholds(
            state_thresholds=patch_state_thresholds,
        )
        self.execute_function(runtime)
        run = get_db().list_runs(db, project=self.project)[0]
        expected_state_thresholds = patch_state_thresholds
        expected_state_thresholds["executing"] = override_state_thresholds["executing"]
        expected_state_thresholds["pending_scheduled"] = (
            mlconf.function.spec.state_thresholds.default.pending_scheduled
        )
        assert run["spec"]["state_thresholds"] == expected_state_thresholds

    @staticmethod
    def _assert_build_commands(expected_commands, runtime):
        assert (
            deepdiff.DeepDiff(
                expected_commands,
                runtime.spec.build.commands,
                ignore_order=True,
            )
            == {}
        )
