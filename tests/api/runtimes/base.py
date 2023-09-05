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
import pathlib
import sys
import typing
import unittest.mock
from base64 import b64encode
from copy import deepcopy
from datetime import datetime, timezone

import deepdiff
import fastapi.testclient
import pytest
import sqlalchemy.orm
from kubernetes import client
from kubernetes import client as k8s_client
from kubernetes.client import V1EnvVar

import mlrun.api.api.endpoints.functions
import mlrun.api.crud
import mlrun.common.schemas
import mlrun.k8s_utils
import mlrun.runtimes.pod
import tests.api.api.utils
from mlrun.api.utils.singletons.k8s import get_k8s_helper
from mlrun.config import config as mlconf
from mlrun.model import new_task
from mlrun.runtimes.constants import PodPhases
from mlrun.utils import create_logger
from mlrun.utils.azure_vault import AzureVaultStore

logger = create_logger(level="debug", name="test-runtime")


class TestRuntimeBase:
    def setup_method(self, method):
        self.namespace = mlconf.namespace = "test-namespace"
        get_k8s_helper().namespace = self.namespace

        # set auto-mount to work as if this is an Iguazio system (otherwise it may try to mount PVC)
        mlconf.igz_version = "1.1.1"
        mlconf.storage.auto_mount_type = "auto"
        mlconf.storage.auto_mount_params = ""

        self._logger = logger
        self.project = "test-project"
        self.name = "test-function"
        self.run_uid = "test_run_uid"
        self.image_name = "mlrun/mlrun:latest"
        self.artifact_path = "/tmp"
        self.function_name_label = "mlrun/name"
        self.code_filename = str(self.assets_path / "sample_function.py")
        self.requirements_file = str(self.assets_path / "requirements.txt")

        self.vault_secrets = ["secret1", "secret2", "AWS_KEY"]
        # TODO: Vault: uncomment when vault returns to be relevant
        # self.vault_secret_value = "secret123!@"
        # self.vault_secret_name = "vault-secret"

        self.azure_vault_secrets = ["azure_secret1", "azure_secret2"]
        self.azure_secret_value = "azure-secret-123!@"
        self.azure_vault_secret_name = "k8s-vault-secret"

        self.k8s_api = k8s_client.ApiClient()

        self._logger.info(
            f"Setting up test {self.__class__.__name__}::{method.__name__}"
        )

        self.custom_setup()

        self._logger.info(
            f"Finished setting up test {self.__class__.__name__}::{method.__name__}"
        )

    @pytest.fixture(autouse=True)
    def setup_method_fixture(
        self, db: sqlalchemy.orm.Session, client: fastapi.testclient.TestClient
    ):
        # We want this mock for every test, ideally we would have simply put it in the setup_method
        # but it is happening before the fixtures initialization. We need the client fixture (which needs the db one)
        # in order to be able to mock k8s stuff
        get_k8s_helper().get_project_secret_keys = unittest.mock.Mock(return_value=[])
        get_k8s_helper().v1api = unittest.mock.Mock()
        get_k8s_helper().crdapi = unittest.mock.Mock()
        get_k8s_helper().is_running_inside_kubernetes_cluster = unittest.mock.Mock(
            return_value=True
        )
        self._create_project(client)
        # enable inheriting classes to do the same
        self.custom_setup_after_fixtures()

    def teardown_method(self, method):
        self._logger.info(
            f"Tearing down test {self.__class__.__name__}::{method.__name__}"
        )

        self.custom_teardown()

        self._logger.info(
            f"Finished tearing down test {self.__class__.__name__}::{method.__name__}"
        )

    @property
    def assets_path(self):
        return (
            pathlib.Path(sys.modules[self.__module__].__file__).absolute().parent
            / "assets"
        )

    def _generate_runtime(
        self,
    ) -> typing.Union[
        mlrun.runtimes.MpiRuntimeV1Alpha1,
        mlrun.runtimes.MpiRuntimeV1,
        mlrun.runtimes.RemoteRuntime,
        mlrun.runtimes.ServingRuntime,
        mlrun.runtimes.DaskCluster,
        mlrun.runtimes.KubejobRuntime,
        mlrun.runtimes.LocalRuntime,
        mlrun.runtimes.Spark3Runtime,
        mlrun.runtimes.RemoteSparkRuntime,
    ]:
        pass

    def custom_setup(self):
        pass

    def custom_setup_after_fixtures(self):
        pass

    def custom_teardown(self):
        pass

    def _create_project(
        self, client: fastapi.testclient.TestClient, project_name: str = None
    ):
        tests.api.api.utils.create_project(client, project_name or self.project)

    def _generate_task(self):
        return new_task(
            name=self.name, project=self.project, artifact_path=self.artifact_path
        )

    def _generate_preemptible_tolerations(self) -> typing.List[k8s_client.V1Toleration]:
        return mlrun.k8s_utils.generate_preemptible_tolerations()

    def _generate_tolerations(self):
        return [self._generate_toleration()]

    def _generate_toleration(self):
        return k8s_client.V1Toleration(
            effect="NoSchedule",
            key="test1",
            operator="Exists",
            toleration_seconds=3600,
        )

    def _generate_node_selector(self):
        return {
            "label-1": "val1",
            "label-2": "val2",
        }

    def _generate_node_name(self):
        return "node-name"

    def _generate_preemptible_anti_affinity(self):
        return k8s_client.V1Affinity(
            node_affinity=k8s_client.V1NodeAffinity(
                required_during_scheduling_ignored_during_execution=k8s_client.V1NodeSelector(
                    node_selector_terms=mlrun.k8s_utils.generate_preemptible_nodes_anti_affinity_terms(),
                ),
            ),
        )

    def _generate_preemptible_affinity(self):
        return k8s_client.V1Affinity(
            node_affinity=k8s_client.V1NodeAffinity(
                required_during_scheduling_ignored_during_execution=k8s_client.V1NodeSelector(
                    node_selector_terms=mlrun.k8s_utils.generate_preemptible_nodes_affinity_terms(),
                ),
            ),
        )

    def _generate_not_preemptible_tolerations(self):
        return [
            k8s_client.V1Toleration(
                effect="NoSchedule",
                key="not-preemptible",
                operator="Exists",
                toleration_seconds=3600,
            )
        ]

    def _generate_not_preemptible_affinity(self):
        return k8s_client.V1Affinity(
            node_affinity=k8s_client.V1NodeAffinity(
                required_during_scheduling_ignored_during_execution=k8s_client.V1NodeSelector(
                    node_selector_terms=[
                        k8s_client.V1NodeSelectorTerm(
                            match_expressions=[
                                k8s_client.V1NodeSelectorRequirement(
                                    key="not_preemptible_node",
                                    operator="In",
                                    values=[
                                        "not_preemptible_required-label-value-1",
                                        "not_preemptible_required-label-value-2",
                                    ],
                                )
                            ]
                        ),
                    ]
                )
            )
        )

    def _generate_affinity(self) -> k8s_client.V1Affinity:
        return k8s_client.V1Affinity(
            node_affinity=k8s_client.V1NodeAffinity(
                preferred_during_scheduling_ignored_during_execution=[
                    k8s_client.V1PreferredSchedulingTerm(
                        weight=1,
                        preference=k8s_client.V1NodeSelectorTerm(
                            match_expressions=[
                                k8s_client.V1NodeSelectorRequirement(
                                    key="some_node_label",
                                    operator="In",
                                    values=[
                                        "possible-label-value-1",
                                        "possible-label-value-2",
                                    ],
                                )
                            ]
                        ),
                    )
                ],
                required_during_scheduling_ignored_during_execution=k8s_client.V1NodeSelector(
                    node_selector_terms=[
                        k8s_client.V1NodeSelectorTerm(
                            match_expressions=[
                                k8s_client.V1NodeSelectorRequirement(
                                    key="some_node_label",
                                    operator="In",
                                    values=[
                                        "required-label-value-1",
                                        "required-label-value-2",
                                    ],
                                )
                            ]
                        ),
                    ]
                ),
            ),
            pod_affinity=k8s_client.V1PodAffinity(
                required_during_scheduling_ignored_during_execution=[
                    k8s_client.V1PodAffinityTerm(
                        label_selector=k8s_client.V1LabelSelector(
                            match_labels={"some-pod-label-key": "some-pod-label-value"}
                        ),
                        namespaces=["namespace-a", "namespace-b"],
                        topology_key="key-1",
                    )
                ]
            ),
            pod_anti_affinity=k8s_client.V1PodAntiAffinity(
                preferred_during_scheduling_ignored_during_execution=[
                    k8s_client.V1WeightedPodAffinityTerm(
                        weight=1,
                        pod_affinity_term=k8s_client.V1PodAffinityTerm(
                            label_selector=k8s_client.V1LabelSelector(
                                match_expressions=[
                                    k8s_client.V1LabelSelectorRequirement(
                                        key="some_pod_label",
                                        operator="NotIn",
                                        values=[
                                            "forbidden-label-value-1",
                                            "forbidden-label-value-2",
                                        ],
                                    )
                                ]
                            ),
                            namespaces=["namespace-c"],
                            topology_key="key-2",
                        ),
                    )
                ]
            ),
        )

    def _generate_security_context(
        self,
        run_as_user: typing.Optional[int] = None,
        run_as_group: typing.Optional[int] = None,
    ) -> k8s_client.V1SecurityContext:
        return k8s_client.V1SecurityContext(
            run_as_user=run_as_user,
            run_as_group=run_as_group,
        )

    def _mock_create_namespaced_pod(self):
        def _generate_pod(namespace, pod):
            terminated_container_state = client.V1ContainerStateTerminated(
                finished_at=datetime.now(timezone.utc), exit_code=0
            )
            container_state = client.V1ContainerState(
                terminated=terminated_container_state
            )
            container_status = client.V1ContainerStatus(
                state=container_state,
                image=self.image_name,
                image_id="must-provide-image-id",
                name=self.name,
                ready=True,
                restart_count=0,
            )
            status = client.V1PodStatus(
                phase=PodPhases.succeeded, container_statuses=[container_status]
            )
            response_pod = deepcopy(pod)
            response_pod.status = status
            response_pod.metadata.name = "test-pod"
            response_pod.metadata.namespace = namespace
            return response_pod

        get_k8s_helper().v1api.create_namespaced_pod = unittest.mock.Mock(
            side_effect=_generate_pod
        )

        self._mock_get_logger_pods()

    def _mock_get_logger_pods(self):
        # Our purpose is not to test the client watching on logs, mock empty list (used in get_logger_pods)
        get_k8s_helper().v1api.list_namespaced_pod = unittest.mock.Mock(
            return_value=client.V1PodList(items=[])
        )
        get_k8s_helper().v1api.read_namespaced_pod_log = unittest.mock.Mock(
            return_value="Mocked pod logs"
        )

    def _mock_create_namespaced_custom_object(self):
        def _generate_custom_object(
            group: str,
            version: str,
            namespace: str,
            plural: str,
            body: object,
            **kwargs,
        ):
            return deepcopy(body)

        get_k8s_helper().crdapi.create_namespaced_custom_object = unittest.mock.Mock(
            side_effect=_generate_custom_object
        )
        self._mock_get_logger_pods()

    # Vault now supported in KubeJob and Serving, so moved to base.
    def _mock_vault_functionality(self):
        # TODO: Vault: uncomment when vault returns to be relevant
        # secret_dict = {key: self.vault_secret_value for key in self.vault_secrets}
        # VaultStore.get_secrets = unittest.mock.Mock(return_value=secret_dict)

        azure_secret_dict = {
            key: self.azure_secret_value for key in self.azure_vault_secrets
        }
        AzureVaultStore.get_secrets = unittest.mock.Mock(return_value=azure_secret_dict)

        object_meta = client.V1ObjectMeta(
            name="test-service-account", namespace=self.namespace
        )
        secret = client.V1ObjectReference(
            name=self.vault_secret_name, namespace=self.namespace
        )
        service_account = client.V1ServiceAccount(
            metadata=object_meta, secrets=[secret]
        )
        get_k8s_helper().v1api.read_namespaced_service_account = unittest.mock.Mock(
            return_value=service_account
        )

    def execute_function(self, runtime, **kwargs):
        # simulating sending to API - serialization through dict
        runtime = runtime.from_dict(runtime.to_dict())
        # set watch to False, to mimic the API behavior (API doesn't watch on the job)
        kwargs.update({"watch": False})
        self._execute_run(runtime, **kwargs)

    @staticmethod
    def deploy(db_session, runtime, with_mlrun=True):
        auth_info = mlrun.common.schemas.AuthInfo()
        mlrun.api.api.endpoints.functions._build_function(
            db_session, auth_info, runtime, with_mlrun=with_mlrun
        )

    def _reset_mocks(self):
        get_k8s_helper().v1api.create_namespaced_pod.reset_mock()
        get_k8s_helper().v1api.list_namespaced_pod.reset_mock()
        get_k8s_helper().v1api.read_namespaced_pod_log.reset_mock()

    def _reset_custom_object_mocks(self):
        mlrun.api.utils.singletons.k8s.get_k8s_helper().crdapi.create_namespaced_custom_object.reset_mock()
        get_k8s_helper().v1api.list_namespaced_pod.reset_mock()

    def _execute_run(self, runtime, **kwargs):
        # Reset the mock, so that when checking is create_pod was called, no leftovers are there (in case running
        # multiple runs in the same test)
        self._reset_mocks()

        runtime.run(
            name=self.name,
            project=self.project,
            artifact_path=self.artifact_path,
            auth_info=mlrun.common.schemas.AuthInfo(),
            **kwargs,
        )

    def _assert_labels(self, labels: dict, expected_class_name):
        expected_labels = {
            "mlrun/class": expected_class_name,
            self.function_name_label: self.name,
            "mlrun/project": self.project,
            "mlrun/tag": "latest",
        }

        for key in expected_labels:
            assert labels[key] == expected_labels[key]

    def _assert_function_config(
        self,
        config,
        expected_params,
        expected_inputs,
        expected_hyper_params,
        expected_secrets,
        expected_labels,
    ):
        function_metadata = config["metadata"]
        assert function_metadata["name"] == self.name
        assert function_metadata["project"] == self.project

        function_spec = config["spec"]
        assert function_spec["output_path"] == self.artifact_path
        if expected_params:
            assert (
                deepdiff.DeepDiff(
                    function_spec["parameters"], expected_params, ignore_order=True
                )
                == {}
            )
        if expected_inputs:
            assert (
                deepdiff.DeepDiff(
                    function_spec["inputs"], expected_inputs, ignore_order=True
                )
                == {}
            )
        if expected_hyper_params:
            assert (
                deepdiff.DeepDiff(
                    function_spec["hyperparams"],
                    expected_hyper_params,
                    ignore_order=True,
                )
                == {}
            )
        if expected_secrets:
            assert (
                deepdiff.DeepDiff(
                    function_spec["secret_sources"],
                    [expected_secrets],
                    ignore_order=True,
                )
                == {}
            )
        if expected_labels:
            diff_result = deepdiff.DeepDiff(
                function_metadata["labels"],
                expected_labels,
                ignore_order=True,
            )
            # We just care that the values we look for are fully there.
            diff_result.pop("dictionary_item_removed", None)
            assert diff_result == {}

    @staticmethod
    def _assert_pod_env(pod_env, expected_variables, expected_secrets=None):
        expected_secrets = expected_secrets or {}
        for env_variable in pod_env:
            if isinstance(env_variable, V1EnvVar):
                env_variable = env_variable.to_dict()
            name = env_variable["name"]
            if name in expected_variables:
                if expected_variables[name]:
                    assert expected_variables[name] == env_variable["value"]
                expected_variables.pop(name)
            elif name in expected_secrets:
                assert (
                    env_variable["value_from"]["secret_key_ref"]["name"]
                    == expected_secrets[name]["name"]
                )
                assert (
                    env_variable["value_from"]["secret_key_ref"]["key"]
                    == expected_secrets[name]["key"]
                )
                expected_secrets.pop(name)

        # Make sure all variables were accounted for
        assert len(expected_variables) == 0
        assert len(expected_secrets) == 0

    @staticmethod
    def _assert_pod_env_from_secrets(pod_env, expected_variables):
        for env_variable in pod_env:
            if isinstance(env_variable, dict) and env_variable.setdefault(
                "valueFrom", None
            ):
                # Nuclio spec comes in as a dict, with some differences from the V1EnvVar - convert it.
                value_from = client.V1EnvVarSource(
                    secret_key_ref=client.V1SecretKeySelector(
                        name=env_variable["valueFrom"]["secretKeyRef"]["name"],
                        key=env_variable["valueFrom"]["secretKeyRef"]["key"],
                    )
                )
                env_variable = V1EnvVar(
                    name=env_variable["name"], value_from=value_from
                )
            if (
                isinstance(env_variable, V1EnvVar)
                and env_variable.value_from is not None
            ):
                name = env_variable.name
                if name in expected_variables:
                    expected_value = expected_variables[name]
                    secret_key = env_variable.value_from.secret_key_ref.key
                    secret_name = env_variable.value_from.secret_key_ref.name
                    assert expected_value[secret_name] == secret_key
                    expected_variables.pop(name)
        assert len(expected_variables) == 0

    def _get_pod_creation_args(self):
        args, _ = get_k8s_helper().v1api.create_namespaced_pod.call_args
        return args[1]

    def _get_custom_object_creation_body(self):
        (
            _,
            kwargs,
        ) = (
            mlrun.api.utils.singletons.k8s.get_k8s_helper().crdapi.create_namespaced_custom_object.call_args
        )
        return kwargs["body"]

    def _get_create_custom_object_namespace_arg(self):
        (
            _,
            kwargs,
        ) = (
            mlrun.api.utils.singletons.k8s.get_k8s_helper().crdapi.create_namespaced_custom_object.call_args
        )
        return kwargs["namespace"]

    def _get_create_pod_namespace_arg(self):
        args, _ = get_k8s_helper().v1api.create_namespaced_pod.call_args
        return args[0]

    def _assert_v3io_mount_or_creds_configured(
        self, v3io_user, v3io_access_key, cred_only=False, masked=True
    ):
        args = self._get_pod_creation_args()
        pod_spec = args.spec
        container_spec = pod_spec.containers[0]

        pod_env = container_spec.env
        expected_variables = {
            "V3IO_API": None,
            "V3IO_USERNAME": v3io_user,
        }
        expected_secrets = {}
        if masked:
            expected_secrets = {
                "V3IO_ACCESS_KEY": {
                    "name": f"secret-ref-{v3io_user}-{v3io_access_key}",
                    "key": "accessKey",
                },
            }
        else:
            expected_variables["V3IO_ACCESS_KEY"] = v3io_access_key

        self._assert_pod_env(
            pod_env,
            expected_variables=expected_variables,
            expected_secrets=expected_secrets,
        )

        if cred_only:
            assert len(pod_spec.volumes) == 0
            assert len(container_spec.volume_mounts) == 0
            return

        expected_volume = {
            "flexVolume": {
                "driver": "v3io/fuse",
                "options": {
                    "dirsToCreate": f'[{{"name": "users//{v3io_user}", "permissions": 488}}]'
                },
            },
            "name": "v3io",
        }
        if masked:
            expected_volume["flexVolume"]["secretRef"] = {
                "name": f"secret-ref-{v3io_user}-{v3io_access_key}"
            }
        else:
            expected_volume["flexVolume"]["options"]["accessKey"] = v3io_access_key

        assert (
            deepdiff.DeepDiff(pod_spec.volumes[0], expected_volume, ignore_order=True)
            == {}
        )

        expected_volume_mounts = [
            {"mountPath": "/v3io", "name": "v3io", "subPath": ""},
            {"mountPath": "/User", "name": "v3io", "subPath": f"users/{v3io_user}"},
        ]
        assert (
            deepdiff.DeepDiff(
                container_spec.volume_mounts, expected_volume_mounts, ignore_order=True
            )
            == {}
        )

    def _assert_pvc_mount_configured(self, pvc_name, pvc_mount_path, volume_name):
        args = self._get_pod_creation_args()
        pod_spec = args.spec

        expected_volume = {
            "name": volume_name,
            "persistentVolumeClaim": {"claimName": pvc_name},
        }
        assert (
            deepdiff.DeepDiff(pod_spec.volumes[0], expected_volume, ignore_order=True)
            == {}
        )

        expected_volume_mounts = [
            {"mountPath": pvc_mount_path, "name": volume_name},
        ]

        container_spec = pod_spec.containers[0]
        assert (
            deepdiff.DeepDiff(
                container_spec.volume_mounts, expected_volume_mounts, ignore_order=True
            )
            == {}
        )

    def _assert_secret_mount(self, volume_name, secret_name, default_mode, mount_path):
        args = self._get_pod_creation_args()
        pod_spec = args.spec

        expected_volume = {
            "name": volume_name,
            "secret": {"defaultMode": default_mode, "secretName": secret_name},
        }
        assert (
            deepdiff.DeepDiff(pod_spec.volumes[0], expected_volume, ignore_order=True)
            == {}
        )

        expected_volume_mounts = [
            {"mountPath": mount_path, "name": volume_name},
        ]

        container_spec = pod_spec.containers[0]
        assert (
            deepdiff.DeepDiff(
                container_spec.volume_mounts, expected_volume_mounts, ignore_order=True
            )
            == {}
        )

    def _assert_pod_creation_config(
        self,
        expected_runtime_class_name="job",
        expected_params=None,
        expected_inputs=None,
        expected_hyper_params=None,
        expected_secrets=None,
        expected_limits=None,
        expected_requests=None,
        expected_code=None,
        expected_env={},
        expected_node_name=None,
        expected_node_selector=None,
        expected_affinity=None,
        expected_priority_class_name=None,
        assert_create_pod_called=True,
        assert_namespace_env_variable=True,
        expected_labels=None,
        expected_env_from_secrets={},
        expected_args=None,
    ):
        if assert_create_pod_called:
            create_pod_mock = get_k8s_helper().v1api.create_namespaced_pod
            create_pod_mock.assert_called_once()

        assert self._get_create_pod_namespace_arg() == self.namespace

        pod = self._get_pod_creation_args()
        self._assert_labels(pod.metadata.labels, expected_runtime_class_name)

        container_spec = pod.spec.containers[0]

        self._assert_container_resources(
            container_spec, expected_limits, expected_requests
        )

        pod_env = container_spec.env

        expected_code_found = False

        if assert_namespace_env_variable:
            expected_env["MLRUN_NAMESPACE"] = self.namespace

        self._assert_pod_env(pod_env, expected_env)
        self._assert_pod_env_from_secrets(pod_env, expected_env_from_secrets)
        for env_variable in pod_env:
            if isinstance(env_variable, V1EnvVar):
                env_variable = dict(name=env_variable.name, value=env_variable.value)
            if env_variable["name"] == "MLRUN_EXEC_CONFIG":
                function_config = json.loads(env_variable["value"])
                self._assert_function_config(
                    function_config,
                    expected_params,
                    expected_inputs,
                    expected_hyper_params,
                    expected_secrets,
                    expected_labels,
                )

            if expected_code and env_variable["name"] == "MLRUN_EXEC_CODE":
                assert env_variable["value"] == b64encode(
                    expected_code.encode("utf-8")
                ).decode("utf-8")
                expected_code_found = True

        if expected_code:
            assert expected_code_found

        if expected_node_name:
            assert pod.spec.node_name == expected_node_name

        if expected_node_selector:
            assert (
                deepdiff.DeepDiff(
                    pod.spec.node_selector,
                    expected_node_selector,
                    ignore_order=True,
                )
                == {}
            )
        if expected_affinity:
            assert (
                deepdiff.DeepDiff(
                    pod.spec.affinity.to_dict(),
                    expected_affinity.to_dict(),
                    ignore_order=True,
                )
                == {}
            )

        if expected_priority_class_name:
            assert pod.spec.priority_class_name == expected_priority_class_name

        assert pod.spec.containers[0].image == self.image_name

        if expected_args:
            assert container_spec.args == expected_args

    def _assert_container_resources(
        self, container_spec, expected_limits, expected_requests
    ):
        if expected_limits:
            assert (
                deepdiff.DeepDiff(
                    container_spec.resources["limits"],
                    expected_limits,
                    ignore_order=True,
                )
                == {}
            )
        if expected_requests:
            assert (
                deepdiff.DeepDiff(
                    container_spec.resources["requests"],
                    expected_requests,
                    ignore_order=True,
                )
                == {}
            )

    def assert_run_without_specifying_resources(self):
        for test_case in [
            {
                # when are not defaults defined
                "default_function_pod_resources": {
                    "requests": {"cpu": None, "memory": None, "gpu": None},
                    "limits": {"cpu": None, "memory": None, "gpu": None},
                },
                "expected_resources": {},
            },
            {
                # with defaults
                "default_function_pod_resources": {
                    "requests": {"cpu": "25m", "memory": "1M"},
                    "limits": {"cpu": "2", "memory": "1G"},
                },
                "expected_resources": {
                    "requests": {"cpu": "25m", "memory": "1M"},
                    "limits": {"cpu": "2", "memory": "1G"},
                },
            },
        ]:
            mlconf.default_function_pod_resources = test_case.get(
                "default_function_pod_resources"
            )

            runtime = self._generate_runtime()
            expected_resources = test_case.get("expected_resources")
            self._assert_container_resources(
                runtime.spec,
                expected_limits=expected_resources.get("limits"),
                expected_requests=expected_resources.get("requests"),
            )

    def assert_node_selection(
        self,
        node_name=None,
        node_selector=None,
        affinity=None,
        tolerations=None,
    ):
        pass

    def assert_security_context(
        self,
        security_context=None,
    ):
        pass

    def assert_run_preemption_mode_with_preemptible_node_selector_and_tolerations(
        self,
    ):
        preemptible_node_selector = self._generate_node_selector()
        mlrun.mlconf.preemptible_nodes.node_selector = base64.b64encode(
            json.dumps(preemptible_node_selector).encode("utf-8")
        )
        mlrun.mlconf.function_defaults.preemption_mode = (
            mlrun.common.schemas.PreemptionModes.prevent.value
        )

        # set default preemptible tolerations
        tolerations = self._generate_tolerations()
        serialized_tolerations = self.k8s_api.sanitize_for_serialization(tolerations)
        mlrun.mlconf.preemptible_nodes.tolerations = base64.b64encode(
            json.dumps(serialized_tolerations).encode("utf-8")
        )
        logger.info(
            "prevent -> prevent, without any node selection, expecting nothing to be added"
        )
        runtime = self._generate_runtime()
        self.execute_function(runtime)
        self.assert_node_selection()

        preemptible_affinity = self._generate_preemptible_affinity()
        preemptible_tolerations = self._generate_preemptible_tolerations()
        logger.info("prevent -> constrain, expecting preemptible affinity")
        runtime.with_preemption_mode(
            mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(
            affinity=preemptible_affinity, tolerations=preemptible_tolerations
        )

        logger.info("constrain -> allow, expecting only preemption tolerations to stay")
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.allow.value)
        self.execute_function(runtime)
        self.assert_node_selection(tolerations=preemptible_tolerations)

        logger.info(
            "allow -> constrain, expecting preemptible affinity with tolerations"
        )
        runtime.with_preemption_mode(
            mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(
            affinity=preemptible_affinity, tolerations=preemptible_tolerations
        )

        logger.info(
            "constrain -> prevent, expecting affinity and tolerations to be removed"
        )
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.prevent.value)
        self.execute_function(runtime)
        self.assert_node_selection()

        logger.info("prevent -> allow, expecting preemptible tolerations")
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.allow.value)
        self.execute_function(runtime)
        self.assert_node_selection(tolerations=preemptible_tolerations)

        logger.info(
            "allow -> prevent, expecting affinity and tolerations to be removed"
        )
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.prevent.value)
        self.execute_function(runtime)
        self.assert_node_selection()

    def assert_run_preemption_mode_with_preemptible_node_selector_and_tolerations_with_extra_settings(
        self,
    ):
        preemptible_node_selector = self._generate_node_selector()
        mlrun.mlconf.preemptible_nodes.node_selector = base64.b64encode(
            json.dumps(preemptible_node_selector).encode("utf-8")
        )
        mlrun.mlconf.function_defaults.preemption_mode = (
            mlrun.common.schemas.PreemptionModes.prevent.value
        )

        # set default preemptible tolerations
        tolerations = self._generate_tolerations()
        serialized_tolerations = self.k8s_api.sanitize_for_serialization(tolerations)
        mlrun.mlconf.preemptible_nodes.tolerations = base64.b64encode(
            json.dumps(serialized_tolerations).encode("utf-8")
        )

        preemptible_affinity = self._generate_preemptible_affinity()
        preemptible_tolerations = self._generate_preemptible_tolerations()
        runtime = self._generate_runtime()
        logger.info(
            "prevent -> prevent, expecting preemptible node selector to be removed"
        )
        runtime.with_node_selection(node_selector=self._generate_node_selector())
        self.execute_function(runtime)
        self.assert_node_selection()

        logger.info(
            "prevent -> constrain with preemptible node selector, expecting preemptible node selector to stay "
            "and preemptible anti-affinity to be removed and preemptible affinity to be added"
        )
        runtime.with_node_selection(node_selector=self._generate_node_selector())
        runtime.with_preemption_mode(
            mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(
            node_selector=preemptible_node_selector,
            affinity=preemptible_affinity,
            tolerations=preemptible_tolerations,
        )
        logger.info(
            "constrain -> allow, with preemptible node selector and affinity and tolerations,"
            " expecting affinity and node selector to be removed and only preemptible tolerations to stay"
        )
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.allow.value)
        self.execute_function(runtime)
        self.assert_node_selection(tolerations=preemptible_tolerations)

        logger.info(
            "allow -> allow, with not preemptible node selector and preemptible tolerations, expecting to stay"
        )
        not_preemptible_node_selector = {"not-preemptible": "true"}
        runtime.with_node_selection(node_selector=not_preemptible_node_selector)
        self.execute_function(runtime)
        self.assert_node_selection(
            node_selector=not_preemptible_node_selector,
            tolerations=preemptible_tolerations,
        )

        logger.info(
            "allow -> prevent, with not preemptible node selector, expecting to stay"
        )
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.prevent.value)
        self.execute_function(runtime)
        self.assert_node_selection(
            node_selector=not_preemptible_node_selector,
        )

        logger.info(
            "prevent -> constrain, with not preemptible node selector, expecting to stay and"
            " preemptible affinity and tolerations to be added"
        )
        runtime.with_preemption_mode(
            mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(
            node_selector=not_preemptible_node_selector,
            affinity=preemptible_affinity,
            tolerations=preemptible_tolerations,
        )

        ##########################################################################################################
        not_preemptible_affinity = self._generate_not_preemptible_affinity()
        logger.info("prevent, with not preemptible affinity, expecting to stay")
        runtime = self._generate_runtime()
        runtime.with_node_selection(affinity=not_preemptible_affinity)
        self.execute_function(runtime)
        self.assert_node_selection(affinity=not_preemptible_affinity)

        logger.info(
            "prevent -> constrain, with not preemptible affinity,"
            " expecting to override affinity with preemptible affinity and add tolerations"
        )
        runtime.with_preemption_mode(
            mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(
            affinity=preemptible_affinity, tolerations=preemptible_tolerations
        )

        logger.info("constrain > constrain, expecting to stay the same")
        runtime.with_preemption_mode(
            mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(
            affinity=preemptible_affinity, tolerations=preemptible_tolerations
        )

        ##########################################################################################################
        logger.info(
            "prevent -> allow, with not preemptible affinity expecting to stay and tolerations to be added"
        )
        runtime = self._generate_runtime()
        runtime.with_node_selection(affinity=self._generate_not_preemptible_affinity())
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.allow.value)
        self.execute_function(runtime)
        self.assert_node_selection(
            affinity=self._generate_not_preemptible_affinity(),
            tolerations=preemptible_tolerations,
        )

        logger.info("allow -> allow, expecting to stay the same")
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.allow.value)
        self.execute_function(runtime)
        self.assert_node_selection(
            affinity=self._generate_not_preemptible_affinity(),
            tolerations=preemptible_tolerations,
        )

        logger.info(
            "allow -> prevent, with not preemptible affinity expecting tolerations to be removed"
        )
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.prevent.value)
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_not_preemptible_affinity())

        logger.info(
            "prevent -> prevent, with not preemptible affinity expecting to stay the same"
        )
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.prevent.value)
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_not_preemptible_affinity())

        ##########################################################################################################
        logger.info(
            "prevent -> constrain, sets different affinity with pod_affinity and"
            " preferred_during_scheduling_ignored_during_execution, expects to override the"
            " required_during_scheduling_ignored_during_execution and tolerations to be added"
        )
        runtime = self._generate_runtime()
        runtime.with_node_selection(affinity=self._generate_affinity())
        runtime.with_preemption_mode(
            mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        expected_affinity = self._generate_affinity()
        expected_affinity.node_affinity.required_during_scheduling_ignored_during_execution = k8s_client.V1NodeSelector(
            node_selector_terms=mlrun.k8s_utils.generate_preemptible_nodes_affinity_terms(),
        )
        self.assert_node_selection(
            affinity=expected_affinity,
            tolerations=self._generate_preemptible_tolerations(),
        )

        ##########################################################################################################

        logger.info(
            "prevent -> prevent, set not preemptible tolerations, expecting to stay"
        )
        runtime = self._generate_runtime()
        runtime.with_node_selection(
            tolerations=self._generate_not_preemptible_tolerations()
        )
        self.execute_function(runtime)
        self.assert_node_selection(
            tolerations=self._generate_not_preemptible_tolerations()
        )

        logger.info(
            "prevent -> constrain, set not preemptible tolerations, expecting preemptible"
            " tolerations merged with not preemptible tolerations and add preemptible affinity"
        )

        merged_preemptible_tolerations = (
            self._generate_not_preemptible_tolerations()
            + self._generate_preemptible_tolerations()
        )
        runtime.with_preemption_mode(
            mode=mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(
            tolerations=merged_preemptible_tolerations,
            affinity=self._generate_preemptible_affinity(),
        )

        logger.info(
            "constrain -> allow, with merged preemptible tolerations and preemptible affinity, "
            "expecting only merged preemptible tolerations"
        )
        runtime.with_preemption_mode(
            mode=mlrun.common.schemas.PreemptionModes.allow.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(
            tolerations=merged_preemptible_tolerations,
        )

    def assert_run_preemption_mode_with_preemptible_node_selector_without_preemptible_tolerations(
        self,
    ):
        # no preemptible nodes tolerations configured, test modes based on affinity/anti-affinity
        preemptible_node_selector = self._generate_node_selector()
        mlrun.mlconf.preemptible_nodes.node_selector = base64.b64encode(
            json.dumps(preemptible_node_selector).encode("utf-8")
        )
        mlrun.mlconf.function_defaults.preemption_mode = (
            mlrun.common.schemas.PreemptionModes.prevent.value
        )
        logger.info(
            "prevent, without setting any node selection expecting preemptible anti-affinity to be set"
        )
        runtime = self._generate_runtime()
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_anti_affinity())

        logger.info("prevent -> constrain, expecting preemptible affinity")
        runtime.with_preemption_mode(
            mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_affinity())

        logger.info("constrain -> allow, expecting no node selection to be set")
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.allow.value)
        self.execute_function(runtime)
        self.assert_node_selection()

        logger.info("allow -> constrain, expecting preemptible affinity")
        runtime.with_preemption_mode(
            mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_affinity())

        logger.info("constrain -> prevent, expecting preemptible anti-affinity")
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.prevent.value)
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_anti_affinity())

        logger.info("prevent -> allow, expecting no node selection to be set")
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.allow.value)
        self.execute_function(runtime)
        self.assert_node_selection()

        logger.info("allow -> prevent, expecting preemptible anti-affinity")
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.prevent.value)
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_anti_affinity())

    def assert_run_preemption_mode_with_preemptible_node_selector_without_preemptible_tolerations_with_extra_settings(
        self,
    ):
        # no preemptible nodes tolerations configured, test modes based on affinity/anti-affinity
        preemptible_node_selector = self._generate_node_selector()
        mlrun.mlconf.preemptible_nodes.node_selector = base64.b64encode(
            json.dumps(preemptible_node_selector).encode("utf-8")
        )
        mlrun.mlconf.function_defaults.preemption_mode = (
            mlrun.common.schemas.PreemptionModes.prevent.value
        )

        logger.info(
            "prevent, expecting preemptible node selector to be removed and only contain anti affinity"
        )
        runtime = self._generate_runtime()
        runtime.with_node_selection(node_selector=preemptible_node_selector)
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_anti_affinity())

        logger.info(
            "prevent -> constrain with preemptible node selector, expecting preemptible node selector to stay "
            "and preemptible anti-affinity to be removed and preemptible affinity to be added"
        )
        runtime.with_node_selection(node_selector=preemptible_node_selector)
        runtime.with_preemption_mode(
            mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(
            node_selector=preemptible_node_selector,
            affinity=self._generate_preemptible_affinity(),
        )
        logger.info(
            "constrain -> allow with preemptible node selector and affinity, expecting both to be removed"
        )
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.allow.value)
        self.execute_function(runtime)
        self.assert_node_selection()

        logger.info(
            "allow -> allow, with not preemptible node selector, expecting to stay"
        )
        not_preemptible_node_selector = {"not-preemptible": "true"}
        runtime.with_node_selection(node_selector=not_preemptible_node_selector)
        self.execute_function(runtime)
        self.assert_node_selection(node_selector=not_preemptible_node_selector)

        logger.info(
            "allow -> prevent, with not preemptible node selector, expecting to stay and preemptible"
            " anti-affinity"
        )
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.prevent.value)
        self.execute_function(runtime)
        self.assert_node_selection(
            node_selector=not_preemptible_node_selector,
            affinity=self._generate_preemptible_anti_affinity(),
        )
        logger.info(
            "prevent -> constrain, with not preemptible node selector, expecting to stay and"
            " preemptible affinity to be add and anti affinity to be remove"
        )
        runtime.with_preemption_mode(
            mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(
            node_selector=not_preemptible_node_selector,
            affinity=self._generate_preemptible_affinity(),
        )

        ##########################################################################################################
        logger.info(
            "prevent -> prevent, with not preemptible affinity, expecting preemptible anti-affinity"
        )
        runtime = self._generate_runtime()
        runtime.with_node_selection(affinity=self._generate_not_preemptible_affinity())
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_anti_affinity())

        logger.info(
            "prevent -> constrain, with preemptible anti-affinity,"
            " expecting to override anti-affinity with preemptible affinity"
        )
        runtime.with_preemption_mode(
            mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_affinity())

        logger.info("constrain > constrain, expecting to stay the same")
        runtime.with_preemption_mode(
            mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_affinity())

        ##########################################################################################################

        logger.info("prevent -> allow, with not preemptible affinity expecting to stay")
        runtime = self._generate_runtime()
        runtime.with_node_selection(affinity=self._generate_not_preemptible_affinity())
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.allow.value)
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_not_preemptible_affinity())

        logger.info("allow -> allow, expecting to stay the same")
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.allow.value)
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_not_preemptible_affinity())

        logger.info(
            "allow -> prevent, with not preemptible affinity expecting to be overridden with anti-affinity"
        )
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.prevent.value)
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_anti_affinity())

        logger.info(
            "prevent -> prevent, with anti-affinity, expecting to stay the same"
        )
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.prevent.value)
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_anti_affinity())

        ##########################################################################################################
        logger.info(
            "prevent -> constrain, sets different affinity, expects to override the"
            " required_during_scheduling_ignored_during_execution and tolerations to be added"
        )
        runtime = self._generate_runtime()
        runtime.with_node_selection(affinity=self._generate_affinity())
        runtime.with_preemption_mode(
            mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        expected_affinity = self._generate_affinity()
        expected_affinity.node_affinity.required_during_scheduling_ignored_during_execution = k8s_client.V1NodeSelector(
            node_selector_terms=mlrun.k8s_utils.generate_preemptible_nodes_affinity_terms(),
        )
        self.assert_node_selection(
            affinity=expected_affinity,
        )

        ##########################################################################################################

        logger.info(
            "prevent -> prevent, set not preemptible tolerations, expecting to stay and anti-affinity to be added"
        )
        runtime = self._generate_runtime()
        runtime.with_node_selection(
            tolerations=self._generate_not_preemptible_tolerations()
        )
        self.execute_function(runtime)
        self.assert_node_selection(
            tolerations=self._generate_not_preemptible_tolerations(),
            affinity=self._generate_preemptible_anti_affinity(),
        )

        logger.info(
            "prevent -> constrain, set not preemptible tolerations, expecting preemptible affinity to be added"
        )

        runtime.with_preemption_mode(
            mode=mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(
            tolerations=self._generate_not_preemptible_tolerations(),
            affinity=self._generate_preemptible_affinity(),
        )

        logger.info(
            "constrain -> allow, with not preemptible tolerations and preemptible affinity, "
            "expecting only not preemptible tolerations"
        )
        runtime.with_preemption_mode(
            mode=mlrun.common.schemas.PreemptionModes.allow.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(
            tolerations=self._generate_not_preemptible_tolerations(),
        )

    def assert_run_with_preemption_mode_none_transitions(self):
        # no preemptible nodes tolerations configured, test modes based on affinity/anti-affinity
        preemptible_node_selector = self._generate_node_selector()
        mlrun.mlconf.preemptible_nodes.node_selector = base64.b64encode(
            json.dumps(preemptible_node_selector).encode("utf-8")
        )
        mlrun.mlconf.function_defaults.preemption_mode = (
            mlrun.common.schemas.PreemptionModes.prevent.value
        )

        logger.info("prevent, expecting anti affinity")
        runtime = self._generate_runtime()
        runtime.with_node_selection()
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_anti_affinity())

        logger.info("prevent -> none, expecting to stay the same")
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.none.value)
        self.execute_function(runtime)
        self.assert_node_selection(affinity=self._generate_preemptible_anti_affinity())

        logger.info(
            "none, enrich with tolerations expecting anti-affinity to stay and tolerations to be added"
        )
        runtime.with_node_selection(tolerations=self._generate_tolerations())
        self.execute_function(runtime)
        self.assert_node_selection(
            affinity=self._generate_preemptible_anti_affinity(),
            tolerations=self._generate_tolerations(),
        )

        logger.info(
            "none -> constrain, expecting preemptible affinity and user's tolerations"
        )
        runtime.with_preemption_mode(
            mlrun.common.schemas.PreemptionModes.constrain.value
        )
        self.execute_function(runtime)
        self.assert_node_selection(
            affinity=self._generate_preemptible_affinity(),
            tolerations=self._generate_tolerations(),
        )

        logger.info(
            "constrain -> none, expecting preemptible affinity to stay and user's tolerations"
        )
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.none.value)
        self.execute_function(runtime)
        self.assert_node_selection(
            affinity=self._generate_preemptible_affinity(),
            tolerations=self._generate_tolerations(),
        )

        logger.info("none -> allow, expecting user's tolerations to stay")
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.allow.value)
        self.execute_function(runtime)
        self.assert_node_selection(tolerations=self._generate_tolerations())

        logger.info("allow -> none, expecting user's tolerations to stay")
        runtime.with_preemption_mode(mlrun.common.schemas.PreemptionModes.none.value)
        self.execute_function(runtime)
        self.assert_node_selection(tolerations=self._generate_tolerations())

    def assert_run_with_preemption_mode_without_preemptible_configuration(self):
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
            for preemption_mode in mlrun.common.schemas.PreemptionModes:
                runtime = self._generate_runtime()
                runtime.with_node_selection(
                    node_name=node_name,
                    node_selector=node_selector,
                    affinity=affinity,
                    tolerations=tolerations,
                )
                runtime.with_preemption_mode(mode=preemption_mode.value)
                self.execute_function(runtime)
                self.assert_node_selection(
                    node_name, node_selector, affinity, tolerations
                )
