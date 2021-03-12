import json
import pathlib
import sys
import unittest.mock
from base64 import b64encode
from copy import deepcopy
from datetime import datetime, timezone

import deepdiff
from kubernetes import client
from kubernetes.client import V1EnvVar

from mlrun.api.utils.singletons.k8s import get_k8s
from kubernetes import client as k8s_client
from mlrun.config import config as mlconf
from mlrun.model import new_task
from mlrun.runtimes.constants import PodPhases
from mlrun.utils import create_logger
from mlrun.utils.vault import VaultStore

logger = create_logger(level="debug", name="test-runtime")


class TestRuntimeBase:
    def setup_method(self, method):
        self.namespace = mlconf.namespace = "test-namespace"
        get_k8s().namespace = self.namespace
        self._logger = logger
        self.project = "test-project"
        self.name = "test-function"
        self.run_uid = "test_run_uid"
        self.image_name = "mlrun/mlrun:latest"
        self.artifact_path = "/tmp"
        self.function_name_label = "mlrun/name"
        self.code_filename = str(self.assets_path / "sample_function.py")

        self.vault_secrets = ["secret1", "secret2", "AWS_KEY"]
        self.vault_secret_value = "secret123!@"
        self.vault_secret_name = "vault-secret"

        self._logger.info(
            f"Setting up test {self.__class__.__name__}::{method.__name__}"
        )

        self.custom_setup()

        self._logger.info(
            f"Finished setting up test {self.__class__.__name__}::{method.__name__}"
        )

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

    def _generate_runtime(self):
        pass

    def custom_setup(self):
        pass

    def custom_teardown(self):
        pass

    def _generate_task(self):
        return new_task(
            name=self.name, project=self.project, artifact_path=self.artifact_path
        )

    def _generate_affinity(self):
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
                                    values=["possible-label-value-1", "possible-label-value-2"],
                                )
                            ]
                        )
                    )
                ]
            ),
            pod_affinity=k8s_client.V1PodAffinity(
                required_during_scheduling_ignored_during_execution=[
                    k8s_client.V1PodAffinityTerm(
                        label_selector=k8s_client.V1LabelSelector(
                            match_labels={
                                "some-pod-label-key": "some-pod-label-value",
                            }
                        ),
                        namespaces=[
                            "namespace-a",
                            "namespace-b"
                        ],
                        topology_key="key-1"
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
                                        values=["forbidden-label-value-1", "forbidden-label-value-2"],
                                    )
                                ]
                            ),
                            namespaces=[
                                "namespace-c",
                            ],
                            topology_key="key-2"
                        )
                    )
                ]
            )
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

        get_k8s().v1api.create_namespaced_pod = unittest.mock.Mock(
            side_effect=_generate_pod
        )

    # Vault now supported in KubeJob and Serving, so moved to base.
    def _mock_vault_functionality(self):
        secret_dict = {key: self.vault_secret_value for key in self.vault_secrets}
        VaultStore.get_secrets = unittest.mock.Mock(return_value=secret_dict)

        object_meta = client.V1ObjectMeta(
            name="test-service-account", namespace=self.namespace
        )
        secret = client.V1ObjectReference(
            name=self.vault_secret_name, namespace=self.namespace
        )
        service_account = client.V1ServiceAccount(
            metadata=object_meta, secrets=[secret]
        )
        get_k8s().v1api.read_namespaced_service_account = unittest.mock.Mock(
            return_value=service_account
        )

    def _execute_run(self, runtime, **kwargs):
        # Reset the mock, so that when checking is create_pod was called, no leftovers are there (in case running
        # multiple runs in the same test)
        get_k8s().v1api.create_namespaced_pod.reset_mock()

        runtime.run(
            name=self.name,
            project=self.project,
            artifact_path=self.artifact_path,
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
                function_metadata["labels"], expected_labels, ignore_order=True,
            )
            # We just care that the values we look for are fully there.
            diff_result.pop("dictionary_item_removed", None)
            assert diff_result == {}

    @staticmethod
    def _assert_pod_env(pod_env, expected_variables):
        for env_variable in pod_env:
            if isinstance(env_variable, V1EnvVar):
                env_variable = dict(name=env_variable.name, value=env_variable.value)
            name = env_variable["name"]
            if name in expected_variables:
                if expected_variables[name]:
                    assert expected_variables[name] == env_variable["value"]
                expected_variables.pop(name)

        # Make sure all variables were accounted for
        assert len(expected_variables) == 0

    def _get_pod_creation_args(self):
        args, _ = get_k8s().v1api.create_namespaced_pod.call_args
        return args[1]

    def _get_namespace_arg(self):
        args, _ = get_k8s().v1api.create_namespaced_pod.call_args
        return args[0]

    def _assert_v3io_mount_configured(self, v3io_user, v3io_access_key):
        args = self._get_pod_creation_args()
        pod_spec = args.spec
        container_spec = pod_spec.containers[0]

        pod_env = container_spec.env
        self._assert_pod_env(
            pod_env,
            {
                "V3IO_API": None,
                "V3IO_USERNAME": v3io_user,
                "V3IO_ACCESS_KEY": v3io_access_key,
            },
        )

        expected_volume = {
            "flexVolume": {
                "driver": "v3io/fuse",
                "options": {"accessKey": v3io_access_key},
            },
            "name": "v3io",
        }
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
        assert_create_pod_called=True,
        assert_namespace_env_variable=True,
        expected_labels=None,
    ):
        if assert_create_pod_called:
            create_pod_mock = get_k8s().v1api.create_namespaced_pod
            create_pod_mock.assert_called_once()

        assert self._get_namespace_arg() == self.namespace

        pod = self._get_pod_creation_args()
        self._assert_labels(pod.metadata.labels, expected_runtime_class_name)

        container_spec = pod.spec.containers[0]

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

        pod_env = container_spec.env

        expected_code_found = False

        if assert_namespace_env_variable:
            expected_env["MLRUN_NAMESPACE"] = self.namespace

        self._assert_pod_env(pod_env, expected_env)
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
            assert deepdiff.DeepDiff(
                    pod.spec.node_selector,
                    expected_node_selector,
                    ignore_order=True,
                ) == {}
        if expected_affinity:
            assert deepdiff.DeepDiff(
                    pod.spec.affinity,
                    expected_affinity.to_dict(),
                    ignore_order=True,
                ) == {}

        assert pod.spec.containers[0].image == self.image_name
