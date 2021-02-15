import unittest.mock
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.utils import create_logger
from mlrun.runtimes.constants import PodPhases
from kubernetes import client
from datetime import datetime, timezone
from copy import deepcopy
from mlrun.config import config as mlconf
import json
from mlrun.model import new_task

logger = create_logger(level="debug", name="test-runtime")


class TestRuntimeBase:
    def setup_method(self, method):
        self.namespace = mlconf.namespace = "test-namespace"
        self._logger = logger
        self.project = "test-project"
        self.name = "test-function"
        self.run_uid = "test_run_uid"
        self.image_name = "mlrun/mlrun:latest"
        self.artifact_path = "/tmp"

        self._logger.info(
            f"Setting up test {self.__class__.__name__}::{method.__name__}"
        )

        self.custom_setup()

        self._logger.info(
            f"Finished setting up test {self.__class__.__name__}::{method.__name__}"
        )

    def custom_setup(self):
        pass

    def _generate_task(self):
        return new_task(
            name=self.name, project=self.project, artifact_path=self.artifact_path
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
                name="must-provide-name",
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

    def _assert_labels_exist_in_pod_creation(self, labels: dict):
        expected_labels = {
            "mlrun/class": "job",
            "mlrun/name": self.name,
            "mlrun/project": self.project,
            "mlrun/tag": "latest",
        }

        for key in expected_labels:
            assert labels[key] == expected_labels[key]

    def _assert_function_config_as_expected(
        self,
        config,
        expected_params,
        expected_inputs,
        expected_hyper_params,
        expected_secrets,
    ):
        function_metadata = config["metadata"]
        assert function_metadata["name"] == self.name
        assert function_metadata["project"] == self.project

        function_spec = config["spec"]
        assert function_spec["output_path"] == self.artifact_path
        if expected_params:
            assert function_spec["parameters"] == expected_params
        if expected_inputs:
            assert function_spec["inputs"] == expected_inputs
        if expected_hyper_params:
            assert function_spec["hyperparams"] == expected_hyper_params
        if expected_secrets:
            assert function_spec["secret_sources"] == [expected_secrets]

    @staticmethod
    def _assert_env_variables_set_correctly_in_pod_env(pod_env, expected_variables):
        for env_variable in pod_env:
            name = env_variable["name"]
            if name in expected_variables:
                if expected_variables[name]:
                    assert expected_variables[name] == env_variable["value"]
                expected_variables.pop(name)

        # Make sure all variables were accounted for
        assert len(expected_variables) == 0

    def _assert_v3io_mount_configured(self, v3io_user, v3io_access_key):
        args, _ = get_k8s().v1api.create_namespaced_pod.call_args
        pod_spec = args[1].spec
        container_spec = pod_spec.containers[0]

        pod_env = container_spec.env
        self._assert_env_variables_set_correctly_in_pod_env(
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
        assert pod_spec.volumes[0] == expected_volume

        expected_volume_mounts = [
            {"mountPath": "/v3io", "name": "v3io", "subPath": ""},
            {"mountPath": "/User", "name": "v3io", "subPath": f"users/{v3io_user}"},
        ]
        assert container_spec.volume_mounts == expected_volume_mounts

    def _assert_pod_create_called(
        self,
        expected_params={},
        expected_inputs={},
        expected_hyper_params={},
        expected_secrets={},
        expected_limits={},
    ):
        create_pod_mock = get_k8s().v1api.create_namespaced_pod
        create_pod_mock.assert_called_once()
        args, _ = create_pod_mock.call_args
        # assert args[0] == self.namespace
        pod_spec = args[1]
        self._assert_labels_exist_in_pod_creation(pod_spec.metadata.labels)

        container_spec = pod_spec.spec.containers[0]
        if expected_limits:
            assert container_spec.resources["limits"] == expected_limits

        pod_env = container_spec.env

        self._assert_env_variables_set_correctly_in_pod_env(
            pod_env, {"MLRUN_NAMESPACE": self.namespace}
        )
        for env_variable in pod_env:
            if env_variable["name"] == "MLRUN_EXEC_CONFIG":
                function_config = json.loads(env_variable["value"])
                self._assert_function_config_as_expected(
                    function_config,
                    expected_params,
                    expected_inputs,
                    expected_hyper_params,
                    expected_secrets,
                )

        assert pod_spec.spec.containers[0].image == self.image_name

        # print(f"POD spec:\n {args[1]}")

    # Needed for tracking pod status/logs
    @staticmethod
    def _mock_get_pod():
        get_k8s().v1api.read_namespaced_pod = unittest.mock.Mock()

    # Needed for Vault functionality
    @staticmethod
    def _mock_read_namespaced_service_account():
        get_k8s().v1api.read_namespaced_service_account = unittest.mock.Mock()
