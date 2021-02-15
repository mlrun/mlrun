from tests.api.runtimes.base import TestRuntimeBase
from mlrun.runtimes.kubejob import KubejobRuntime
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from mlrun.runtimes.utils import generate_resources
from mlrun.platforms import auto_mount
import os


class TestKubejobRuntime(TestRuntimeBase):
    def custom_setup(self):
        self._mock_create_namespaced_pod()
        self.image_name = "mlrun/mlrun:latest"

    def _generate_runtime(self):
        runtime = KubejobRuntime()
        runtime.spec.image = self.image_name
        return runtime

    def test_kubejob_runtime_run_without_runspec(self, db: Session, client: TestClient):
        runtime = self._generate_runtime()
        self._execute_run(runtime)
        self._assert_pod_create_called()

        params = {"p1": "v1", "p2": 20}
        inputs = {"input1": f"{self.artifact_path}/input1.txt"}

        self._execute_run(runtime, params=params, inputs=inputs)
        self._assert_pod_create_called(expected_params=params, expected_inputs=inputs)

    def test_kubejob_runtime_run_with_runspec(self, db: Session, client: TestClient):
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
        self._assert_pod_create_called(
            expected_params=params,
            expected_inputs=inputs,
            expected_hyper_params=hyper_params,
            expected_secrets=secret_source,
        )

    def test_kubejob_runtime_run_with_pod_modifications(
        self, db: Session, client: TestClient
    ):
        runtime = self._generate_runtime()

        expected_limits = generate_resources(2, 4, 4, "test/gpu")
        runtime.with_limits(
            mem=expected_limits["memory"],
            cpu=expected_limits["cpu"],
            gpus=expected_limits["test/gpu"],
            gpu_type="test/gpu",
        )

        # Set the env variable, so auto_mount() will pick it up and mount v3io
        v3io_access_key = "1111-2222-3333-4444"
        os.environ["V3IO_ACCESS_KEY"] = v3io_access_key
        os.environ["V3IO_USERNAME"] = "test_user"
        runtime.apply(auto_mount())

        self._execute_run(runtime)
        self._assert_pod_create_called(expected_limits=expected_limits)
        self._assert_v3io_mount_configured("test_user", v3io_access_key)
