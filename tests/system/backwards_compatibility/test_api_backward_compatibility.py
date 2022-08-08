import pathlib

import pytest

import mlrun
import mlrun.artifacts
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestAPIBackwardCompatibility(TestMLRunSystem):
    project_name = "db-system-test-project"

    def test_endpoints_called_by_sdk_from_inside_jobs(self):
        filename = str(pathlib.Path(__file__).parent / "assets" / "function.py")
        success_handler = "api_backward_compatibility_tests_succeeding_function"
        failure_handler = "api_backward_compatibility_tests_failing_function"
        dataset_name = "test_dataset"

        function = mlrun.code_to_function(
            project=self.project_name,
            filename=filename,
            kind="job",
            image="mlrun/mlrun",
        )
        artifact_path = ""
        if TestMLRunSystem.is_enterprise_environment():
            artifact_path = f"v3io:///projects/{self.project_name}/artifacts"

        run = function.run(
            name="log_dataset",
            handler="log_dataset",
            artifact_path=artifact_path,
            params={"dataset_name": dataset_name},
        )

        function.run(
            name=f"test_{success_handler}",
            handler=success_handler,
            inputs={"dataset_src": run.outputs[dataset_name]},
        )

        with pytest.raises(mlrun.runtimes.utils.RunError):
            function.run(
                name=f"test_{failure_handler}",
                handler=failure_handler,
            )
