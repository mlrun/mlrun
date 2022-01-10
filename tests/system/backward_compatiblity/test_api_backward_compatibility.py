import pathlib

import pytest

import mlrun
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestAPIBackwardCompatibility(TestMLRunSystem):
    project_name = "db-system-test-project"

    def test_endpoints_called_by_sdk_from_inside_jobs(self):
        filename = str(pathlib.Path(__file__).parent / "assets" / "function.py")
        success_handler = "api_backward_compatibility_tests_succeeding_function"
        failure_handler = "api_backward_compatibility_tests_failing_function"

        function = mlrun.code_to_function(
            project=self.project_name,
            filename=filename,
            kind="job",
            image="mlrun/mlrun",
        )
        function.run(
            name=f"test_{success_handler}", handler=success_handler,
        )

        with pytest.raises(mlrun.runtimes.utils.RunError):
            function.run(
                name=f"test_{failure_handler}", handler=failure_handler,
            )
