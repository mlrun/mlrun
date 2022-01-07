import pathlib

import pytest

import mlrun
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestAPIBackwardCompatibility(TestMLRunSystem):
    project_name = "db-system-test-project"

    def test_endpoints_called_by_sdk_from_inside_jobs(self):
        filename = str(pathlib.Path(__file__).parent / "assets" / "function.py")
        function = mlrun.code_to_function(
            project=self.project_name,
            filename=filename,
            kind="job",
            image="mlrun/ml-models",
        )
        function.run(
            name="test-backward-compatibility-in-running-job-success",
            handler="backward_compatibility_test_in_runtime_success",
        )

        with pytest.raises(mlrun.runtimes.utils.RunError):
            function.run(
                name="test-backward-compatibility-in-running-job-failure",
                handler="backward_compatibility_test_in_runtime_failure",
            )
