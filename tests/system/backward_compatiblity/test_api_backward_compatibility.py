import pathlib

import pandas as pd
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

        raw_data = {
            "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
        }
        df = pd.DataFrame(raw_data, columns=["first_name"])
        dataset_artifact = self.project.log_dataset(
            "mydf", df=df, stats=True, format="parquet"
        )
        function = mlrun.code_to_function(
            project=self.project_name,
            filename=filename,
            kind="job",
            image="mlrun/mlrun",
        )
        function.run(
            name=f"test_{success_handler}",
            handler=success_handler,
            inputs={"dataset_src": dataset_artifact.uri},
        )

        with pytest.raises(mlrun.runtimes.utils.RunError):
            function.run(
                name=f"test_{failure_handler}", handler=failure_handler,
            )
