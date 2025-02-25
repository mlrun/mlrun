import pytest
import mlrun_pipelines.utils


@pytest.fixture
def client():
    return mlrun_pipelines.utils.ExtendedKfpClient


@pytest.mark.parametrize("original_name, project, expected", [
    ("sample_run", "projectX", "projectX-Retry of sample_run"),
    ("projectX-sample_run", "projectX", "projectX-Retry of sample_run"),
    ("Retry of sample_run", "projectX", "projectX-Retry of sample_run"),
    ("projectX-Retry of sample_run", "projectX", "projectX-Retry of sample_run"),
    ("  projectX-Retry of   sample_run  ", "projectX", "projectX-Retry of sample_run"),
])
def test_normalize_retry_run(client, original_name, project, expected):
    assert client._normalize_retry_run(original_name, project) == expected
