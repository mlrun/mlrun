from pathlib import Path

from mlrun import code_to_function

HELPERS_FILE_PATH = Path(__file__)
STREAM_PROCESSING_FUNCTION_PATH = HELPERS_FILE_PATH.parent / "stream_processing.py"


def get_model_monitoring_stream_processing_function(project: str):
    return code_to_function(
        name="model-monitoring-stream",
        project=project,
        filename=str(STREAM_PROCESSING_FUNCTION_PATH),
        kind="nuclio",
        image="mlrun/mlrun",
    )
