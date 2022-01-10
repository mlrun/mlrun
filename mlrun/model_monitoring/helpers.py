from pathlib import Path

import mlrun
import mlrun.feature_store as fs
from mlrun import code_to_function, v3io_cred
from mlrun.api.crud.secrets import Secrets
from mlrun.config import config
from mlrun.model_monitoring.stream_processing_fs import EventStreamProcessor

HELPERS_FILE_PATH = Path(__file__)
STREAM_PROCESSING_FUNCTION_PATH = HELPERS_FILE_PATH.parent / "stream_processing_fs.py"


def get_model_monitoring_stream_processing_function(
    project: str, model_monitoring_access_key: str, db_session
):
    stream_processor = EventStreamProcessor(
        project=project,
        model_monitoring_access_key=model_monitoring_access_key,
        parquet_batching_max_events=config.model_endpoint_monitoring.parquet_batching_max_events,
    )
    fset = stream_processor.create_feature_set()
    fset._override_run_db(db_session)

    http_source = mlrun.datastore.sources.HttpSource()
    fset.spec.source = http_source

    function = code_to_function(
        name="model-monitoring-stream",
        project=project,
        filename=str(STREAM_PROCESSING_FUNCTION_PATH),
        kind="serving",
        image="mlrun/mlrun",
    )

    # add stream trigger
    function.metadata.project = project

    stream_path = config.model_endpoint_monitoring.store_prefixes.default.format(
        project=project, kind="stream"
    )
    function.add_v3io_stream_trigger(
        stream_path=stream_path, name="monitoring_stream_trigger"
    )

    function.set_env_from_secret(
        "MODEL_MONITORING_ACCESS_KEY",
        mlrun.api.utils.singletons.k8s.get_k8s().get_project_secret_name(project),
        Secrets().generate_model_monitoring_secret_key("MODEL_MONITORING_ACCESS_KEY"),
    )

    run_config = fs.RunConfig(function=function, local=False)
    _, run_config.parameters = fs.api.set_task_params(
        fset, http_source, fset.spec.targets, run_config.parameters
    )

    function.spec.graph = fset.spec.graph
    function.spec.parameters = run_config.parameters
    function.spec.graph_initializer = (
        "mlrun.feature_store.ingestion.featureset_initializer"
    )
    function = http_source.add_nuclio_trigger(function)
    function.metadata.credentials.access_key = model_monitoring_access_key
    function.apply(v3io_cred())

    return function
