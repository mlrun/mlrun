from pathlib import Path

import mlrun
import mlrun.feature_store as fs
from mlrun import code_to_function, v3io_cred
from mlrun.api.api.utils import get_run_db_instance
from mlrun.utils.helpers import logger
from mlrun.api.utils.singletons.db import get_db
from mlrun.api.crud.secrets import Secrets, SecretsClientType
from mlrun.runtimes import KubejobRuntime

from mlrun.config import config
from mlrun.model_monitoring.stream_processing_fs import EventStreamProcessor

HELPERS_FILE_PATH = Path(__file__)
STREAM_PROCESSING_FUNCTION_PATH = HELPERS_FILE_PATH.parent / "stream_processing_fs.py"
MONIOTINRG_BATCH_FUNCTION_PATH = Path(__file__).parent / "model_monitoring_batch.py"

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
        image="eyaligu/mlrun-api:latest",
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
        Secrets().generate_client_project_secret_key(
            SecretsClientType.model_monitoring, "MODEL_MONITORING_ACCESS_KEY"
        ),
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

def get_model_monitoring_batch_function(
        project: str,
        model_monitoring_access_key: str,
        db_session,
        auth_info: mlrun.api.schemas.AuthInfo,
    ):
    logger.info(
        f"Checking deployment status for model monitoring batch processing function [{project}]"
    )
    function_list = get_db().list_functions(
        session=db_session, name="model-monitoring-batch", project=project
    )

    if function_list:
        logger.info(
            f"Detected model monitoring batch processing function [{project}] already deployed"
        )
        return

    logger.info(f"Deploying model monitoring batch processing function [{project}]")

    function: KubejobRuntime = code_to_function(
        name="model-monitoring-batch",
        project=project,
        filename=str(MONIOTINRG_BATCH_FUNCTION_PATH),
        kind="job",
        image='eyaligu/mlrun-api:latest',
        handler='handler'
    )
    function.set_db_connection(get_run_db_instance(db_session))

    function.metadata.project = project

    function.apply(mlrun.mount_v3io())

    function.set_env_from_secret(
        "MODEL_MONITORING_ACCESS_KEY",
        mlrun.api.utils.singletons.k8s.get_k8s().get_project_secret_name(project),
        Secrets().generate_client_project_secret_key(
            SecretsClientType.model_monitoring, "MODEL_MONITORING_ACCESS_KEY"
        ),
    )

    # Needs to be a member of the project and have access to project data path
    function.metadata.credentials.access_key = model_monitoring_access_key

    return function
