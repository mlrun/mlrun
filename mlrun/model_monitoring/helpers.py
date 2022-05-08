from pathlib import Path

from sqlalchemy.orm import Session

import mlrun
import mlrun.feature_store as fs
from mlrun import code_to_function, v3io_cred
from mlrun.api.api.utils import get_run_db_instance
from mlrun.api.crud.secrets import Secrets, SecretsClientType
from mlrun.api.utils.singletons.db import get_db
from mlrun.config import config
from mlrun.model_monitoring.stream_processing_fs import EventStreamProcessor
from mlrun.runtimes import KubejobRuntime
from mlrun.utils.helpers import logger

_CURRENT_FILE_PATH = Path(__file__)
_STREAM_PROCESSING_FUNCTION_PATH = _CURRENT_FILE_PATH.parent / "stream_processing_fs.py"
_MONIOTINRG_BATCH_FUNCTION_PATH = (
    _CURRENT_FILE_PATH.parent / "model_monitoring_batch.py"
)


def get_model_monitoring_stream_processing_function(
    project: str, model_monitoring_access_key: str, db_session: Session
):
    """
    Initialize model monitoring stream processing function.

    :param project:                     project name.
    :param model_monitoring_access_key: access key to apply the model monitoring process.
    :param db_session:                  A session that manages the current dialog with the database.

    :return:                            A function object from a mlrun runtime class

    """

    # initialize Stream Processor object
    stream_processor = EventStreamProcessor(
        project=project,
        model_monitoring_access_key=model_monitoring_access_key,
        parquet_batching_max_events=config.model_endpoint_monitoring.parquet_batching_max_events,
    )

    # create feature set for this project
    fset = stream_processor.create_feature_set()
    fset._override_run_db(db_session)
    http_source = mlrun.datastore.sources.HttpSource()
    fset.spec.source = http_source

    # create a new serving function for the streaming process
    function = code_to_function(
        name="model-monitoring-stream",
        project=project,
        filename=str(_STREAM_PROCESSING_FUNCTION_PATH),
        kind="serving",
        image="mlrun/mlrun",
    )

    # set the project to the serving function
    function.metadata.project = project

    # add v3io stream trigger
    stream_path = config.model_endpoint_monitoring.store_prefixes.default.format(
        project=project, kind="stream"
    )
    function.add_v3io_stream_trigger(
        stream_path=stream_path, name="monitoring_stream_trigger"
    )

    # set model monitoring access key for managing permissions
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
    db_session: Session,
    auth_info: mlrun.api.schemas.AuthInfo,
):
    """
    Initialize model monitoring batch function.

    :param project:                     project name.
    :param model_monitoring_access_key: access key to apply the model monitoring process.
    :param db_session:                  A session that manages the current dialog with the database.

    :return:                            A function object from a mlrun runtime class

    """

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

    # create job function runtime for the model monitoring batch
    function: KubejobRuntime = code_to_function(
        name="model-monitoring-batch",
        project=project,
        filename=str(_MONIOTINRG_BATCH_FUNCTION_PATH),
        kind="job",
        image="mlrun/mlrun",
        handler="handler",
    )
    function.set_db_connection(get_run_db_instance(db_session))

    # set the project to the job function
    function.metadata.project = project

    # set model monitoring access key for managing permissions
    function.set_env_from_secret(
        "MODEL_MONITORING_ACCESS_KEY",
        mlrun.api.utils.singletons.k8s.get_k8s().get_project_secret_name(project),
        Secrets().generate_client_project_secret_key(
            SecretsClientType.model_monitoring, "MODEL_MONITORING_ACCESS_KEY"
        ),
    )

    function.apply(mlrun.mount_v3io())

    # Needs to be a member of the project and have access to project data path
    function.metadata.credentials.access_key = model_monitoring_access_key

    return function
