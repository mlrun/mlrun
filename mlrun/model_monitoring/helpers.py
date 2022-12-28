# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pathlib

import sqlalchemy.orm

import mlrun
import mlrun.api.api.utils
import mlrun.api.crud.secrets
import mlrun.api.utils.singletons.db
import mlrun.config
import mlrun.feature_store as fstore
import mlrun.model_monitoring.constants as model_monitoring_constants
import mlrun.model_monitoring.stream_processing_fs
import mlrun.runtimes
import mlrun.utils.helpers

_CURRENT_FILE_PATH = pathlib.Path(__file__)
_STREAM_PROCESSING_FUNCTION_PATH = _CURRENT_FILE_PATH.parent / "stream_processing_fs.py"
_MONIOTINRG_BATCH_FUNCTION_PATH = (
    _CURRENT_FILE_PATH.parent / "model_monitoring_batch.py"
)


def initial_model_monitoring_stream_processing_function(
    project: str,
    model_monitoring_access_key: str,
    db_session: sqlalchemy.orm.Session,
    tracking_policy: mlrun.utils.model_monitoring.TrackingPolicy,
):
    """
    Initialize model monitoring stream processing function.

    :param project:                     project name.
    :param model_monitoring_access_key: access key to apply the model monitoring process.
    :param db_session:                  A session that manages the current dialog with the database.
    :param tracking_policy:             Model monitoring configurations.

    :return:                            A function object from a mlrun runtime class

    """

    # Initialize Stream Processor object
    stream_processor = mlrun.model_monitoring.stream_processing_fs.EventStreamProcessor(
        project=project,
        model_monitoring_access_key=model_monitoring_access_key,
        parquet_batching_max_events=mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events,
    )

    http_source = mlrun.datastore.sources.HttpSource()

    # Create a new serving function for the streaming process
    function = mlrun.code_to_function(
        name="model-monitoring-stream",
        project=project,
        filename=str(_STREAM_PROCESSING_FUNCTION_PATH),
        kind="serving",
        image=tracking_policy[model_monitoring_constants.EventFieldType.STREAM_IMAGE],
    )

    # Create monitoring serving graph
    stream_processor.apply_monitoring_serving_graph(function)

    # Set the project to the serving function
    function.metadata.project = project

    # Add v3io stream trigger
    stream_path = mlrun.mlconf.model_endpoint_monitoring.store_prefixes.default.format(
        project=project, kind="stream"
    )
    function.add_v3io_stream_trigger(
        stream_path=stream_path, name="monitoring_stream_trigger"
    )

    # Set model monitoring access key for managing permissions
    function.set_env_from_secret(
        "MODEL_MONITORING_ACCESS_KEY",
        mlrun.api.utils.singletons.k8s.get_k8s().get_project_secret_name(project),
        mlrun.api.crud.secrets.Secrets().generate_client_project_secret_key(
            mlrun.api.crud.secrets.SecretsClientType.model_monitoring,
            "MODEL_MONITORING_ACCESS_KEY",
        ),
    )

    run_config = fstore.RunConfig(function=function, local=False)
    function.spec.parameters = run_config.parameters

    func = http_source.add_nuclio_trigger(function)
    func.metadata.credentials.access_key = model_monitoring_access_key
    func.apply(mlrun.v3io_cred())

    return func


def get_model_monitoring_batch_function(
    project: str,
    model_monitoring_access_key: str,
    db_session: sqlalchemy.orm.Session,
    auth_info: mlrun.api.schemas.AuthInfo,
    tracking_policy: mlrun.utils.model_monitoring.TrackingPolicy,
):
    """
    Initialize model monitoring batch function.

    :param project:                     project name.
    :param model_monitoring_access_key: access key to apply the model monitoring process.
    :param db_session:                  A session that manages the current dialog with the database.
    :param auth_info:                   The auth info of the request.
    :param tracking_policy:             Model monitoring configurations.

    :return:                            A function object from a mlrun runtime class

    """

    # Create job function runtime for the model monitoring batch
    function: mlrun.runtimes.KubejobRuntime = mlrun.code_to_function(
        name="model-monitoring-batch",
        project=project,
        filename=str(_MONIOTINRG_BATCH_FUNCTION_PATH),
        kind="job",
        image=tracking_policy[
            model_monitoring_constants.EventFieldType.DEFAULT_BATCH_IMAGE
        ],
        handler="handler",
    )
    function.set_db_connection(mlrun.api.api.utils.get_run_db_instance(db_session))

    # Set the project to the job function
    function.metadata.project = project

    # Set model monitoring access key for managing permissions
    function.set_env_from_secret(
        "MODEL_MONITORING_ACCESS_KEY",
        mlrun.api.utils.singletons.k8s.get_k8s().get_project_secret_name(project),
        mlrun.api.crud.secrets.Secrets().generate_client_project_secret_key(
            mlrun.api.crud.secrets.SecretsClientType.model_monitoring,
            "MODEL_MONITORING_ACCESS_KEY",
        ),
    )

    function.apply(mlrun.mount_v3io())

    # Needs to be a member of the project and have access to project data path
    function.metadata.credentials.access_key = model_monitoring_access_key

    # Ensure that the auth env vars are set
    mlrun.api.api.utils.ensure_function_has_auth_set(function, auth_info)

    return function
