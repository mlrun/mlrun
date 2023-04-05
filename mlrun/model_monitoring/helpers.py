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
import typing

import sqlalchemy.orm
from fastapi import Depends

import mlrun
import mlrun.api.api.utils
import mlrun.api.crud.secrets
import mlrun.api.schemas
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.k8s
import mlrun.config
import mlrun.feature_store as fstore
import mlrun.model_monitoring.constants as model_monitoring_constants
import mlrun.model_monitoring.stream_processing_fs
import mlrun.runtimes
import mlrun.utils.helpers
import mlrun.utils.model_monitoring
from mlrun.api.api import deps

_CURRENT_FILE_PATH = pathlib.Path(__file__)
_STREAM_PROCESSING_FUNCTION_PATH = _CURRENT_FILE_PATH.parent / "stream_processing_fs.py"
_MONIOTINRG_BATCH_FUNCTION_PATH = (
    _CURRENT_FILE_PATH.parent / "model_monitoring_batch.py"
)


def initial_model_monitoring_stream_processing_function(
    project: str,
    model_monitoring_access_key: str,
    tracking_policy: mlrun.utils.model_monitoring.TrackingPolicy,
    auth_info: mlrun.api.schemas.AuthInfo,
):
    """
    Initialize model monitoring stream processing function.

    :param project:                     Project name.
    :param model_monitoring_access_key: Access key to apply the model monitoring process. Please note that in CE
                                        deployments this parameter will be None.
    :param tracking_policy:             Model monitoring configurations.
    :param auth_info:                   The auth info of the request.

    :return:                            A function object from a mlrun runtime class

    """

    # Initialize Stream Processor object
    stream_processor = mlrun.model_monitoring.stream_processing_fs.EventStreamProcessor(
        project=project,
        parquet_batching_max_events=mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events,
        model_monitoring_access_key=model_monitoring_access_key,
    )

    # Create a new serving function for the streaming process
    function = mlrun.code_to_function(
        name="model-monitoring-stream",
        project=project,
        filename=str(_STREAM_PROCESSING_FUNCTION_PATH),
        kind="serving",
        image=tracking_policy.stream_image,
    )

    # Create monitoring serving graph
    stream_processor.apply_monitoring_serving_graph(function)

    # Set the project to the serving function
    function.metadata.project = project

    # Add stream triggers
    function = _apply_stream_trigger(
        project=project,
        function=function,
        model_monitoring_access_key=model_monitoring_access_key,
        auth_info=auth_info,
    )

    # Apply feature store run configurations on the serving function
    run_config = fstore.RunConfig(function=function, local=False)
    function.spec.parameters = run_config.parameters

    return function


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
    :param model_monitoring_access_key: access key to apply the model monitoring process. Please note that in CE
                                        deployments this parameter will be None.
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
        image=tracking_policy.default_batch_image,
        handler="handler",
    )
    function.set_db_connection(mlrun.api.api.utils.get_run_db_instance(db_session))

    # Set the project to the job function
    function.metadata.project = project

    if not mlrun.mlconf.is_ce_mode():
        function = _apply_access_key_and_mount_function(
            project=project,
            function=function,
            model_monitoring_access_key=model_monitoring_access_key,
            auth_info=auth_info,
        )

    # Enrich runtime with the required configurations
    mlrun.api.api.utils.apply_enrichment_and_validation_on_function(function, auth_info)

    return function


def _apply_stream_trigger(
    project: str,
    function: mlrun.runtimes.ServingRuntime,
    model_monitoring_access_key: str = None,
    auth_info: mlrun.api.schemas.AuthInfo = Depends(deps.authenticate_request),
) -> mlrun.runtimes.ServingRuntime:
    """Adding stream source for the nuclio serving function. By default, the function has HTTP stream trigger along
    with another supported stream source that can be either Kafka or V3IO, depends on the stream path schema that is
    defined under mlrun.mlconf.model_endpoint_monitoring.store_prefixes. Note that if no valid stream path has been
    provided then the function will have a single HTTP stream source.

    :param project:                     Project name.
    :param function:                    The serving function object that will be applied with the stream trigger.
    :param model_monitoring_access_key: Access key to apply the model monitoring stream function when the stream is
                                        schema is V3IO.
    :param auth_info:                   The auth info of the request.

    :return: ServingRuntime object with stream trigger.
    """

    # Get the stream path from the configuration
    # stream_path = mlrun.mlconf.get_file_target_path(project=project, kind="stream", target="stream")
    stream_path = mlrun.utils.model_monitoring.get_stream_path(project=project)

    if stream_path.startswith("kafka://"):

        topic, brokers = mlrun.datastore.utils.parse_kafka_url(url=stream_path)
        # Generate Kafka stream source
        stream_source = mlrun.datastore.sources.KafkaSource(
            brokers=brokers,
            topics=[topic],
        )
        function = stream_source.add_nuclio_trigger(function)

    if not mlrun.mlconf.is_ce_mode():
        function = _apply_access_key_and_mount_function(
            project=project,
            function=function,
            model_monitoring_access_key=model_monitoring_access_key,
            auth_info=auth_info,
        )
        if stream_path.startswith("v3io://"):
            # Generate V3IO stream trigger
            function.add_v3io_stream_trigger(
                stream_path=stream_path, name="monitoring_stream_trigger"
            )
    # Add the default HTTP source
    http_source = mlrun.datastore.sources.HttpSource()
    function = http_source.add_nuclio_trigger(function)

    return function


def _apply_access_key_and_mount_function(
    project: str,
    function: typing.Union[
        mlrun.runtimes.KubejobRuntime, mlrun.runtimes.ServingRuntime
    ],
    model_monitoring_access_key: str,
    auth_info: mlrun.api.schemas.AuthInfo,
) -> typing.Union[mlrun.runtimes.KubejobRuntime, mlrun.runtimes.ServingRuntime]:
    """Applying model monitoring access key on the provided function when using V3IO path. In addition, this method
    mount the V3IO path for the provided function to configure the access to the system files.

    :param project:                     Project name.
    :param function:                    Model monitoring function object that will be filled with the access key and
                                        the access to the system files.
    :param model_monitoring_access_key: Access key to apply the model monitoring stream function when the stream is
                                        schema is V3IO.
    :param auth_info:                   The auth info of the request.

    :return: function runtime object with access key and access to system files.
    """

    # Set model monitoring access key for managing permissions
    function.set_env_from_secret(
        model_monitoring_constants.ProjectSecretKeys.ACCESS_KEY,
        mlrun.api.utils.singletons.k8s.get_k8s().get_project_secret_name(project),
        mlrun.api.crud.secrets.Secrets().generate_client_project_secret_key(
            mlrun.api.crud.secrets.SecretsClientType.model_monitoring,
            model_monitoring_constants.ProjectSecretKeys.ACCESS_KEY,
        ),
    )
    function.metadata.credentials.access_key = model_monitoring_access_key
    function.apply(mlrun.mount_v3io())

    # Ensure that the auth env vars are set
    mlrun.api.api.utils.ensure_function_has_auth_set(function, auth_info)

    return function
