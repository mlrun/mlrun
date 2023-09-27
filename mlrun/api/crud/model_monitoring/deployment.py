# Copyright 2023 Iguazio
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

import mlrun.api.api.endpoints.functions
import mlrun.api.api.utils
import mlrun.api.crud.model_monitoring.helpers
import mlrun.api.utils.scheduler
import mlrun.api.utils.singletons.db
import mlrun.api.utils.singletons.k8s
import mlrun.common.schemas.model_monitoring
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.model_monitoring.stream_processing
import mlrun.model_monitoring.tracking_policy
from mlrun import feature_store as fstore
from mlrun.api.api import deps
from mlrun.api.crud.model_monitoring.helpers import Seconds, seconds2minutes
from mlrun.model_monitoring.writer import ModelMonitoringWriter
from mlrun.utils import logger

_MODEL_MONITORING_COMMON_PATH = pathlib.Path(__file__).parents[3] / "model_monitoring"
_STREAM_PROCESSING_FUNCTION_PATH = (
    _MODEL_MONITORING_COMMON_PATH / "stream_processing.py"
)
_MONITORING_ORIGINAL_BATCH_FUNCTION_PATH = _MODEL_MONITORING_COMMON_PATH / "batch.py"

_MONITORING_APPLICATION_BATCH_FUNCTION_PATH = (
    _MODEL_MONITORING_COMMON_PATH / "batch_application_handler.py"
)

_MONITORING_WRITER_FUNCTION_PATH = _MODEL_MONITORING_COMMON_PATH / "writer.py"


class MonitoringDeployment:
    def __init__(
        self,
        parquet_batching_max_events: int = mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events,
        max_parquet_save_interval: int = mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs,
    ) -> None:
        self._parquet_batching_max_events = parquet_batching_max_events
        self._max_parquet_save_interval = max_parquet_save_interval
        """
        Initialize a MonitoringDeployment object, which handles the deployment & scheduling of:
         1. model monitoring stream
         2. model monitoring batch
         3. model monitoring batch application
         4. model monitoring writer

        :param parquet_batching_max_events: Maximum number of events that will be used for writing the monitoring
                                            parquet by the monitoring stream function.
        :param max_parquet_save_interval:   Maximum number of seconds to hold events before they are written to the
                                            monitoring parquet target. Note that this value will be used to handle the
                                            offset by the scheduled batch job.
        """

    def deploy_monitoring_functions(
        self,
        project: str,
        model_monitoring_access_key: str,
        db_session: sqlalchemy.orm.Session,
        auth_info: mlrun.common.schemas.AuthInfo,
        tracking_policy: mlrun.model_monitoring.tracking_policy.TrackingPolicy,
    ):
        """
        Invoking monitoring deploying functions.

        :param project:                     The name of the project.
        :param model_monitoring_access_key: Access key to apply the model monitoring process.
        :param db_session:                  A session that manages the current dialog with the database.
        :param auth_info:                   The auth info of the request.
        :param tracking_policy:             Model monitoring configurations.
        """
        self.deploy_model_monitoring_stream_processing(
            project=project,
            model_monitoring_access_key=model_monitoring_access_key,
            db_session=db_session,
            auth_info=auth_info,
            tracking_policy=tracking_policy,
        )
        self.deploy_model_monitoring_batch_processing(
            project=project,
            model_monitoring_access_key=model_monitoring_access_key,
            db_session=db_session,
            auth_info=auth_info,
            tracking_policy=tracking_policy,
            tracking_offset=Seconds(self._max_parquet_save_interval),
            function_name=mm_constants.MonitoringFunctionNames.BATCH,
        )
        if tracking_policy.application_batch:
            self.deploy_model_monitoring_batch_processing(
                project=project,
                model_monitoring_access_key=model_monitoring_access_key,
                db_session=db_session,
                auth_info=auth_info,
                tracking_policy=tracking_policy,
                tracking_offset=Seconds(self._max_parquet_save_interval),
                function_name=mm_constants.MonitoringFunctionNames.BATCH_APPLICATION,
            )
            self.deploy_model_monitoring_writer_application(
                project=project,
                model_monitoring_access_key=model_monitoring_access_key,
                db_session=db_session,
                auth_info=auth_info,
                tracking_policy=tracking_policy,
            )

    def deploy_model_monitoring_stream_processing(
        self,
        project: str,
        model_monitoring_access_key: str,
        db_session: sqlalchemy.orm.Session,
        auth_info: mlrun.common.schemas.AuthInfo,
        tracking_policy: mlrun.model_monitoring.tracking_policy.TrackingPolicy,
    ) -> None:
        """
        Deploying model monitoring stream real time nuclio function. The goal of this real time function is
        to monitor the log of the data stream. It is triggered when a new log entry is detected.
        It processes the new events into statistics that are then written to statistics databases.

        :param project:                     The name of the project.
        :param model_monitoring_access_key: Access key to apply the model monitoring process.
        :param db_session:                  A session that manages the current dialog with the database.
        :param auth_info:                   The auth info of the request.
        :param tracking_policy:             Model monitoring configurations.
        """

        logger.info(
            "Checking if model monitoring stream is already deployed",
            project=project,
        )
        try:
            # validate that the model monitoring stream has not yet been deployed
            mlrun.runtimes.function.get_nuclio_deploy_status(
                name="model-monitoring-stream",
                project=project,
                tag="",
                auth_info=auth_info,
            )
            logger.info(
                "Detected model monitoring stream processing function already deployed",
                project=project,
            )
            return
        except mlrun.errors.MLRunNotFoundError:
            logger.info(
                "Deploying model monitoring stream processing function", project=project
            )

        # Get parquet target value for model monitoring stream function
        parquet_target = (
            mlrun.api.crud.model_monitoring.helpers.get_monitoring_parquet_path(
                db_session=db_session, project=project
            )
        )

        fn = self._initial_model_monitoring_stream_processing_function(
            project=project,
            model_monitoring_access_key=model_monitoring_access_key,
            tracking_policy=tracking_policy,
            auth_info=auth_info,
            parquet_target=parquet_target,
        )

        # Adding label to the function - will be used to identify the stream pod
        fn.metadata.labels = {"type": "model-monitoring-stream"}

        mlrun.api.api.endpoints.functions._build_function(
            db_session=db_session,
            auth_info=auth_info,
            function=fn,
        )

    def deploy_model_monitoring_batch_processing(
        self,
        project: str,
        model_monitoring_access_key: str,
        db_session: sqlalchemy.orm.Session,
        auth_info: mlrun.common.schemas.AuthInfo,
        tracking_policy: mlrun.model_monitoring.tracking_policy.TrackingPolicy,
        with_schedule: bool = True,
        overwrite: bool = False,
        tracking_offset: Seconds = Seconds(0),
        function_name: str = mm_constants.MonitoringFunctionNames.BATCH,
    ) -> typing.Union[mlrun.runtimes.kubejob.KubejobRuntime, None]:
        """
        Deploying model monitoring batch job or model monitoring batch application job.
        The goal of this job is to identify drift in the data based on the latest batch of events. By default,
        this job is executed on the hour every hour.
        Note that if this job was already deployed then you will either have to pass overwrite=True or
        to delete the old job before deploying a new one.

        :param project:                     The name of the project.
        :param model_monitoring_access_key: Access key to apply the model monitoring process.
        :param db_session:                  A session that manages the current dialog with the database.
        :param auth_info:                   The auth info of the request.
        :param tracking_policy:             Model monitoring configurations.
        :param with_schedule:               If true, submit a scheduled batch drift job.
        :param overwrite:                   If true, overwrite the existing model monitoring batch job.
        :param tracking_offset:             Offset for the tracking policy (for synchronization with the stream)
        :param function_name:               model-monitoring-batch or model-monitoring-application-batch
                                            indicates witch one to deploy.

        :return: Model monitoring batch job as a runtime function.
        """
        job_valid_names = [
            mm_constants.MonitoringFunctionNames.BATCH,
            mm_constants.MonitoringFunctionNames.BATCH_APPLICATION,
        ]
        if function_name not in job_valid_names:
            raise mlrun.errors.MLRunRuntimeError(
                f"Model Monitoring batch job can be only within {job_valid_names}"
            )
        fn = None
        if not overwrite:
            logger.info(
                f"Checking if {function_name.replace('-',' ')} processing function is already deployed",
                project=project,
            )

            # Try to list functions that named model monitoring batch
            # to make sure that this job has not yet been deployed
            try:
                fn = mlrun.api.crud.Functions().get_function(
                    db_session=db_session,
                    name=function_name,
                    project=project,
                )
                logger.info(
                    f"Detected {function_name.replace('-',' ')} processing function already deployed",
                    project=project,
                )

            except mlrun.errors.MLRunNotFoundError:
                logger.info(
                    f"Deploying {function_name.replace('-',' ')} processing function ",
                    project=project,
                )

        if not fn:
            # Create a monitoring batch job function object
            fn = self._get_model_monitoring_batch_function(
                project=project,
                model_monitoring_access_key=model_monitoring_access_key,
                db_session=db_session,
                auth_info=auth_info,
                tracking_policy=tracking_policy,
                function_name=function_name,
            )

            # Get the function uri
            function_uri = fn.save(versioned=True)

            if with_schedule:
                if not overwrite:
                    try:
                        mlrun.api.utils.scheduler.Scheduler().get_schedule(
                            db_session=db_session,
                            project=project,
                            name=function_name,
                        )
                        logger.info(
                            f"Already deployed {function_name.replace('-',' ')} scheduled job function ",
                            project=project,
                        )
                        return
                    except mlrun.errors.MLRunNotFoundError:
                        logger.info(
                            f"Deploying {function_name.replace('-',' ')} scheduled job function ",
                            project=project,
                        )
                # Submit batch scheduled job
                self._submit_schedule_batch_job(
                    project=project,
                    function_uri=function_uri,
                    db_session=db_session,
                    auth_info=auth_info,
                    tracking_policy=tracking_policy,
                    tracking_offset=tracking_offset,
                    function_name=function_name,
                )
        return fn

    def deploy_model_monitoring_writer_application(
        self,
        project,
        model_monitoring_access_key,
        db_session,
        auth_info,
        tracking_policy,
    ):
        """
        Deploying model monitoring writer real time nuclio function. The goal of this real time function is
        to write all the monitoring application result to the databases. It is triggered by those applications.
        It processes and writes the result to the databases.

        :param project:                     The name of the project.
        :param model_monitoring_access_key: Access key to apply the model monitoring process.
        :param db_session:                  A session that manages the current dialog with the database.
        :param auth_info:                   The auth info of the request.
        :param tracking_policy:             Model monitoring configurations.
        """

        logger.info(
            "Checking if model monitoring writer is already deployed",
            project=project,
        )
        try:
            # validate that the model monitoring stream has not yet been deployed
            mlrun.runtimes.function.get_nuclio_deploy_status(
                name=mm_constants.MonitoringFunctionNames.WRITER,
                project=project,
                tag="",
                auth_info=auth_info,
            )
            logger.info(
                "Detected model monitoring writer processing function already deployed",
                project=project,
            )
            return
        except mlrun.errors.MLRunNotFoundError:
            logger.info(
                "Deploying model monitoring writer processing function", project=project
            )

        fn = self._initial_model_monitoring_writer_function(
            project=project,
            model_monitoring_access_key=model_monitoring_access_key,
            tracking_policy=tracking_policy,
            auth_info=auth_info,
        )

        # Adding label to the function - will be used to identify the stream pod
        fn.metadata.labels = {"type": "model-monitoring-writer"}

        mlrun.api.api.endpoints.functions._build_function(
            db_session=db_session,
            auth_info=auth_info,
            function=fn,
        )

    def _initial_model_monitoring_stream_processing_function(
        self,
        project: str,
        model_monitoring_access_key: str,
        tracking_policy: mlrun.model_monitoring.tracking_policy.TrackingPolicy,
        auth_info: mlrun.common.schemas.AuthInfo,
        parquet_target: str,
    ):
        """
        Initialize model monitoring stream processing function.

        :param project:                     Project name.
        :param model_monitoring_access_key: Access key to apply the model monitoring process. Please note that in CE
                                            deployments this parameter will be None.
        :param tracking_policy:             Model monitoring configurations.
        :param auth_info:                   The auth info of the request.
        :param parquet_target:              Path to model monitoring parquet file that will be generated by the
                                            monitoring stream nuclio function.

        :return:                            A function object from a mlrun runtime class

        """

        # Initialize Stream Processor object
        stream_processor = (
            mlrun.model_monitoring.stream_processing.EventStreamProcessor(
                project=project,
                parquet_batching_max_events=self._parquet_batching_max_events,
                parquet_batching_timeout_secs=self._max_parquet_save_interval,
                parquet_target=parquet_target,
                model_monitoring_access_key=model_monitoring_access_key,
            )
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
        function = self._apply_stream_trigger(
            project=project,
            function=function,
            model_monitoring_access_key=model_monitoring_access_key,
            auth_info=auth_info,
        )

        # Apply feature store run configurations on the serving function
        run_config = fstore.RunConfig(function=function, local=False)
        function.spec.parameters = run_config.parameters

        return function

    def _get_model_monitoring_batch_function(
        self,
        project: str,
        model_monitoring_access_key: str,
        db_session: sqlalchemy.orm.Session,
        auth_info: mlrun.common.schemas.AuthInfo,
        tracking_policy: mlrun.model_monitoring.tracking_policy.TrackingPolicy,
        function_name: str = "model-monitoring-batch",
    ):
        """
        Initialize model monitoring batch function.

        :param project:                     project name.
        :param model_monitoring_access_key: access key to apply the model monitoring process. Please note that in CE
                                            deployments this parameter will be None.
        :param db_session:                  A session that manages the current dialog with the database.
        :param auth_info:                   The auth info of the request.
        :param tracking_policy:             Model monitoring configurations.
        :param function_name:               model-monitoring-batch or model-monitoring-application-batch
                                            indicates witch one to create.
        :return:                            A function object from a mlrun runtime class

        """
        filename = (
            str(_MONITORING_ORIGINAL_BATCH_FUNCTION_PATH)
            if function_name == "model-monitoring-batch"
            else str(_MONITORING_APPLICATION_BATCH_FUNCTION_PATH)
        )
        # Create job function runtime for the model monitoring batch
        function: mlrun.runtimes.KubejobRuntime = mlrun.code_to_function(
            name=function_name,
            project=project,
            filename=filename,
            kind="job",
            image=tracking_policy.default_batch_image,
            handler="handler",
        )
        function.set_db_connection(mlrun.api.api.utils.get_run_db_instance(db_session))

        # Set the project to the job function
        function.metadata.project = project

        if not mlrun.mlconf.is_ce_mode():
            function = self._apply_access_key_and_mount_function(
                project=project,
                function=function,
                model_monitoring_access_key=model_monitoring_access_key,
                auth_info=auth_info,
                function_name=function_name,
            )

        # Enrich runtime with the required configurations
        mlrun.api.api.utils.apply_enrichment_and_validation_on_function(
            function, auth_info
        )

        return function

    @staticmethod
    def _submit_schedule_batch_job(
        project: str,
        function_uri: str,
        db_session: sqlalchemy.orm.Session,
        auth_info: mlrun.common.schemas.AuthInfo,
        tracking_policy: mlrun.model_monitoring.tracking_policy.TrackingPolicy,
        tracking_offset: Seconds = Seconds(0),
        function_name: str = "model-monitoring-batch",
    ):
        """
        Create a new scheduled monitoring batch job analysis based on the model-monitoring-batch function that has
        been already registered.

        :param project:         Project name.
        :param function_uri:    Function URI of the registered model monitoring batch job. This URI includes the
                                related project name, function name, and hash key.
        :param db_session:      A session that manages the current dialog with the database.
        :param auth_info:       The auth info of the request.
        :param tracking_policy: Model monitoring configurations.
        :param tracking_offset: Offset for the tracking policy (for synchronization with the stream).

        """

        function_uri = function_uri.replace("db://", "")

        task = mlrun.new_task(name=function_name, project=project)
        task.spec.function = function_uri

        # Apply batching interval params
        interval_list = [
            tracking_policy.default_batch_intervals.minute,
            tracking_policy.default_batch_intervals.hour,
            tracking_policy.default_batch_intervals.day,
        ]
        (
            minutes,
            hours,
            days,
        ) = mlrun.api.crud.model_monitoring.helpers.get_batching_interval_param(
            interval_list
        )
        batch_dict = {"minutes": minutes, "hours": hours, "days": days}

        task.spec.parameters[
            mlrun.common.schemas.model_monitoring.EventFieldType.BATCH_INTERVALS_DICT
        ] = batch_dict

        data = {
            "task": task.to_dict(),
            "schedule": mlrun.api.crud.model_monitoring.helpers.convert_to_cron_string(
                tracking_policy.default_batch_intervals,
                minute_delay=seconds2minutes(tracking_offset),
            ),
        }

        logger.info(
            f"Deploying {function_name.replace('-',' ')} processing function",
            project=project,
        )

        # Add job schedule policy (every hour by default)
        mlrun.api.api.utils.submit_run_sync(
            db_session=db_session, auth_info=auth_info, data=data
        )

    def _apply_stream_trigger(
        self,
        project: str,
        function: mlrun.runtimes.ServingRuntime,
        model_monitoring_access_key: str = None,
        auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
        function_name: str = None,
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
        :param function_name:             the name of the function that be applied with the stream trigger,
                                            None for model_monitoring_stream

        :return: ServingRuntime object with stream trigger.
        """

        # Get the stream path from the configuration
        # stream_path = mlrun.mlconf.get_file_target_path(project=project, kind="stream", target="stream")
        stream_path = mlrun.api.crud.model_monitoring.get_stream_path(
            project=project, application_name=function_name
        )

        if stream_path.startswith("kafka://"):
            topic, brokers = mlrun.datastore.utils.parse_kafka_url(url=stream_path)
            # Generate Kafka stream source
            stream_source = mlrun.datastore.sources.KafkaSource(
                brokers=brokers,
                topics=[topic],
            )
            function = stream_source.add_nuclio_trigger(function)

        if not mlrun.mlconf.is_ce_mode():
            function = self._apply_access_key_and_mount_function(
                project=project,
                function=function,
                model_monitoring_access_key=model_monitoring_access_key,
                auth_info=auth_info,
                function_name=function_name,
            )
            if stream_path.startswith("v3io://"):
                # Generate V3IO stream trigger
                function.add_v3io_stream_trigger(
                    stream_path=stream_path,
                    name="monitoring_stream_trigger"
                    if function_name is None
                    else f"monitoring_{function_name}_trigger",
                    access_key=model_monitoring_access_key
                    if function_name != mm_constants.MonitoringFunctionNames.STREAM
                    else None,
                )
        # Add the default HTTP source
        http_source = mlrun.datastore.sources.HttpSource()
        function = http_source.add_nuclio_trigger(function)

        return function

    @staticmethod
    def _apply_access_key_and_mount_function(
        project: str,
        function: typing.Union[
            mlrun.runtimes.KubejobRuntime, mlrun.runtimes.ServingRuntime
        ],
        model_monitoring_access_key: str,
        auth_info: mlrun.common.schemas.AuthInfo,
        function_name: str = None,
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

        if function_name in mm_constants.MonitoringFunctionNames.all():
            # Set model monitoring access key for managing permissions
            function.set_env_from_secret(
                mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ACCESS_KEY,
                mlrun.api.utils.singletons.k8s.get_k8s_helper().get_project_secret_name(
                    project
                ),
                mlrun.api.crud.secrets.Secrets().generate_client_project_secret_key(
                    mlrun.api.crud.secrets.SecretsClientType.model_monitoring,
                    mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ACCESS_KEY,
                ),
            )

            function.metadata.credentials.access_key = model_monitoring_access_key
            function.apply(mlrun.v3io_cred())

            # Ensure that the auth env vars are set
            mlrun.api.api.utils.ensure_function_has_auth_set(function, auth_info)
        return function

    def _initial_model_monitoring_writer_function(
        self, project, model_monitoring_access_key, tracking_policy, auth_info
    ):
        """
        Initialize model monitoring writer function.

        :param project:                     Project name.
        :param model_monitoring_access_key: Access key to apply the model monitoring process. Please note that in CE
                                            deployments this parameter will be None.
        :param tracking_policy:             Model monitoring configurations.
        :param auth_info:                   The auth info of the request.

        :return:                            A function object from a mlrun runtime class

        """

        # Create a new serving function for the streaming process
        function = mlrun.code_to_function(
            name=mm_constants.MonitoringFunctionNames.WRITER,
            project=project,
            filename=str(_MONITORING_WRITER_FUNCTION_PATH),
            kind="serving",
            image=tracking_policy.stream_image,
        )

        # Create writer monitoring serving graph
        graph = function.set_topology("flow")
        graph.to(ModelMonitoringWriter(project=project)).respond()  # writer

        # Set the project to the serving function
        function.metadata.project = project

        # Add stream triggers
        function = self._apply_stream_trigger(
            project=project,
            function=function,
            model_monitoring_access_key=model_monitoring_access_key,
            auth_info=auth_info,
            function_name=mm_constants.MonitoringFunctionNames.WRITER,
        )

        # Apply feature store run configurations on the serving function
        run_config = fstore.RunConfig(function=function, local=False)
        function.spec.parameters = run_config.parameters

        return function


def get_endpoint_features(
    feature_names: typing.List[str],
    feature_stats: dict = None,
    current_stats: dict = None,
) -> typing.List[mlrun.common.schemas.Features]:
    """
    Getting a new list of features that exist in feature_names along with their expected (feature_stats) and
    actual (current_stats) stats. The expected stats were calculated during the creation of the model endpoint,
    usually based on the data from the Model Artifact. The actual stats are based on the results from the latest
    model monitoring batch job.

    param feature_names: List of feature names.
    param feature_stats: Dictionary of feature stats that were stored during the creation of the model endpoint
                         object.
    param current_stats: Dictionary of the latest stats that were stored during the last run of the model monitoring
                         batch job.

    return: List of feature objects. Each feature has a name, weight, expected values, and actual values. More info
            can be found under `mlrun.common.schemas.Features`.
    """

    # Initialize feature and current stats dictionaries
    safe_feature_stats = feature_stats or {}
    safe_current_stats = current_stats or {}

    # Create feature object and add it to a general features list
    features = []
    for name in feature_names:
        if feature_stats is not None and name not in feature_stats:
            logger.warn("Feature missing from 'feature_stats'", name=name)
        if current_stats is not None and name not in current_stats:
            logger.warn("Feature missing from 'current_stats'", name=name)
        f = mlrun.common.schemas.Features.new(
            name, safe_feature_stats.get(name), safe_current_stats.get(name)
        )
        features.append(f)
    return features
