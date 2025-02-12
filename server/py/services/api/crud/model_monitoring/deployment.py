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

import asyncio
import json
import time
import traceback
import typing
import uuid
from asyncio import Semaphore
from http import HTTPStatus
from pathlib import Path

import fastapi
import kafka
import kafka.errors
import nuclio
import sqlalchemy.orm
import v3io.dataplane
import v3io.dataplane.response
from fastapi import BackgroundTasks
from fastapi.concurrency import run_in_threadpool

import mlrun.common.constants as mlrun_constants
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.datastore.datastore_profile
import mlrun.model_monitoring
import mlrun.model_monitoring.api
import mlrun.model_monitoring.applications
import mlrun.model_monitoring.controller
import mlrun.model_monitoring.stream_processing
import mlrun.model_monitoring.writer
import mlrun.serving.states
import mlrun.utils.v3io_clients
from mlrun import feature_store as fstore
from mlrun.config import config
from mlrun.model_monitoring.writer import ModelMonitoringWriter
from mlrun.platforms.iguazio import split_path
from mlrun.utils import logger

import framework.api.utils
import framework.db.session
import framework.utils.background_tasks
import framework.utils.singletons.k8s
import services.api.api.endpoints.nuclio
import services.api.crud.model_monitoring.helpers
import services.api.utils.functions
from framework.db.sqldb.db import unversioned_tagged_object_uid_prefix

_STREAM_PROCESSING_FUNCTION_PATH = mlrun.model_monitoring.stream_processing.__file__
_MONITORING_APPLICATION_CONTROLLER_FUNCTION_PATH = (
    mlrun.model_monitoring.controller.__file__
)
_MONITORING_WRITER_FUNCTION_PATH = mlrun.model_monitoring.writer.__file__
_HISTOGRAM_DATA_DRIFT_APP_PATH = str(
    Path(mlrun.model_monitoring.applications.__file__).parent
    / "histogram_data_drift.py"
)
BASE_PERIOD_LOOKUP_TABLE = {20: 2, 60: 5, 120: 10, float("inf"): 20}


class MonitoringDeployment:
    def __init__(
        self,
        project: str,
        auth_info: typing.Optional[mlrun.common.schemas.AuthInfo] = None,
        db_session: typing.Optional[sqlalchemy.orm.Session] = None,
        model_monitoring_access_key: typing.Optional[str] = None,
        parquet_batching_max_events: int = mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events,
        max_parquet_save_interval: int = mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs,
    ) -> None:
        """
        Initialize a MonitoringDeployment object, which handles the deployment & scheduling of:
         1. model monitoring stream (stream triggered by model servers)
         2. model monitoring controller (cron and HTTP triggers - self triggered every X minutes or manually via HTTP)
         3. model monitoring writer (stream triggered by user model monitoring functions)

        :param project:                     The name of the project.
        :param auth_info:                   The auth info of the request.
        :param db_session:                  A session that manages the current dialog with the database.
        :param model_monitoring_access_key: Access key to apply the model monitoring process.
        :param parquet_batching_max_events: Maximum number of events that will be used for writing the monitoring
                                            parquet by the monitoring stream function.
        :param max_parquet_save_interval:   Maximum number of seconds to hold events before they are written to the
                                            monitoring parquet target. Note that this value will be used to handle the
                                            offset by the scheduled batch job.
        """
        self.project = project
        self.auth_info = auth_info
        self.db_session = db_session
        self.model_monitoring_access_key = model_monitoring_access_key
        self._parquet_batching_max_events = parquet_batching_max_events
        self._max_parquet_save_interval = max_parquet_save_interval
        self._secret_provider = services.api.crud.secrets.get_project_secret_provider(
            project=project
        )
        self.__stream_profile = None

    @property
    def _stream_profile(self) -> mlrun.datastore.datastore_profile.DatastoreProfile:
        if not self.__stream_profile:
            self.__stream_profile = mlrun.model_monitoring.helpers._get_stream_profile(
                project=self.project, secret_provider=self._secret_provider
            )
        return self.__stream_profile

    def deploy_monitoring_functions(
        self,
        base_period: int = 10,
        image: str = "mlrun/mlrun",
        deploy_histogram_data_drift_app: bool = True,
        fetch_credentials_from_sys_config: bool = False,
    ) -> None:
        """
        Deploy model monitoring application controller, writer and stream functions.

        :param base_period:                       The time period in minutes in which the model monitoring controller
                                                  function triggers. By default, the base period is 10 minutes.
        :param image:                             The image of the model monitoring controller, writer & monitoring
                                                  stream functions, which are real time nuclio function.
                                                  By default, the image is mlrun/mlrun.
        :param deploy_histogram_data_drift_app:   If true, deploy the default histogram-based data drift application.
        :param fetch_credentials_from_sys_config: If true, fetch the credentials from the system configuration.
        """
        # check if credentials should be fetched from the system configuration or if they are already been set.
        if fetch_credentials_from_sys_config:
            self.set_credentials()
        self.check_if_credentials_are_set()

        self.deploy_model_monitoring_controller(
            controller_image=image, base_period=base_period
        )
        self.deploy_model_monitoring_writer_application(
            writer_image=image,
        )
        self.deploy_model_monitoring_stream_processing(
            stream_image=image,
        )
        if deploy_histogram_data_drift_app:
            self.deploy_histogram_data_drift_app(image=image)

    def deploy_model_monitoring_stream_processing(
        self, stream_image: str = "mlrun/mlrun", overwrite: bool = False
    ) -> None:
        """
        Deploying model monitoring stream real time nuclio function. The goal of this real time function is
        to monitor the log of the data stream. It is triggered when a new log entry is detected.
        It processes the new events into statistics that are then written to statistics databases.

        :param stream_image:                The image of the model monitoring stream function.
                                            By default, the image is mlrun/mlrun.
        :param overwrite:                   If true, overwrite the existing model monitoring stream. Default is False.
        """

        if (
            overwrite
            or self._get_function_state(
                function_name=mm_constants.MonitoringFunctionNames.STREAM,
            )
            != "ready"
        ):
            logger.info(
                f"Deploying {mm_constants.MonitoringFunctionNames.STREAM} function",
                project=self.project,
            )
            # Get parquet target value for model monitoring stream function
            parquet_target = (
                services.api.crud.model_monitoring.helpers.get_monitoring_parquet_path(
                    db_session=self.db_session, project=self.project
                )
            )

            fn = self._initial_model_monitoring_stream_processing_function(
                stream_image=stream_image, parquet_target=parquet_target
            )
            fn, ready = services.api.utils.functions.build_function(
                db_session=self.db_session, auth_info=self.auth_info, function=fn
            )
            logger.debug(
                "Submitted the stream deployment",
                stream_data=fn.to_dict(),
                stream_ready=ready,
            )

    def deploy_model_monitoring_controller(
        self,
        base_period: int,
        controller_image: str = "mlrun/mlrun",
        overwrite: bool = False,
    ) -> None:
        """
        Deploy model monitoring application controller function.
        The main goal of the controller function is to handle the monitoring processing and triggering applications.
        The controller is self triggered by a cron. It also has the default HTTP trigger.

        :param base_period:                 The time period in minutes in which the model monitoring controller function
                                            triggers. By default, the base period is 10 minutes.
        :param controller_image:            The image of the model monitoring controller function.
                                            By default, the image is mlrun/mlrun.
        :param overwrite:                   If true, overwrite the existing model monitoring controller.
                                            By default, False.
        """
        if (
            overwrite
            or self._get_function_state(
                function_name=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            )
            != "ready"
        ):
            logger.info(
                f"Deploying {mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER} function",
                project=self.project,
            )
            fn = self._get_model_monitoring_controller_function(
                image=controller_image, ignore_stream_already_exists_failure=overwrite
            )
            minutes = base_period
            hours = days = 0
            batch_dict = {
                mm_constants.EventFieldType.MINUTES: minutes,
                mm_constants.EventFieldType.HOURS: hours,
                mm_constants.EventFieldType.DAYS: days,
            }
            fn.set_env(
                mm_constants.EventFieldType.BATCH_INTERVALS_DICT,
                json.dumps(batch_dict),
            )

            fn.add_trigger(
                "cron_interval",
                spec=nuclio.CronTrigger(
                    interval=f"{self._get_trigger_frequency(base_period)}m"
                ),
            )
            fn, ready = services.api.utils.functions.build_function(
                db_session=self.db_session, auth_info=self.auth_info, function=fn
            )

            logger.debug(
                "Submitted the controller deployment",
                controller_data=fn.to_dict(),
                controller_ready=ready,
            )

    def deploy_model_monitoring_writer_application(
        self, writer_image: str = "mlrun/mlrun", overwrite: bool = False
    ) -> None:
        """
        Deploying model monitoring writer real time nuclio function. The goal of this real time function is
        to write all the monitoring application result to the databases. It is triggered by those applications.
        It processes and writes the result to the databases.

        :param writer_image:                The image of the model monitoring writer function.
                                            By default, the image is mlrun/mlrun.
        :param overwrite:                   If true, overwrite the existing model monitoring writer. Default is False.
        """

        if (
            overwrite
            or self._get_function_state(
                function_name=mm_constants.MonitoringFunctionNames.WRITER,
            )
            != "ready"
        ):
            logger.info(
                f"Deploying {mm_constants.MonitoringFunctionNames.WRITER} function",
                project=self.project,
            )
            fn = self._initial_model_monitoring_writer_function(
                writer_image=writer_image
            )
            fn, ready = services.api.utils.functions.build_function(
                db_session=self.db_session, auth_info=self.auth_info, function=fn
            )
            logger.debug(
                "Submitted the writer deployment",
                writer_data=fn.to_dict(),
                writer_ready=ready,
            )

    def apply_and_create_stream_trigger(
        self,
        function: mlrun.runtimes.ServingRuntime,
        function_name: str,
        stream_args: mlrun.config.Config,
        ignore_stream_already_exists_failure: bool = False,
    ) -> mlrun.runtimes.ServingRuntime:
        """
        Add stream source for the nuclio serving function. The function's stream trigger can be
        either Kafka or V3IO, depends on the stream profile defined by::

            project.set_model_monitoring_credentials(stream_profile_name="...", ...)

        Note: this method also disables the default HTTP trigger of the function, so it remains
        only with stream trigger(s).

        :param function:                             The serving function object that will be applied with the stream
                                                     trigger.
        :param function_name:                        The name of the function that be applied with the stream trigger.
        :param stream_args:                          Stream args from the config.
        :param ignore_stream_already_exists_failure: If True, ignores `TopicAlreadyExistsError` error on
                                                     MM-infra-functions deployment when using kafka.

        :return: `ServingRuntime` object with stream trigger.
        """
        profile = self._stream_profile
        if isinstance(
            profile, mlrun.datastore.datastore_profile.DatastoreProfileKafkaSource
        ):
            self._apply_and_create_kafka_source(
                kafka_profile=profile,
                function=function,
                function_name=function_name,
                stream_args=stream_args,
                ignore_stream_already_exists_failure=ignore_stream_already_exists_failure,
            )

        elif isinstance(
            profile, mlrun.datastore.datastore_profile.DatastoreProfileV3io
        ):
            self._apply_and_create_v3io_source(
                v3io_profile=profile,
                function=function,
                function_name=function_name,
                stream_args=stream_args,
            )
        else:
            framework.api.utils.log_and_raise(
                HTTPStatus.BAD_REQUEST.value,
                reason="Unexpected stream profile",
            )

        if not mlrun.mlconf.is_ce_mode():
            function = self._apply_access_key_and_mount_function(
                function=function, function_name=function_name
            )

        if function_name != mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER:
            function.spec.disable_default_http_trigger = True

        return function

    def _apply_and_create_kafka_source(
        self,
        *,
        kafka_profile: mlrun.datastore.datastore_profile.DatastoreProfileKafkaSource,
        function: mlrun.runtimes.ServingRuntime,
        function_name: str,
        stream_args: mlrun.config.Config,
        ignore_stream_already_exists_failure: bool,
    ) -> None:
        # Generate Kafka stream source
        topic = mlrun.common.model_monitoring.helpers.get_kafka_topic(
            project=self.project, function_name=function_name
        )
        stream_source = mlrun.datastore.sources.KafkaSource(
            brokers=kafka_profile.brokers,
            topics=[topic],
            group=kafka_profile.group,
            initial_offset=kafka_profile.initial_offset,
            partitions=kafka_profile.partitions,
            attributes=kafka_profile.attributes()
            | {
                "max_workers": stream_args.kafka.num_workers,
                "worker_allocation_mode": "static",
            },
        )
        try:
            stream_source.create_topics(
                num_partitions=stream_args.kafka.partition_count,
                replication_factor=stream_args.kafka.replication_factor,
            )
        except kafka.errors.TopicAlreadyExistsError as exc:
            if ignore_stream_already_exists_failure:
                logger.info(
                    "Kafka topic of model monitoring stream already exists. "
                    "Skipping topic creation and using `earliest` offset",
                    project=self.project,
                    error_message=mlrun.errors.err_to_str(exc),
                )
            else:
                raise exc

        function = stream_source.add_nuclio_trigger(function)
        function.spec.min_replicas = stream_args.kafka.min_replicas
        function.spec.max_replicas = stream_args.kafka.max_replicas

    def _apply_and_create_v3io_source(
        self,
        *,
        v3io_profile: mlrun.datastore.datastore_profile.DatastoreProfileV3io,
        function: mlrun.runtimes.ServingRuntime,
        function_name: str,
        stream_args: mlrun.config.Config,
    ) -> None:
        stream_path = mlrun.mlconf.get_model_monitoring_file_target_path(
            project=self.project,
            kind=mm_constants.FileTargetKind.STREAM,
            target="online",
            function_name=function_name,
        )

        access_key = (
            v3io_profile.v3io_access_key
            if function_name
            != mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER
            else mlrun.mlconf.get_v3io_access_key()
        )
        kwargs = {"access_key": access_key}
        if mlrun.mlconf.is_explicit_ack_enabled():
            kwargs["explicit_ack_mode"] = "explicitOnly"
        kwargs["worker_allocation_mode"] = "static"
        kwargs["max_workers"] = stream_args.v3io.num_workers
        services.api.api.endpoints.nuclio.create_model_monitoring_stream(
            project=self.project,
            stream_path=stream_path,
            shard_count=stream_args.v3io.shard_count,
            retention_period_hours=stream_args.v3io.retention_period_hours,
            access_key=access_key,
        )
        # Generate V3IO stream trigger
        function.add_v3io_stream_trigger(
            stream_path=stream_path,
            name=f"monitoring_{function_name}_trigger",
            **kwargs,
        )
        function.spec.min_replicas = stream_args.v3io.min_replicas
        function.spec.max_replicas = stream_args.v3io.max_replicas

    def _initial_model_monitoring_stream_processing_function(
        self,
        stream_image: str,
        parquet_target: str,
    ):
        """
        Initialize model monitoring stream processing function.

        :param stream_image:   The image of the model monitoring stream function.
        :param parquet_target: Path to model monitoring parquet file that will be generated by the
                               monitoring stream nuclio function.

        :return:               A function object from a mlrun runtime class
        """

        # Initialize Stream Processor object
        stream_processor = (
            mlrun.model_monitoring.stream_processing.EventStreamProcessor(
                project=self.project,
                parquet_batching_max_events=self._parquet_batching_max_events,
                parquet_batching_timeout_secs=self._max_parquet_save_interval,
                parquet_target=parquet_target,
                model_monitoring_access_key=self.model_monitoring_access_key,
            )
        )

        # Create a new serving function for the streaming process
        function = typing.cast(
            mlrun.runtimes.ServingRuntime,
            mlrun.code_to_function(
                name=mm_constants.MonitoringFunctionNames.STREAM,
                project=self.project,
                filename=_STREAM_PROCESSING_FUNCTION_PATH,
                kind=mlrun.run.RuntimeKinds.serving,
                image=stream_image,
                # The label is used to identify the stream function in Prometheus
                labels={"type": mm_constants.MonitoringFunctionNames.STREAM},
            ),
        )
        function.set_db_connection(
            framework.api.utils.get_run_db_instance(self.db_session)
        )

        tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
            project=self.project, secret_provider=self._secret_provider
        )

        controller_stream_uri = mlrun.model_monitoring.get_stream_path(
            project=self.project,
            function_name=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            secret_provider=self._secret_provider,
        )

        # Create monitoring serving graph
        stream_processor.apply_monitoring_serving_graph(
            function, tsdb_connector, controller_stream_uri
        )

        # Set the project to the serving function
        function.metadata.project = self.project

        # Add stream triggers
        function = self.apply_and_create_stream_trigger(
            function=function,
            function_name=mm_constants.MonitoringFunctionNames.STREAM,
            stream_args=config.model_endpoint_monitoring.serving_stream,
            ignore_stream_already_exists_failure=True,
        )

        # Apply feature store run configurations on the serving function
        run_config = fstore.RunConfig(function=function, local=False)
        function.spec.parameters = run_config.parameters

        return function

    def _get_model_monitoring_controller_function(
        self, image: str, ignore_stream_already_exists_failure: bool
    ):
        """
        Initialize model monitoring controller function.

        :param image:                               Base docker image to use for building the function container.
        :param ignore_stream_already_exists_failure: If True, ignores `TopicAlreadyExistsError` error on
                                                     MM-infra-functions deployment when using kafka.
        :return:                                    A function object from a mlrun runtime class.
        """
        # Create job function runtime for the controller
        function = mlrun.code_to_function(
            name=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            project=self.project,
            filename=_MONITORING_APPLICATION_CONTROLLER_FUNCTION_PATH,
            kind=mlrun.run.RuntimeKinds.nuclio,
            image=image,
            handler="handler",
        )
        function.set_db_connection(
            framework.api.utils.get_run_db_instance(self.db_session)
        )

        # Set the project to the job function
        function.metadata.project = self.project

        # Add stream triggers
        function = self.apply_and_create_stream_trigger(
            function=function,
            function_name=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            stream_args=config.model_endpoint_monitoring.controller_stream_args,
            ignore_stream_already_exists_failure=ignore_stream_already_exists_failure,
        )

        function = self._apply_access_key_and_mount_function(
            function=function,
            function_name=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
        )
        # Enrich runtime with the required configurations
        framework.api.utils.apply_enrichment_and_validation_on_function(
            function, self.auth_info
        )

        return function

    def _apply_access_key_and_mount_function(
        self,
        function: typing.Union[
            mlrun.runtimes.KubejobRuntime, mlrun.runtimes.ServingRuntime
        ],
        function_name: typing.Optional[str] = None,
    ) -> typing.Union[mlrun.runtimes.KubejobRuntime, mlrun.runtimes.ServingRuntime]:
        """Applying model monitoring access key on the provided function when using V3IO path. In addition, this method
        mount the V3IO path for the provided function to configure the access to the system files.

        :param function:                    Model monitoring function object that will be filled with the access key and
                                            the access to the system files.

        :return: function runtime object with access key and access to system files.
        """

        if (
            function_name in mm_constants.MonitoringFunctionNames.list()
            and not mlrun.mlconf.is_ce_mode()
        ):
            # Set model monitoring access key for managing permissions
            function.set_env_from_secret(
                mm_constants.ProjectSecretKeys.ACCESS_KEY,
                framework.utils.singletons.k8s.get_k8s_helper().get_project_secret_name(
                    self.project
                ),
                services.api.crud.secrets.Secrets().generate_client_project_secret_key(
                    services.api.crud.secrets.SecretsClientType.model_monitoring,
                    mm_constants.ProjectSecretKeys.ACCESS_KEY,
                ),
            )

            function.metadata.credentials.access_key = self.model_monitoring_access_key
            function.apply(mlrun.v3io_cred())

            # Ensure that the auth env vars are set
            framework.api.utils.ensure_function_has_auth_set(function, self.auth_info)
        return function

    def _initial_model_monitoring_writer_function(self, writer_image: str):
        """
        Initialize model monitoring writer function.

        :param writer_image:                The image of the model monitoring writer function.

        :return:                            A function object from a mlrun runtime class
        """

        # Create a new serving function for the streaming process
        function = typing.cast(
            mlrun.runtimes.ServingRuntime,
            mlrun.code_to_function(
                name=mm_constants.MonitoringFunctionNames.WRITER,
                project=self.project,
                filename=_MONITORING_WRITER_FUNCTION_PATH,
                kind=mlrun.run.RuntimeKinds.serving,
                image=writer_image,
            ),
        )
        function.set_db_connection(
            framework.api.utils.get_run_db_instance(self.db_session)
        )

        # Create writer monitoring serving graph
        graph = function.set_topology(mlrun.serving.states.StepKinds.flow)
        graph.to(
            ModelMonitoringWriter(
                project=self.project, secret_provider=self._secret_provider
            )
        )  # writer

        # Set the project to the serving function
        function.metadata.project = self.project

        # Add stream triggers
        function = self.apply_and_create_stream_trigger(
            function=function,
            function_name=mm_constants.MonitoringFunctionNames.WRITER,
            stream_args=config.model_endpoint_monitoring.writer_stream_args,
            ignore_stream_already_exists_failure=True,
        )

        # Apply feature store run configurations on the serving function
        run_config = fstore.RunConfig(function=function, local=False)
        function.spec.parameters = run_config.parameters

        return function

    def _get_function_state(
        self,
        function_name: str,
    ) -> typing.Optional[str]:
        """
        :param function_name:   The name of the function to check.

        :return:                Function state if deployed, else None.
        """
        logger.info(
            f"Checking if {function_name} is already deployed",
            project=self.project,
        )
        try:
            # validate that the function has not yet been deployed
            state, _, _, _, _, _ = (
                mlrun.runtimes.nuclio.function.get_nuclio_deploy_status(
                    name=function_name,
                    project=self.project,
                    tag="",
                    auth_info=self.auth_info,
                )
            )
            logger.info(
                f"Detected {function_name} function already deployed",
                project=self.project,
                state=state,
            )
            return state
        except mlrun.errors.MLRunNotFoundError:
            pass

    def deploy_histogram_data_drift_app(
        self, image: str, overwrite: bool = False
    ) -> None:
        """
        Deploy the histogram data drift application.

        :param image:       The image on with the function will run.
        :param overwrite:   If True, the function will be overwritten.
        """
        if (
            overwrite
            or self._get_function_state(
                function_name=mm_constants.HistogramDataDriftApplicationConstants.NAME,
            )
            != "ready"
        ):
            logger.info("Preparing the histogram data drift function")
            func = mlrun.model_monitoring.api._create_model_monitoring_function_base(
                project=self.project,
                func=_HISTOGRAM_DATA_DRIFT_APP_PATH,
                name=mm_constants.HistogramDataDriftApplicationConstants.NAME,
                application_class="HistogramDataDriftApplication",
                image=image,
            )

            if not mlrun.mlconf.is_ce_mode():
                logger.info(
                    "Setting the access key for the histogram data drift function"
                )
                func.metadata.credentials.access_key = self.model_monitoring_access_key
                framework.api.utils.ensure_function_has_auth_set(func, self.auth_info)
                logger.info("Ensured the histogram data drift function auth")

            func.set_label(
                mm_constants.ModelMonitoringAppLabel.KEY,
                mm_constants.ModelMonitoringAppLabel.VAL,
            )

            fn, ready = services.api.utils.functions.build_function(
                db_session=self.db_session, auth_info=self.auth_info, function=func
            )

            logger.debug(
                "Submitted the histogram data drift app deployment",
                app_data=fn.to_dict(),
                app_ready=ready,
            )

    def _create_tsdb_tables(
        self, tsdb_profile: mlrun.datastore.datastore_profile.DatastoreProfile
    ) -> None:
        """
        Create the TSDB tables using the TSDB connector. At the moment we support 3 types of tables:
        - app_results: a detailed result that includes status, kind, extra data, etc.
        - metrics: a basic key value that represents a numeric metric.
        - predictions: latency of each prediction.
        """
        mlrun.model_monitoring.get_tsdb_connector(
            project=self.project, profile=tsdb_profile
        ).create_tables()

    def list_model_monitoring_functions(self) -> list:
        """Retrieve a list of all the model monitoring functions."""
        model_monitoring_labels_list = [
            f"{mm_constants.ModelMonitoringAppLabel.KEY}={mm_constants.ModelMonitoringAppLabel.VAL}"
        ]
        return services.api.crud.Functions().list_functions(
            db_session=self.db_session,
            project=self.project,
            labels=model_monitoring_labels_list,
        )

    async def disable_model_monitoring(
        self,
        delete_resources: bool = True,
        delete_stream_function: bool = False,
        delete_histogram_data_drift_app: bool = True,
        delete_user_applications: bool = False,
        user_application_list: typing.Optional[list[str]] = None,
        background_tasks: fastapi.BackgroundTasks = None,
    ) -> mlrun.common.schemas.BackgroundTaskList:
        """
        Disable model monitoring application controller, writer, stream, histogram data drift application
        and the user's applications functions, according to the given params.

        :param delete_resources:                    If True, delete the model monitoring controller & writer functions.
                                                    Default True.
        :param delete_stream_function:              If True, delete model monitoring stream function,
                                                    need to use wisely because if you're deleting this function
                                                    this can cause data loss in case you will want to
                                                    enable the model monitoring capability to the project.
                                                    Default False.
        :param delete_histogram_data_drift_app:     If True, it would delete the default histogram-based data drift
                                                    application. Default False.
        :param delete_user_applications:            If True, it would delete the user's model monitoring
                                                    application according to user_application_list, Default False.
        :param user_application_list:               List of the user's model monitoring application to disable.
                                                    Default all the applications.
                                                    Note: you have to set delete_user_applications to True
                                                    in order to delete the desired application.
        :param background_tasks:                    Fastapi Background tasks.
        """
        function_to_delete = []
        if delete_resources:
            function_to_delete = mm_constants.MonitoringFunctionNames.list()
        if not delete_stream_function and delete_resources:
            function_to_delete.remove(mm_constants.MonitoringFunctionNames.STREAM)

        function_to_delete.extend(
            self._get_monitoring_application_to_delete(
                delete_histogram_data_drift_app,
                delete_user_applications,
                user_application_list,
            )
        )
        tasks: list[mlrun.common.schemas.BackgroundTask] = []
        for function_name in function_to_delete:
            if self._get_function_state(function_name):
                task = await run_in_threadpool(
                    framework.db.session.run_function_with_new_db_session,
                    MonitoringDeployment._create_monitoring_function_deletion_background_task,
                    background_tasks=background_tasks,
                    project_name=self.project,
                    function_name=function_name,
                    auth_info=self.auth_info,
                    delete_app_stream_resources=function_name
                    != mm_constants.MonitoringFunctionNames.STREAM,
                )
                tasks.append(task)

        return mlrun.common.schemas.BackgroundTaskList(background_tasks=tasks)

    def _get_monitoring_application_to_delete(
        self,
        delete_histogram_data_drift_app: bool = True,
        delete_user_applications: bool = False,
        user_application_list: typing.Optional[list[str]] = None,
    ):
        application_to_delete = []

        if delete_user_applications:
            if not user_application_list:
                application_to_delete.extend(
                    list(
                        {
                            app["metadata"]["name"]
                            for app in self.list_model_monitoring_functions()
                        }
                    )
                )
            else:
                for name in user_application_list:
                    try:
                        fn = services.api.crud.Functions().get_function(
                            db_session=self.db_session,
                            name=name,
                            project=self.project,
                        )
                        if (
                            fn["metadata"]["labels"].get(
                                mm_constants.ModelMonitoringAppLabel.KEY
                            )
                            == mm_constants.ModelMonitoringAppLabel.VAL
                        ):
                            # checks if the given function is a model monitoring application
                            application_to_delete.append(name)
                        else:
                            logger.warning(
                                f"{name} is not a model monitoring application, skipping",
                                project=self.project,
                            )

                    except mlrun.errors.MLRunNotFoundError:
                        logger.warning(
                            f"{name} is not found, skipping",
                        )

        if (
            delete_histogram_data_drift_app
            and mm_constants.HistogramDataDriftApplicationConstants.NAME
            not in application_to_delete
        ):
            application_to_delete.append(
                mm_constants.HistogramDataDriftApplicationConstants.NAME
            )
        return application_to_delete

    @staticmethod
    def _create_monitoring_function_deletion_background_task(
        db_session: sqlalchemy.orm.Session,
        background_tasks: BackgroundTasks,
        project_name: str,
        function_name: str,
        auth_info: mlrun.common.schemas.AuthInfo,
        delete_app_stream_resources: bool,
    ):
        background_task_name = str(uuid.uuid4())

        # create the background task for function deletion
        return framework.utils.background_tasks.ProjectBackgroundTasksHandler().create_background_task(
            db_session,
            project_name,
            background_tasks,
            MonitoringDeployment.delete_monitoring_function,
            mlrun.mlconf.background_tasks.default_timeouts.operations.delete_function,
            background_task_name,
            db_session,
            project_name,
            function_name,
            auth_info,
            background_task_name,
            delete_app_stream_resources,
        )

    @staticmethod
    async def delete_monitoring_function(
        db_session: sqlalchemy.orm.Session,
        project: str,
        function_name: str,
        auth_info: mlrun.common.schemas.AuthInfo,
        background_task_name: str,
        delete_app_stream_resources: bool,
    ) -> None:
        """
        Delete the model monitoring function and its resources.

        :param db_session:                  A session that manages the current dialog with the database.
        :param project:                     The name of the project.
        :param function_name:               The name of the function to delete.
        :param auth_info:                   The auth info of the request.
        :param background_task_name:        The name of the background task.
        :param delete_app_stream_resources: If True, delete the stream resources (e.g., v3io stream or kafka  topics).
        :param access_key:                  Model monitoring access key, relevant only for V3IO stream.
        """
        await framework.api.utils._delete_function(
            db_session=db_session,
            project=project,
            function_name=function_name,
            auth_info=auth_info,
            background_task_name=background_task_name,
        )
        if delete_app_stream_resources:
            try:
                MonitoringDeployment(
                    project=project
                )._delete_model_monitoring_stream_resources(
                    function_names=[function_name]
                )
            except mlrun.errors.MLRunStreamConnectionFailureError as e:
                logger.warning(
                    "Failed to delete stream resources, you may need to delete them manually",
                    project_name=project,
                    function=function_name,
                    error=mlrun.errors.err_to_str(e),
                )

    def _delete_model_monitoring_stream_resources(
        self,
        function_names: list[str],
        stream_profile: typing.Optional[
            mlrun.datastore.datastore_profile.DatastoreProfile
        ] = None,
    ) -> None:
        """
        :param function_names: A list of functions that their resources should be deleted.
        :param stream_profile: An optional datastore profile for the stream.
        """
        logger.debug(
            "Deleting model monitoring stream resources deployment",
            project_name=self.project,
        )
        profile = stream_profile or self._stream_profile
        stream_paths = []
        for function_name in function_names:
            qualified_function_name = f"{self.project}-{function_name}"
            if len(qualified_function_name) > 63:
                logger.info(
                    "k8s 63 characters limit exceeded, skipping deletion of stream resources",
                    project_name=self.project,
                    function_label_name=qualified_function_name,
                )
                continue
            label_selector = f"{mlrun_constants.MLRunInternalLabels.nuclio_function_name}={qualified_function_name}"
            for _ in range(10):
                # waiting for the function pod to be deleted
                # max 10 retries (5 sec sleep between each retry)
                try:
                    function_pod = (
                        framework.utils.singletons.k8s.get_k8s_helper().list_pods(
                            selector=label_selector
                        )
                    )
                except Exception as exc:
                    raise mlrun.errors.MLRunStreamConnectionFailureError(
                        f"Failed to list pods for function {function_name}"
                    ) from exc
                if not function_pod:
                    logger.debug(
                        "No function pod found for project, deleting stream",
                        project_name=self.project,
                        function=function_name,
                    )
                    break
                else:
                    logger.debug(f"{function_name} pod found, retrying")
                    time.sleep(5)

            stream_paths.append(
                mlrun.model_monitoring.get_stream_path(
                    project=self.project,
                    function_name=function_name,
                    secret_provider=self._secret_provider,
                    profile=stream_profile,
                )
            )

        if not stream_paths:
            # No stream paths to delete
            return

        elif isinstance(
            profile, mlrun.datastore.datastore_profile.DatastoreProfileV3io
        ):
            # Delete V3IO stream
            v3io_client = v3io.dataplane.Client(endpoint=mlrun.mlconf.v3io_api)

            for stream_path in stream_paths:
                _, container, stream_path = (
                    mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
                        stream_path
                    )
                )
                logger.debug(
                    "Deleting v3io stream",
                    project=self.project,
                    stream_path=stream_path,
                )
                try:
                    # if the stream path is in the users directory, we need to use pipelines access key to delete it
                    v3io_client.stream.delete(
                        container,
                        stream_path,
                        access_key=mlrun.mlconf.get_v3io_access_key()
                        if container.startswith("users")
                        else profile.v3io_access_key,
                    )
                    logger.debug(
                        "Deleted v3io stream",
                        project=self.project,
                        stream_path=stream_path,
                    )
                except Exception as exc:
                    # Raise an error that will be caught by the caller and skip the deletion of the stream
                    raise mlrun.errors.MLRunStreamConnectionFailureError(
                        f"Failed to delete v3io stream {stream_path}"
                    ) from exc
        elif isinstance(
            profile, mlrun.datastore.datastore_profile.DatastoreProfileKafkaSource
        ):
            # Delete Kafka topics
            topics = [
                mlrun.datastore.utils.parse_kafka_url(url=stream_path)[0]
                for stream_path in stream_paths
            ]

            kafka_profile_attributes = profile.attributes()
            kafka_admin_client_kwargs = {}
            if "sasl" in kafka_profile_attributes:
                sasl = kafka_profile_attributes["sasl"]
                kafka_admin_client_kwargs.update(
                    {
                        "security_protocol": "SASL_PLAINTEXT",
                        "sasl_mechanism": sasl["mechanism"],
                        "sasl_plain_username": sasl["user"],
                        "sasl_plain_password": sasl["password"],
                    }
                )

            client_id = f"{mlrun.mlconf.system_id}_{self.project}_kafka-python_{kafka.__version__}"

            try:
                kafka_client = kafka.KafkaAdminClient(
                    bootstrap_servers=profile.brokers,
                    client_id=client_id,
                    **kafka_admin_client_kwargs,
                )
                kafka_client.delete_topics(topics)
                logger.debug("Deleted kafka topics", topics=topics)
            except Exception as exc:
                # Raise an error that will be caught by the caller and skip the deletion of the stream
                raise mlrun.errors.MLRunStreamConnectionFailureError(
                    "Failed to delete kafka topics"
                ) from exc
        else:
            logger.warning(
                "Stream profile is not supported and therefore can't be deleted, expected v3io or kafka",
                stream_profile_type=str(type(profile)),
            )
        logger.debug(
            "Successfully deleted model monitoring stream resources deployment",
            project_name=self.project,
        )

    def _get_monitoring_mandatory_project_secrets(self) -> dict[str, str]:
        credentials_dict = {
            key: mlrun.get_secret_or_env(key, secret_provider=self._secret_provider)
            for key in mlrun.common.schemas.model_monitoring.ProjectSecretKeys.mandatory_secrets()
        }

        return credentials_dict

    def check_if_credentials_are_set(self) -> None:
        """
        Check if the model monitoring credentials are set. If not, raise an error.

        :raise mlrun.errors.MLRunBadRequestError:  if the credentials are not set.
        """

        credentials_dict = self._get_monitoring_mandatory_project_secrets()
        if all([val is not None for val in credentials_dict.values()]):
            return

        raise mlrun.errors.MLRunBadRequestError(
            "Model monitoring credentials are not set. "
            "Please set them using the set_model_monitoring_credentials API/SDK "
            "or pass fetch_credentials_from_sys_config=True when using enable_model_monitoring API/SDK."
        )

    def _validate_and_get_tsdb_profile(
        self, tsdb_profile_name: str
    ) -> mlrun.datastore.datastore_profile.DatastoreProfile:
        try:
            tsdb_profile = mlrun.datastore.datastore_profile.datastore_profile_read(
                url=f"ds://{tsdb_profile_name}",
                project_name=self.project,
                secrets=self._secret_provider,
            )
        except mlrun.errors.MLRunNotFoundError:
            raise mlrun.errors.MLRunNotFoundError(
                f"The given model monitoring TSDB profile name '{tsdb_profile_name}' "
                "was not found. Please make sure to register it properly in the project with "
                "`project.register_datastore_profile(tsdb_profile)`."
            )

        if isinstance(
            tsdb_profile, mlrun.datastore.datastore_profile.DatastoreProfileV3io
        ):
            if mlrun.mlconf.is_ce_mode():
                raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                    "MLRun CE supports only TDEngine TSDB, received a V3IO profile for the TSDB"
                )
        elif not isinstance(
            tsdb_profile, mlrun.datastore.datastore_profile.TDEngineDatastoreProfile
        ):
            raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                f"The model monitoring TSDB profile is of an unexpected type: '{type(tsdb_profile)}'\n"
                "Expects `DatastoreProfileV3io` or `TDEngineDatastoreProfile`."
            )

        return tsdb_profile

    def _validate_stream_profile(self, stream_profile_name: str) -> None:
        try:
            stream_profile = mlrun.datastore.datastore_profile.datastore_profile_read(
                url=f"ds://{stream_profile_name}",
                project_name=self.project,
                secrets=self._secret_provider,
            )
        except mlrun.errors.MLRunNotFoundError:
            raise mlrun.errors.MLRunNotFoundError(
                f"The given model monitoring stream profile name '{stream_profile_name}' "
                "was not found. Please make sure to register it properly in the project with "
                "`project.register_datastore_profile(stream_profile)`."
            )
        if isinstance(
            stream_profile,
            mlrun.datastore.datastore_profile.DatastoreProfileKafkaSource,
        ):
            self._validate_kafka_stream(stream_profile)
        elif isinstance(
            stream_profile, mlrun.datastore.datastore_profile.DatastoreProfileV3io
        ):
            self._validate_v3io_stream(stream_profile)
        else:
            raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                f"The model monitoring stream profile is of an unexpected type: '{type(stream_profile)}'\n"
                "Expects `DatastoreProfileV3io` or `DatastoreProfileKafkaSource`."
            )

    def _validate_kafka_stream(
        self,
        kafka_profile: mlrun.datastore.datastore_profile.DatastoreProfileKafkaSource,
    ) -> None:
        if kafka_profile.topics:
            raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                "Custom Kafka topics are not supported"
            )
        self._verify_kafka_access(kafka_profile)

    @staticmethod
    def _verify_kafka_access(
        kafka_profile: mlrun.datastore.datastore_profile.DatastoreProfileKafkaSource,
    ) -> None:
        import kafka.errors

        kafka_brokers = kafka_profile.brokers
        try:
            # The following constructor attempts to establish a connection
            consumer = kafka.KafkaConsumer(bootstrap_servers=kafka_brokers)
        except kafka.errors.NoBrokersAvailable as err:
            logger.warn(
                "No Kafka brokers available for the given kafka source profile in model monitoring",
                kafka_brokers=kafka_brokers,
                err=mlrun.errors.err_to_str(err),
            )
            raise
        else:
            consumer.close()

    def _validate_v3io_stream(
        self,
        v3io_profile: mlrun.datastore.datastore_profile.DatastoreProfileV3io,
    ) -> None:
        if mlrun.mlconf.is_ce_mode():
            raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                "MLRun CE supports only Kafka streams, received a V3IO profile for the stream"
            )
        self._verify_v3io_access(v3io_profile)

    def _verify_v3io_access(
        self, v3io_profile: mlrun.datastore.datastore_profile.DatastoreProfileV3io
    ) -> None:
        stream_access_key = v3io_profile.v3io_access_key
        if not stream_access_key:
            raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                "The model monitoring stream profile must be set with an explicit `v3io_access_key`. "
                f"The passed profile '{v3io_profile.name}' has an empty access key. "
                "You may register it again and set `v3io_access_key=mlrun.mlconf.get_v3io_access_key()`"
            )

        stream_path = mlrun.model_monitoring.get_stream_path(
            project=self.project, profile=v3io_profile
        )
        container, path = split_path(stream_path)

        v3io_client = mlrun.utils.v3io_clients.get_v3io_client(
            endpoint=mlrun.mlconf.v3io_api, access_key=stream_access_key
        )
        # We don't expect the stream to exist. The purpose is to make sure we have access.
        v3io_client.stream.describe(
            container, path, raise_for_status=[HTTPStatus.OK, HTTPStatus.NOT_FOUND]
        )

    def set_credentials(
        self,
        *,
        tsdb_profile_name: typing.Optional[str] = None,
        stream_profile_name: typing.Optional[str] = None,
        replace_creds: bool = False,
    ) -> None:
        """
        Set the model monitoring credentials for the project. The credentials are stored in the project secrets.

        :param tsdb_profile_name:         The TSDB profile name to be used in the project's model monitoring framework.
                                          Either V3IO or TDEngine profile.
        :param stream_profile_name:       The stream profile name to be used in the project's model monitoring
                                          framework. Either V3IO or KafkaSource profile.
        :param replace_creds:             If True, the credentials will be set even if they are already set.
        :raise MLRunConflictError:        If the credentials are already set for the project and the user
                                          provided different creds.
        :raise MLRunInvalidMMStoreTypeError: If the user provided invalid credentials.
        """

        if not replace_creds:
            try:
                self.check_if_credentials_are_set()
                if self._is_the_same_cred(stream_profile_name, tsdb_profile_name):
                    logger.debug(
                        "The same credentials are already set for the project - aborting with no error",
                        project=self.project,
                    )
                    return
                raise mlrun.errors.MLRunConflictError(
                    f"For {self.project} the credentials are already set, if you want to set new credentials, "
                    f"please set replace_creds=True"
                )
            except mlrun.errors.MLRunBadRequestError:
                # the credentials are not set
                pass

        secrets_dict = {}
        old_secrets_dict = self._get_monitoring_mandatory_project_secrets()

        stream_profile_name = stream_profile_name or old_secrets_dict.get(
            mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PROFILE_NAME
        )
        if stream_profile_name:
            self._validate_stream_profile(stream_profile_name)
            secrets_dict[
                mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PROFILE_NAME
            ] = stream_profile_name

        tsdb_profile_name = tsdb_profile_name or old_secrets_dict.get(
            mlrun.common.schemas.model_monitoring.ProjectSecretKeys.TSDB_PROFILE_NAME
        )
        if tsdb_profile_name:
            tsdb_profile = self._validate_and_get_tsdb_profile(tsdb_profile_name)
            secrets_dict[
                mlrun.common.schemas.model_monitoring.ProjectSecretKeys.TSDB_PROFILE_NAME
            ] = tsdb_profile_name

        # Check the cred are valid
        for key in (
            mlrun.common.schemas.model_monitoring.ProjectSecretKeys.mandatory_secrets()
        ):
            if key not in secrets_dict:
                raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                    f"You must provide a valid {key} connection while using set_model_monitoring_credentials."
                )

        # Create TSDB tables that will be used for storing the model monitoring data
        self._create_tsdb_tables(tsdb_profile)

        services.api.crud.Secrets().store_project_secrets(
            project=self.project,
            secrets=mlrun.common.schemas.SecretsData(
                provider=mlrun.common.schemas.SecretProviderName.kubernetes,
                secrets=secrets_dict,
            ),
        )

    def _is_the_same_cred(
        self,
        stream_profile_name: typing.Optional[str],
        tsdb_profile_name: typing.Optional[str],
    ) -> bool:
        credentials_dict = {
            key: mlrun.get_secret_or_env(key, self._secret_provider)
            for key in mlrun.common.schemas.model_monitoring.ProjectSecretKeys.mandatory_secrets()
        }

        old_stream_profile_name = credentials_dict[
            mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PROFILE_NAME
        ]
        if stream_profile_name and old_stream_profile_name != stream_profile_name:
            logger.debug(
                "User provided different stream profile name",
            )
            return False
        old_tsdb_profile_name = credentials_dict[
            mlrun.common.schemas.model_monitoring.ProjectSecretKeys.TSDB_PROFILE_NAME
        ]
        if tsdb_profile_name and old_tsdb_profile_name != tsdb_profile_name:
            logger.debug(
                "User provided different TSDB profile name",
            )
            return False
        return True

    @staticmethod
    async def create_model_endpoints(
        function: dict,
        function_name: str,
        project: str,
    ):
        """
        Create model endpoints for the given function.
        1. Create model endpoint instructions list from the function graph.
        The list is tuple which created from the model endpoint object, creation strategy and model path.
        2. Create the Node/Leaf model endpoints according to the instructions list.
        3. Update the router model endpoint instructions with the children uids.
        4. Create the Router model endpoints according to the instructions list.

        :param function:        The function object.
        :param function_name:   The name of the function.
        :param project:         The project name.
        """
        logger.info(
            "Start Running BGT for model endpoint creation",
            project=project,
            function=function_name,
        )
        try:
            function = mlrun.new_function(
                runtime=function,
                project=project,
                name=function_name,
            )
        except Exception as err:
            logger.error(traceback.format_exc())
            framework.api.utils.log_and_raise(
                HTTPStatus.BAD_REQUEST.value,
                reason=f"Runtime error: {mlrun.errors.err_to_str(err)}",
            )
        model_endpoints_instructions: list[
            tuple[
                mlrun.common.schemas.ModelEndpoint,
                mm_constants.ModelEndpointCreationStrategy,
                str,
            ]
        ] = MonitoringDeployment(
            project=project
        )._extract_model_endpoints_from_function_graph(
            function_name=function.metadata.name,
            function_tag=function.metadata.tag,
            track_models=function.spec.track_models,
            graph=function.spec.graph,
            sampling_percentage=function.spec.parameters.get(
                mm_constants.EventFieldType.SAMPLING_PERCENTAGE, 100
            ),
        )  # model endpoint, creation strategy, model path
        semaphore = Semaphore(50)  # Limit concurrent tasks
        coroutines = []
        batchsize = 500
        for i in range(0, len(model_endpoints_instructions), batchsize):
            batch = model_endpoints_instructions[i : i + batchsize]
            coroutines.append(
                MonitoringDeployment._create_model_endpoint_limited(
                    semaphore, batch, project
                )
            )

        await asyncio.gather(*coroutines)
        logger.info(
            "Finish Running BGT for model endpoint creation",
            project=project,
            function=function_name,
        )

    @staticmethod
    async def _create_model_endpoint_limited(
        semaphore: Semaphore,
        model_endpoints_instructions: list[
            tuple[
                mlrun.common.schemas.ModelEndpoint,
                mm_constants.ModelEndpointCreationStrategy,
                str,
            ]
        ],
        project: str,
    ):
        async with semaphore:
            result = await framework.db.session.run_async_function_with_new_db_session(
                func=services.api.crud.ModelEndpoints().create_model_endpoints,
                model_endpoints_instructions=model_endpoints_instructions,
                project=project,
            )
            return result

    def _extract_model_endpoints_from_function_graph(
        self,
        function_name: str,
        function_tag: str,
        track_models: bool,
        graph: typing.Union[
            mlrun.serving.states.RouterStep, mlrun.serving.states.RootFlowStep
        ],
        sampling_percentage: float,
    ) -> list[
        tuple[
            mlrun.common.schemas.ModelEndpoint,
            mm_constants.ModelEndpointCreationStrategy,
            str,
        ]
    ]:
        model_endpoints_instructions = []
        if isinstance(graph, mlrun.serving.states.RouterStep):
            model_endpoints_instructions.extend(
                self._extract_meps_from_router_step(
                    function_name=function_name,
                    function_tag=function_tag,
                    track_models=track_models,
                    router_step=graph,
                    sampling_percentage=sampling_percentage,
                )
            )
        elif isinstance(graph, mlrun.serving.states.RootFlowStep):
            model_endpoints_instructions.extend(
                self._extract_meps_from_root_flow_step(
                    function_name=function_name,
                    function_tag=function_tag,
                    track_models=track_models,
                    root_flow_step=graph,
                    sampling_percentage=sampling_percentage,
                )
            )
        return model_endpoints_instructions

    def _extract_meps_from_router_step(
        self,
        function_name: str,
        function_tag: str,
        track_models: bool,
        router_step: mlrun.serving.states.RouterStep,
        sampling_percentage: float,
    ) -> list[
        tuple[
            mlrun.common.schemas.ModelEndpoint,
            mm_constants.ModelEndpointCreationStrategy,
            str,
        ]
    ]:
        model_endpoints_instructions = []
        routes_names = []
        routes_uids = []
        for route in router_step.routes.values():
            if (
                route.model_endpoint_creation_strategy
                != mm_constants.ModelEndpointCreationStrategy.SKIP
            ):
                uid = uuid.uuid4().hex
                model_endpoints_instructions.append(
                    (
                        self._model_endpoint_draft(
                            name=route.name,
                            endpoint_type=route.endpoint_type,
                            model_class=route.class_name,
                            function_name=function_name,
                            function_tag=function_tag,
                            track_models=track_models,
                            sampling_percentage=sampling_percentage,
                            uid=uid,
                        ),
                        route.model_endpoint_creation_strategy,
                        route.class_args.get("model_path", ""),
                    )
                )
                routes_names.append(route.name)
                routes_uids.append(uid)
        if (
            router_step.model_endpoint_creation_strategy
            != mm_constants.ModelEndpointCreationStrategy.SKIP
        ):
            model_endpoints_instructions.append(
                (
                    self._model_endpoint_draft(
                        name=router_step.name,
                        endpoint_type=router_step.endpoint_type,
                        model_class=router_step.class_name,
                        function_name=function_name,
                        function_tag=function_tag,
                        track_models=track_models,
                        children_names=routes_names,
                        children_uids=routes_uids,
                        sampling_percentage=sampling_percentage,
                    ),
                    router_step.model_endpoint_creation_strategy,
                    "",
                )
            )

        return model_endpoints_instructions

    def _extract_meps_from_root_flow_step(
        self,
        function_name: str,
        function_tag: str,
        track_models: bool,
        root_flow_step: mlrun.serving.states.RootFlowStep,
        sampling_percentage: float,
    ) -> list[
        tuple[
            mlrun.common.schemas.ModelEndpoint,
            mm_constants.ModelEndpointCreationStrategy,
            str,
        ]
    ]:
        model_endpoints_instructions = []
        for step in root_flow_step.steps.values():
            if isinstance(step, mlrun.serving.states.RouterStep):
                model_endpoints_instructions.extend(
                    self._extract_meps_from_router_step(
                        function_name=function_name,
                        function_tag=function_tag,
                        track_models=track_models,
                        router_step=step,
                        sampling_percentage=sampling_percentage,
                    )
                )
            else:
                if (
                    step.model_endpoint_creation_strategy
                    != mm_constants.ModelEndpointCreationStrategy.SKIP
                ):
                    model_endpoints_instructions.append(
                        (
                            self._model_endpoint_draft(
                                name=step.name,
                                endpoint_type=step.endpoint_type,
                                model_class=step.class_name,
                                function_name=function_name,
                                function_tag=function_tag,
                                track_models=track_models,
                            ),
                            step.model_endpoint_creation_strategy,
                            step.class_args.get("model_path", ""),
                        )
                    )
        return model_endpoints_instructions

    def _model_endpoint_draft(
        self,
        name: str,
        endpoint_type: mm_constants.EndpointType,
        model_class: str,
        function_name: str,
        function_tag: str,
        track_models: bool,
        uid: typing.Optional[str] = None,
        children_names: typing.Optional[list[str]] = None,
        children_uids: typing.Optional[list[str]] = None,
        sampling_percentage: typing.Optional[float] = None,
    ) -> mlrun.common.schemas.ModelEndpoint:
        function_tag = function_tag or "latest"
        return mlrun.common.schemas.ModelEndpoint(
            metadata=mlrun.common.schemas.ModelEndpointMetadata(
                project=self.project, name=name, endpoint_type=endpoint_type, uid=uid
            ),
            spec=mlrun.common.schemas.ModelEndpointSpec(
                function_name=function_name,
                function_tag=function_tag,
                function_uid=f"{unversioned_tagged_object_uid_prefix}{function_tag}",  # TODO: remove after ML-8596
                model_class=model_class,
                children=children_names,
                children_uids=children_uids,
            ),
            status=mlrun.common.schemas.ModelEndpointStatus(
                monitoring_mode=mlrun.common.schemas.model_monitoring.ModelMonitoringMode.enabled
                if track_models
                else mlrun.common.schemas.model_monitoring.ModelMonitoringMode.disabled,
                sampling_percentage=sampling_percentage,
            ),
        )

    @staticmethod
    def _create_model_endpoint_background_task(
        db_session: sqlalchemy.orm.Session,
        background_tasks: BackgroundTasks,
        function_name: str,
        function: dict,
        project_name: str,
    ):
        background_task_name = str(uuid.uuid4())
        return framework.utils.background_tasks.ProjectBackgroundTasksHandler().create_background_task(
            db_session,
            project_name,
            background_tasks,
            MonitoringDeployment.create_model_endpoints,
            mlrun.mlconf.background_tasks.default_timeouts.operations.model_endpoint_creation,
            background_task_name,
            function,
            function_name,
            project_name,
        )

    @staticmethod
    def _get_trigger_frequency(base_period: int) -> int:
        """
        Determines the trigger frequency based on the base period using a lookup dictionary.

        :param base_period: The base period in minutes.
        :return: The trigger frequency in minutes.
        """
        for threshold, frequency in BASE_PERIOD_LOOKUP_TABLE.items():
            if base_period <= threshold:
                return frequency

        return BASE_PERIOD_LOOKUP_TABLE[float("inf")]


def get_endpoint_features(
    feature_names: list[str],
    feature_stats: typing.Optional[dict] = None,
    current_stats: typing.Optional[dict] = None,
) -> list[mlrun.common.schemas.Features]:
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
        f = mlrun.common.schemas.Features.new(
            name, safe_feature_stats.get(name), safe_current_stats.get(name)
        )
        features.append(f)
    return features
