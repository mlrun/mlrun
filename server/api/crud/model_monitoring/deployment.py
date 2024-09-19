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

import json
import os
import time
import typing
import uuid
from http import HTTPStatus
from pathlib import Path

import fastapi
import nuclio
import nuclio.utils
import sqlalchemy.orm
from fastapi import BackgroundTasks
from fastapi.concurrency import run_in_threadpool

import mlrun.common.constants as mlrun_constants
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.model_monitoring.api
import mlrun.model_monitoring.applications
import mlrun.model_monitoring.controller
import mlrun.model_monitoring.stream_processing
import mlrun.model_monitoring.writer
import mlrun.serving.states
import server.api.api.endpoints.nuclio
import server.api.api.utils
import server.api.crud.model_monitoring.helpers
import server.api.db.session
import server.api.utils.background_tasks
import server.api.utils.functions
import server.api.utils.singletons.k8s
from mlrun import feature_store as fstore
from mlrun.config import config
from mlrun.model_monitoring.writer import ModelMonitoringWriter
from mlrun.utils import logger

_STREAM_PROCESSING_FUNCTION_PATH = mlrun.model_monitoring.stream_processing.__file__
_MONITORING_APPLICATION_CONTROLLER_FUNCTION_PATH = (
    mlrun.model_monitoring.controller.__file__
)
_MONITORING_WRITER_FUNCTION_PATH = mlrun.model_monitoring.writer.__file__
_HISTOGRAM_DATA_DRIFT_APP_PATH = str(
    Path(mlrun.model_monitoring.applications.__file__).parent
    / "histogram_data_drift.py"
)


class MonitoringDeployment:
    def __init__(
        self,
        project: str,
        auth_info: mlrun.common.schemas.AuthInfo,
        db_session: sqlalchemy.orm.Session,
        model_monitoring_access_key: typing.Optional[str],
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

    def deploy_monitoring_functions(
        self,
        base_period: int = 10,
        image: str = "mlrun/mlrun",
        deploy_histogram_data_drift_app: bool = True,
        rebuild_images: bool = False,
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
        :param rebuild_images:                    If true, force rebuild of model monitoring infrastructure images
                                                  (controller, writer & stream).
        :param fetch_credentials_from_sys_config: If true, fetch the credentials from the system configuration.
        """
        # check if credentials should be fetched from the system configuration or if they are already been set.
        if fetch_credentials_from_sys_config:
            self.set_credentials()
        self.check_if_credentials_are_set()

        self.deploy_model_monitoring_controller(
            controller_image=image, base_period=base_period, overwrite=rebuild_images
        )
        self.deploy_model_monitoring_writer_application(
            writer_image=image, overwrite=rebuild_images
        )
        self.deploy_model_monitoring_stream_processing(
            stream_image=image, overwrite=rebuild_images
        )
        if deploy_histogram_data_drift_app:
            self.deploy_histogram_data_drift_app(image=image, overwrite=rebuild_images)

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
                server.api.crud.model_monitoring.helpers.get_monitoring_parquet_path(
                    db_session=self.db_session, project=self.project
                )
            )
            fn = self._initial_model_monitoring_stream_processing_function(
                stream_image=stream_image, parquet_target=parquet_target
            )
            fn, ready = server.api.utils.functions.build_function(
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
            fn = self._get_model_monitoring_controller_function(image=controller_image)
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
                spec=nuclio.CronTrigger(interval=f"{base_period}m"),
            )
            fn, ready = server.api.utils.functions.build_function(
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
            fn, ready = server.api.utils.functions.build_function(
                db_session=self.db_session, auth_info=self.auth_info, function=fn
            )
            logger.debug(
                "Submitted the writer deployment",
                writer_data=fn.to_dict(),
                writer_ready=ready,
            )

    def apply_and_create_stream_trigger(
        self, function: mlrun.runtimes.ServingRuntime, function_name: str
    ) -> mlrun.runtimes.ServingRuntime:
        """
        Add stream source for the nuclio serving function. The function's stream trigger can be
        either Kafka or V3IO, depends on the stream path schema that is defined by:

            project.set_model_monitoring_credentials(..., stream_path="...")

        Note: this method also disables the default HTTP trigger of the function, so it remains
        only with stream trigger(s).

        :param function:      The serving function object that will be applied with the stream trigger.
        :param function_name: The name of the function that be applied with the stream trigger.

        :return: `ServingRuntime` object with stream trigger.
        """

        # Get the stream path from the configuration
        stream_path = server.api.crud.model_monitoring.get_stream_path(
            project=self.project, function_name=function_name
        )
        # set all MM app and infra to have only 1 replica
        function.spec.max_replicas = 1
        if stream_path.startswith("kafka://"):
            topic, brokers = mlrun.datastore.utils.parse_kafka_url(url=stream_path)
            # Generate Kafka stream source
            stream_source = mlrun.datastore.sources.KafkaSource(
                brokers=brokers,
                topics=[topic],
            )
            stream_source.create_topics(num_partitions=1, replication_factor=1)
            function = stream_source.add_nuclio_trigger(function)
        elif stream_path.startswith("v3io://"):
            if "projects" in stream_path:
                stream_args = config.model_endpoint_monitoring.application_stream_args
                access_key = self.model_monitoring_access_key
                kwargs = {"access_key": self.model_monitoring_access_key}
            else:
                stream_args = config.model_endpoint_monitoring.serving_stream_args
                access_key = os.getenv("V3IO_ACCESS_KEY")
                kwargs = {}
            if mlrun.mlconf.is_explicit_ack_enabled():
                kwargs["explicit_ack_mode"] = "explicitOnly"
                kwargs["worker_allocation_mode"] = "static"
            server.api.api.endpoints.nuclio.create_model_monitoring_stream(
                project=self.project,
                stream_path=stream_path,
                access_key=access_key,
                stream_args=stream_args,
            )
            # Generate V3IO stream trigger
            function.add_v3io_stream_trigger(
                stream_path=stream_path,
                name=f"monitoring_{function_name}_trigger",
                **kwargs,
            )
        else:
            server.api.api.utils.log_and_raise(
                HTTPStatus.BAD_REQUEST.value,
                reason="Unexpected stream path schema",
            )
        if not mlrun.mlconf.is_ce_mode():
            function = self._apply_access_key_and_mount_function(
                function=function, function_name=function_name
            )

        function.spec.disable_default_http_trigger = True

        return function

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
            server.api.api.utils.get_run_db_instance(self.db_session)
        )

        secret_provider = server.api.crud.secrets.get_project_secret_provider(
            project=self.project
        )

        tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
            project=self.project, secret_provider=secret_provider
        )
        store_object = mlrun.model_monitoring.get_store_object(
            project=self.project, secret_provider=secret_provider
        )

        # Create monitoring serving graph
        stream_processor.apply_monitoring_serving_graph(
            function,
            tsdb_connector,
            store_object,
        )

        # Set the project to the serving function
        function.metadata.project = self.project

        # Add stream triggers
        function = self.apply_and_create_stream_trigger(
            function=function, function_name=mm_constants.MonitoringFunctionNames.STREAM
        )

        # Apply feature store run configurations on the serving function
        run_config = fstore.RunConfig(function=function, local=False)
        function.spec.parameters = run_config.parameters

        return function

    def _get_model_monitoring_controller_function(self, image: str):
        """
        Initialize model monitoring controller function.

        :param image:         Base docker image to use for building the function container
        :return:              A function object from a mlrun runtime class
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
            server.api.api.utils.get_run_db_instance(self.db_session)
        )

        # Set the project to the job function
        function.metadata.project = self.project

        function = self._apply_access_key_and_mount_function(
            function=function,
            function_name=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
        )
        function.spec.max_replicas = 1
        # Enrich runtime with the required configurations
        server.api.api.utils.apply_enrichment_and_validation_on_function(
            function, self.auth_info
        )

        return function

    def _apply_access_key_and_mount_function(
        self,
        function: typing.Union[
            mlrun.runtimes.KubejobRuntime, mlrun.runtimes.ServingRuntime
        ],
        function_name: str = None,
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
                server.api.utils.singletons.k8s.get_k8s_helper().get_project_secret_name(
                    self.project
                ),
                server.api.crud.secrets.Secrets().generate_client_project_secret_key(
                    server.api.crud.secrets.SecretsClientType.model_monitoring,
                    mm_constants.ProjectSecretKeys.ACCESS_KEY,
                ),
            )

            function.metadata.credentials.access_key = self.model_monitoring_access_key
            function.apply(mlrun.v3io_cred())

            # Ensure that the auth env vars are set
            server.api.api.utils.ensure_function_has_auth_set(function, self.auth_info)
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
            server.api.api.utils.get_run_db_instance(self.db_session)
        )

        # Create writer monitoring serving graph
        graph = function.set_topology(mlrun.serving.states.StepKinds.flow)
        graph.to(
            ModelMonitoringWriter(
                project=self.project,
                secret_provider=server.api.crud.secrets.get_project_secret_provider(
                    project=self.project
                ),
            )
        )  # writer

        # Set the project to the serving function
        function.metadata.project = self.project

        # Add stream triggers
        function = self.apply_and_create_stream_trigger(
            function=function, function_name=mm_constants.MonitoringFunctionNames.WRITER
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
                server.api.api.utils.ensure_function_has_auth_set(func, self.auth_info)
                logger.info("Ensured the histogram data drift function auth")

            func.set_label(
                mm_constants.ModelMonitoringAppLabel.KEY,
                mm_constants.ModelMonitoringAppLabel.VAL,
            )

            fn, ready = server.api.utils.functions.build_function(
                db_session=self.db_session, auth_info=self.auth_info, function=func
            )

            logger.debug(
                "Submitted the histogram data drift app deployment",
                app_data=fn.to_dict(),
                app_ready=ready,
            )

    def _create_tsdb_tables(self, connection_string: str):
        """Create the TSDB tables using the TSDB connector. At the moment we support 3 types of tables:
        - app_results: a detailed result that includes status, kind, extra data, etc.
        - metrics: a basic key value that represents a numeric metric.
        - predictions: latency of each prediction."""

        tsdb_connector: mlrun.model_monitoring.db.TSDBConnector = (
            mlrun.model_monitoring.get_tsdb_connector(
                project=self.project,
                tsdb_connection_string=connection_string,
            )
        )

        tsdb_connector.create_tables()

    def _create_sql_tables(self, connection_string: str):
        """Create the SQL tables using the SQL connector"""

        store_connector: mlrun.model_monitoring.db.StoreBase = (
            mlrun.model_monitoring.get_store_object(
                project=self.project,
                store_connection_string=connection_string,
            )
        )

        store_connector.create_tables()

    def list_model_monitoring_functions(self) -> list:
        """Retrieve a list of all the model monitoring functions."""
        model_monitoring_labels_list = [
            f"{mm_constants.ModelMonitoringAppLabel.KEY}={mm_constants.ModelMonitoringAppLabel.VAL}"
        ]
        return server.api.crud.Functions().list_functions(
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
        user_application_list: list[str] = None,
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
                    server.api.db.session.run_function_with_new_db_session,
                    MonitoringDeployment._create_monitoring_function_deletion_background_task,
                    background_tasks=background_tasks,
                    project_name=self.project,
                    function_name=function_name,
                    auth_info=self.auth_info,
                    delete_app_stream_resources=function_name
                    not in [
                        mm_constants.MonitoringFunctionNames.STREAM,
                        mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
                    ],
                    access_key=self.model_monitoring_access_key,
                )
                tasks.append(task)

        return mlrun.common.schemas.BackgroundTaskList(background_tasks=tasks)

    def _get_monitoring_application_to_delete(
        self,
        delete_histogram_data_drift_app: bool = True,
        delete_user_applications: bool = False,
        user_application_list: list[str] = None,
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
                        fn = server.api.crud.Functions().get_function(
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
        access_key: str,
    ):
        background_task_name = str(uuid.uuid4())

        # create the background task for function deletion
        return server.api.utils.background_tasks.ProjectBackgroundTasksHandler().create_background_task(
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
            access_key,
        )

    @staticmethod
    async def delete_monitoring_function(
        db_session: sqlalchemy.orm.Session,
        project: str,
        function_name: str,
        auth_info: mlrun.common.schemas.AuthInfo,
        background_task_name: str,
        delete_app_stream_resources: bool,
        access_key: str,
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
        await server.api.api.utils._delete_function(
            db_session=db_session,
            project=project,
            function_name=function_name,
            auth_info=auth_info,
            background_task_name=background_task_name,
        )
        if delete_app_stream_resources:
            try:
                MonitoringDeployment._delete_model_monitoring_stream_resources(
                    project=project,
                    function_names=[function_name],
                    access_key=access_key,
                )
            except mlrun.errors.MLRunStreamConnectionFailureError as e:
                logger.warning(
                    "Failed to delete stream resources, you may need to delete them manually",
                    project_name=project,
                    function=function_name,
                    error=mlrun.errors.err_to_str(e),
                )

    @staticmethod
    def _delete_model_monitoring_stream_resources(
        project: str,
        function_names: list[str],
        access_key: typing.Optional[str] = None,
    ) -> None:
        """
        :param project:        The name of the project.
        :param function_names: A list of functions that their resources should be deleted.
        :param access_key:     If the stream is V3IO, the access key is required.

        """
        logger.debug(
            "Deleting model monitoring stream resources deployment",
            project_name=project,
        )
        stream_paths = []
        for function_name in function_names:
            for i in range(10):
                # waiting for the function pod to be deleted
                # max 10 retries (5 sec sleep between each retry)

                function_pod = server.api.utils.singletons.k8s.get_k8s_helper().list_pods(
                    selector=f"{mlrun_constants.MLRunInternalLabels.nuclio_function_name}={project}-{function_name}"
                )
                if not function_pod:
                    logger.debug(
                        "No function pod found for project, deleting stream",
                        project_name=project,
                        function=function_name,
                    )
                    break
                else:
                    logger.debug(f"{function_name} pod found, retrying")
                    time.sleep(5)

            stream_paths.append(
                server.api.crud.model_monitoring.get_stream_path(
                    project=project, function_name=function_name
                )
            )

        if not stream_paths:
            # No stream paths to delete
            return

        elif stream_paths[0].startswith("v3io"):
            # Delete V3IO stream
            import v3io.dataplane
            import v3io.dataplane.response

            v3io_client = v3io.dataplane.Client(endpoint=mlrun.mlconf.v3io_api)

            for stream_path in stream_paths:
                _, container, stream_path = (
                    mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
                        stream_path
                    )
                )

                try:
                    # if the stream path is in the users directory, we need to use pipelines access key to delete it
                    logger.debug(
                        "Deleting v3io stream", project=project, stream_path=stream_path
                    )
                    v3io_client.stream.delete(
                        container,
                        stream_path,
                        access_key=mlrun.mlconf.get_v3io_access_key()
                        if container.startswith("users")
                        else access_key,
                    )
                    logger.debug(
                        "Deleted v3io stream", project=project, stream_path=stream_path
                    )
                except Exception as exc:
                    # Raise an error that will be caught by the caller and skip the deletion of the stream
                    raise mlrun.errors.MLRunStreamConnectionFailureError(
                        f"Failed to delete v3io stream {stream_path}"
                    ) from exc
        elif stream_paths[0].startswith("kafka://"):
            # Delete Kafka topics
            import kafka
            import kafka.errors

            topics = []

            topic, brokers = mlrun.datastore.utils.parse_kafka_url(url=stream_paths[0])
            topics.append(topic)

            for stream_path in stream_paths[1:]:
                topic, _ = mlrun.datastore.utils.parse_kafka_url(url=stream_path)
                topics.append(topic)

            try:
                kafka_client = kafka.KafkaAdminClient(
                    bootstrap_servers=brokers,
                    client_id=project,
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
                "Stream path is not supported and therefore can't be deleted, expected v3io or kafka",
                stream_path=stream_paths[0],
            )
        logger.debug(
            "Successfully deleted model monitoring stream resources deployment",
            project_name=project,
        )

    def _get_monitoring_mandatory_project_secrets(self) -> dict[str, str]:
        credentials_dict = {
            key: server.api.crud.Secrets().get_project_secret(
                project=self.project,
                provider=mlrun.common.schemas.SecretProviderName.kubernetes,
                secret_key=key,
                allow_secrets_from_k8s=True,
            )
            for key in mlrun.common.schemas.model_monitoring.ProjectSecretKeys.mandatory_secrets()
        }

        return credentials_dict

    def check_if_credentials_are_set(
        self,
    ):
        """
        Check if the model monitoring credentials are set. If not, raise an error.

        :raise mlrun.errors.MLRunBadRequestError:  if the credentials are not set.
        """

        credentials_dict = self._get_monitoring_mandatory_project_secrets()
        if all([val is not None for key, val in credentials_dict.items()]):
            return

        raise mlrun.errors.MLRunBadRequestError(
            "Model monitoring credentials are not set. "
            "Please set them using the set_model_monitoring_credentials API/SDK "
            "or pass fetch_credentials_from_sys_config=True when using enable_model_monitoring API/SDK."
        )

    def set_credentials(
        self,
        access_key: typing.Optional[str] = None,
        endpoint_store_connection: typing.Optional[str] = None,
        stream_path: typing.Optional[str] = None,
        tsdb_connection: typing.Optional[str] = None,
        replace_creds: bool = False,
        _default_secrets_v3io: typing.Optional[str] = None,
    ) -> None:
        """
        Set the model monitoring credentials for the project. The credentials are stored in the project secrets.

        :param access_key:                Model Monitoring access key for managing user permissions.
        :param endpoint_store_connection: Endpoint store connection string. By default, None.
                                          Options:
                                          1. None, will be set from the system configuration.
                                          2. v3io - for v3io endpoint store,
                                             pass `v3io` and the system will generate the exact path.
                                          3. MySQL/SQLite - for SQL endpoint store, please provide full
                                             connection string, for example
                                             mysql+pymysql://<username>:<password>@<host>:<port>/<db_name>
        :param stream_path:               Path to the model monitoring stream. By default, None.
                                          Options:
                                          1. None, will be set from the system configuration.
                                          2. v3io - for v3io stream,
                                             pass `v3io` and the system will generate the exact path.
                                          3. Kafka - for Kafka stream, please provide full connection string without
                                             custom topic, for example kafka://<some_kafka_broker>:<port>.
        :param tsdb_connection:           Connection string to the time series database. By default, None.
                                          Options:
                                          1. None, will be set from the system configuration.
                                          2. v3io - for v3io stream,
                                             pass `v3io` and the system will generate the exact path.
                                          3. TDEngine - for TDEngine tsdb, please provide full websocket connection URL,
                                             for example taosws://<username>:<password>@<host>:<port>.
        :param replace_creds:             If True, the credentials will be set even if they are already set.
        :param _default_secrets_v3io:     Optional parameter for the upgrade process in which the v3io default secret
                                          key is set.
        :raise MLRunConflictError:        If the credentials are already set for the project and the user
                                          provided different creds.
        :raise MLRunInvalidMMStoreTypeError: If the user provided invalid credentials.
        """
        if not replace_creds:
            try:
                self.check_if_credentials_are_set()
                if self._is_the_same_cred(
                    endpoint_store_connection, stream_path, tsdb_connection
                ):
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
        if access_key:
            secrets_dict[
                mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ACCESS_KEY
            ] = access_key or old_secrets_dict.get(
                mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ACCESS_KEY
            )

        # endpoint_store_connection
        if not endpoint_store_connection:
            endpoint_store_connection = (
                old_secrets_dict.get(
                    mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ENDPOINT_STORE_CONNECTION
                )
                or mlrun.mlconf.model_endpoint_monitoring.endpoint_store_connection
                or _default_secrets_v3io
            )
        if endpoint_store_connection:
            if not endpoint_store_connection.startswith(
                tuple(
                    mlrun.common.schemas.model_monitoring.ModelEndpointTargetSchemas.list()
                )
            ):
                raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                    "Currently only MySQL/SQLite connections are supported for non-v3io endpoint store,"
                    "please provide a full URL (e.g. mysql+pymysql://<username>:<password>@<host>:<port>/<db_name>)"
                )
            if mlrun.mlconf.is_ce_mode() and endpoint_store_connection.startswith(
                "v3io"
            ):
                raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                    "In CE mode, only MySQL/SQLite connections are supported for endpoint store"
                )
            secrets_dict[
                mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ENDPOINT_STORE_CONNECTION
            ] = endpoint_store_connection
        else:
            raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                "You must provide a valid endpoint store connection while using set_model_monitoring_credentials "
                "API/SDK or in the system config"
            )

        # stream_path
        if not stream_path:
            stream_path = (
                old_secrets_dict.get(
                    mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PATH
                )
                or mlrun.mlconf.model_endpoint_monitoring.stream_connection
                or _default_secrets_v3io
            )
        if stream_path:
            if (
                stream_path == mm_constants.V3IO_MODEL_MONITORING_DB
                and mlrun.mlconf.is_ce_mode()
            ):
                raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                    "In CE mode, only kafka stream are supported for stream path"
                )
            elif stream_path.startswith("kafka://") and "?topic" in stream_path:
                raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                    "Custom kafka topic is not allowed"
                )
            elif not stream_path.startswith("kafka://") and (
                stream_path != mm_constants.V3IO_MODEL_MONITORING_DB
            ):
                raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                    "Currently only Kafka connection is supported for non-v3io stream,"
                    "please provide a full URL (e.g. kafka://<some_kafka_broker>:<port>)"
                )
            secrets_dict[
                mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PATH
            ] = stream_path
        else:
            raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                "You must provide a valid stream path connection while using set_model_monitoring_credentials "
                "API/SDK or in the system config"
            )

        if not tsdb_connection:
            tsdb_connection = (
                old_secrets_dict.get(
                    mlrun.common.schemas.model_monitoring.ProjectSecretKeys.TSDB_CONNECTION
                )
                or mlrun.mlconf.model_endpoint_monitoring.tsdb_connection
                or _default_secrets_v3io
            )
        if tsdb_connection:
            if (
                tsdb_connection != mm_constants.V3IO_MODEL_MONITORING_DB
                and not tsdb_connection.startswith("taosws://")
            ):
                raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                    "Currently only TDEngine websocket connection is supported for non-v3io TSDB,"
                    "please provide a full URL (e.g. taosws://<username>:<password>@<host>:<port>)"
                )
            elif (
                tsdb_connection == mm_constants.V3IO_MODEL_MONITORING_DB
                and mlrun.mlconf.is_ce_mode()
            ):
                raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                    "In CE mode, only TDEngine websocket connection is supported for TSDB"
                )
            secrets_dict[
                mlrun.common.schemas.model_monitoring.ProjectSecretKeys.TSDB_CONNECTION
            ] = tsdb_connection
        else:
            raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                "You must provide a valid tsdb connection while using set_model_monitoring_credentials "
                "API/SDK or in the system config"
            )

        # Check the cred are valid
        for key in (
            mlrun.common.schemas.model_monitoring.ProjectSecretKeys.mandatory_secrets()
        ):
            try:
                secrets_dict[key]
            except KeyError:
                raise mlrun.errors.MLRunInvalidMMStoreTypeError(
                    f"You must provide a valid {key} connection while using set_model_monitoring_credentials."
                )
        # Create tsdb & sql tables that will be used for storing the model monitoring data
        # Create the stream output
        self._create_tsdb_tables(
            connection_string=secrets_dict.get(
                mlrun.common.schemas.model_monitoring.ProjectSecretKeys.TSDB_CONNECTION
            )
        )
        self._create_sql_tables(
            connection_string=secrets_dict.get(
                mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ENDPOINT_STORE_CONNECTION
            )
        )

        self._create_stream_output(
            stream_path=secrets_dict.get(
                mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PATH
            )
        )

        server.api.crud.Secrets().store_project_secrets(
            project=self.project,
            secrets=mlrun.common.schemas.SecretsData(
                provider=mlrun.common.schemas.SecretProviderName.kubernetes,
                secrets=secrets_dict,
            ),
        )

    def _is_the_same_cred(
        self, endpoint_store_connection: str, stream_path: str, tsdb_connection: str
    ) -> bool:
        credentials_dict = {
            key: server.api.crud.Secrets().get_project_secret(
                project=self.project,
                provider=mlrun.common.schemas.SecretProviderName.kubernetes,
                secret_key=key,
                allow_secrets_from_k8s=True,
            )
            for key in mlrun.common.schemas.model_monitoring.ProjectSecretKeys.mandatory_secrets()
        }

        old_store = credentials_dict[
            mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ENDPOINT_STORE_CONNECTION
        ]
        if endpoint_store_connection and old_store != endpoint_store_connection:
            logger.debug(
                "User provided different endpoint store connection",
            )
            return False
        old_stream = credentials_dict[
            mlrun.common.schemas.model_monitoring.ProjectSecretKeys.STREAM_PATH
        ]
        if stream_path and old_stream != stream_path:
            logger.debug(
                "User provided different stream path",
            )
            return False
        old_tsdb = credentials_dict[
            mlrun.common.schemas.model_monitoring.ProjectSecretKeys.TSDB_CONNECTION
        ]
        if tsdb_connection and old_tsdb != tsdb_connection:
            logger.debug(
                "User provided different tsdb connection",
            )
            return False
        return True

    def _create_stream_output(self, stream_path: str = None, access_key: str = None):
        stream_path = server.api.crud.model_monitoring.get_stream_path(
            project=self.project, stream_uri=stream_path
        )
        if not mlrun.mlconf.is_ce_mode():
            access_key = access_key or self.model_monitoring_access_key
        else:
            access_key = None

        output_stream = mlrun.datastore.get_stream_pusher(
            stream_path=stream_path,
            endpoint=mlrun.mlconf.v3io_api,
            access_key=access_key,
        )
        if hasattr(output_stream, "_lazy_init"):
            output_stream._lazy_init()


def get_endpoint_features(
    feature_names: list[str],
    feature_stats: dict = None,
    current_stats: dict = None,
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
