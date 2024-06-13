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
import mlrun.model_monitoring.controller_handler
import mlrun.model_monitoring.stream_processing
import mlrun.model_monitoring.writer
import mlrun.serving.states
import mlrun.utils.singleton
import server.api.api.endpoints.nuclio
import server.api.api.utils
import server.api.crud.model_monitoring.helpers
import server.api.utils.background_tasks
import server.api.utils.functions
import server.api.utils.singletons.k8s
from mlrun import feature_store as fstore
from mlrun.config import config
from mlrun.model_monitoring.writer import ModelMonitoringWriter
from mlrun.utils import logger
from server.api.utils.runtimes.nuclio import resolve_nuclio_version

_STREAM_PROCESSING_FUNCTION_PATH = mlrun.model_monitoring.stream_processing.__file__
_MONITORING_APPLICATION_CONTROLLER_FUNCTION_PATH = (
    mlrun.model_monitoring.controller_handler.__file__
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
         1. model monitoring stream
         2. model monitoring controller
         3. model monitoring writer

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
    ) -> None:
        """
        Deploy model monitoring application controller, writer and stream functions.

        :param base_period: The time period in minutes in which the model monitoring controller function
                            triggers. By default, the base period is 10 minutes.
        :param image:       The image of the model monitoring controller, writer & monitoring
                            stream functions, which are real time nuclio functino.
                            By default, the image is mlrun/mlrun.
        :param deploy_histogram_data_drift_app: If true, deploy the default histogram-based data drift application.
        :param rebuild_images:   If true, force rebuild of model monitoring infrastructure images
                                 (controller, writer & stream).
        """
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
        # Create tsdb tables that will be used for storing the model monitoring data
        self._create_tsdb_tables()

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

        if not self._check_if_already_deployed(
            function_name=mm_constants.MonitoringFunctionNames.STREAM,
            overwrite=overwrite,
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

            if (
                overwrite and not self.is_monitoring_stream_has_the_new_stream_trigger()
            ):  # in case of only adding the new stream trigger
                prev_stream_function = server.api.crud.Functions().get_function(
                    name=mm_constants.MonitoringFunctionNames.STREAM,
                    db_session=self.db_session,
                    project=self.project,
                )
                stream_image = prev_stream_function["spec"]["image"]

            fn = self._initial_model_monitoring_stream_processing_function(
                stream_image=stream_image, parquet_target=parquet_target
            )

            # Adding label to the function - will be used to identify the stream pod
            fn.metadata.labels = {"type": mm_constants.MonitoringFunctionNames.STREAM}

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

        :param base_period:                 The time period in minutes in which the model monitoring controller function
                                            triggers. By default, the base period is 10 minutes.
        :param controller_image:            The image of the model monitoring controller function.
                                            By default, the image is mlrun/mlrun.
        :param overwrite:                   If true, overwrite the existing model monitoring controller.
                                            By default, False.
        """
        if not self._check_if_already_deployed(
            function_name=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            overwrite=overwrite,
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

        if not self._check_if_already_deployed(
            function_name=mm_constants.MonitoringFunctionNames.WRITER,
            overwrite=overwrite,
        ):
            logger.info(
                f"Deploying {mm_constants.MonitoringFunctionNames.WRITER} function",
                project=self.project,
            )
            fn = self._initial_model_monitoring_writer_function(
                writer_image=writer_image
            )

            # Adding label to the function - will be used to identify the writer pod
            fn.metadata.labels = {"type": mm_constants.MonitoringFunctionNames.WRITER}

            fn, ready = server.api.utils.functions.build_function(
                db_session=self.db_session, auth_info=self.auth_info, function=fn
            )

            logger.debug(
                "Submitted the writer deployment",
                writer_data=fn.to_dict(),
                writer_ready=ready,
            )

    def apply_and_create_stream_trigger(
        self, function: mlrun.runtimes.ServingRuntime, function_name: str = None
    ) -> mlrun.runtimes.ServingRuntime:
        """Adding stream source for the nuclio serving function. By default, the function has HTTP stream trigger along
        with another supported stream source that can be either Kafka or V3IO, depends on the stream path schema that is
        defined under mlrun.mlconf.model_endpoint_monitoring.store_prefixes. Note that if no valid stream path has been
        provided then the function will have a single HTTP stream source.

        :param function:                    The serving function object that will be applied with the stream trigger.
        :param function_name:               The name of the function that be applied with the stream trigger,
                                            None for model_monitoring_stream

        :return: ServingRuntime object with stream trigger.
        """

        # Get the stream path from the configuration
        stream_paths = server.api.crud.model_monitoring.get_stream_path(
            project=self.project, function_name=function_name
        )
        for i, stream_path in enumerate(stream_paths):
            if stream_path.startswith("kafka://"):
                topic, brokers = mlrun.datastore.utils.parse_kafka_url(url=stream_path)
                # Generate Kafka stream source
                stream_source = mlrun.datastore.sources.KafkaSource(
                    brokers=brokers,
                    topics=[topic],
                )
                function = stream_source.add_nuclio_trigger(function)

            if not mlrun.mlconf.is_ce_mode():
                if stream_path.startswith("v3io://"):
                    if "projects" in stream_path:
                        stream_args = (
                            config.model_endpoint_monitoring.application_stream_args
                        )
                        access_key = self.model_monitoring_access_key
                        kwargs = {"access_key": self.model_monitoring_access_key}
                    else:
                        stream_args = (
                            config.model_endpoint_monitoring.serving_stream_args
                        )
                        access_key = os.getenv("V3IO_ACCESS_KEY")
                        kwargs = {}
                    if mlrun.mlconf.is_explicit_ack(version=resolve_nuclio_version()):
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
                        name=f"monitoring_{function_name}_trigger{f'_{i}' if i != 0 else ''}",
                        **kwargs,
                    )
                function = self._apply_access_key_and_mount_function(
                    function=function, function_name=function_name
                )
        # Add the default HTTP source
        http_source = mlrun.datastore.sources.HttpSource()
        function = http_source.add_nuclio_trigger(function)

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
            ),
        )
        function.set_db_connection(
            server.api.api.utils.get_run_db_instance(self.db_session)
        )

        # Create monitoring serving graph
        stream_processor.apply_monitoring_serving_graph(
            function,
            secret_provider=server.api.crud.secrets.get_project_secret_provider(
                project=self.project
            ),
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
        function = mlrun.code_to_function(
            name=mm_constants.MonitoringFunctionNames.WRITER,
            project=self.project,
            filename=_MONITORING_WRITER_FUNCTION_PATH,
            kind=mlrun.run.RuntimeKinds.serving,
            image=writer_image,
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
        ).respond()  # writer

        # Set the project to the serving function
        function.metadata.project = self.project

        # Add stream triggers
        function = self.apply_and_create_stream_trigger(
            function=function,
            function_name=mm_constants.MonitoringFunctionNames.WRITER,
        )

        # Apply feature store run configurations on the serving function
        run_config = fstore.RunConfig(function=function, local=False)
        function.spec.parameters = run_config.parameters

        return function

    def _check_if_already_deployed(
        self, function_name: str, overwrite: bool = False
    ) -> bool:
        """
         If overwrite equal False the method check the desired function is all ready deployed

        :param function_name:   The name of the function to check.
        :param overwrite:       If true, overwrite the existing model monitoring controller.
                                By default, False.

        :return:                True if the function is already deployed, otherwise False.
        """
        if not overwrite:
            logger.info(
                f"Checking if {function_name} is already deployed",
                project=self.project,
            )
            try:
                # validate that the function has not yet been deployed
                mlrun.runtimes.nuclio.function.get_nuclio_deploy_status(
                    name=function_name,
                    project=self.project,
                    tag="",
                    auth_info=self.auth_info,
                )
                logger.info(
                    f"Detected {function_name} function already deployed",
                    project=self.project,
                )
                return True
            except mlrun.errors.MLRunNotFoundError:
                pass
        return False

    def deploy_histogram_data_drift_app(
        self, image: str, overwrite: bool = False
    ) -> None:
        """
        Deploy the histogram data drift application.

        :param image:       The image on with the function will run.
        :param overwrite:   If True, the function will be overwritten.
        """
        if not self._check_if_already_deployed(
            function_name=mm_constants.HistogramDataDriftApplicationConstants.NAME,
            overwrite=overwrite,
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

    def is_monitoring_stream_has_the_new_stream_trigger(self) -> bool:
        """
        Check if the monitoring stream function has the new stream trigger.

        :return: True if the monitoring stream function has the new stream trigger, otherwise False.
        """

        try:
            function = server.api.crud.Functions().get_function(
                name=mm_constants.MonitoringFunctionNames.STREAM,
                db_session=self.db_session,
                project=self.project,
            )
        except mlrun.errors.MLRunNotFoundError:
            logger.info(
                "The stream function is not deployed yet when the user will run `enable_model_monitoring` "
                "the stream function will be deployed with the new & the old stream triggers",
                project=self.project,
            )
            return True

        if (
            function["spec"]["config"].get(
                f"spec.triggers.monitoring_{mm_constants.MonitoringFunctionNames.STREAM}_trigger_1"
            )
            is None
        ):
            logger.info(
                "The stream function needs to be updated with the new stream trigger",
                project=self.project,
            )
            return False
        return True

    def _create_tsdb_tables(self):
        """Create the TSDB tables using the TSDB connector. At the moment we support 3 types of tables:
        - app_results: a detailed result that includes status, kind, extra data, etc.
        - metrics: a basic key value that represents a numeric metric.
        - predictions: latency of each prediction."""

        tsdb_connector: mlrun.model_monitoring.db.TSDBConnector = (
            mlrun.model_monitoring.get_tsdb_connector(
                project=self.project,
                secret_provider=server.api.crud.secrets.get_project_secret_provider(
                    project=self.project
                ),
            )
        )

        tsdb_connector.create_tables()

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

        :param delete_resources:                    If True, it would delete the model monitoring controller & writer
                                                    functions. Default True
        :param delete_stream_function:              If True, it would delete model monitoring stream function,
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
            if self._check_if_already_deployed(function_name):
                task = await run_in_threadpool(
                    MonitoringDeployment._create_monitoring_function_deletion_background_task,
                    background_tasks,
                    self.db_session,
                    self.project,
                    function_name,
                    self.auth_info,
                    delete_v3io_stream=not mlrun.mlconf.is_ce_mode()
                    and function_name
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
        background_tasks: BackgroundTasks,
        db_session: sqlalchemy.orm.Session,
        project_name: str,
        function_name: str,
        auth_info: mlrun.common.schemas.AuthInfo,
        delete_v3io_stream: bool,
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
            delete_v3io_stream,
            access_key,
        )

    @staticmethod
    async def delete_monitoring_function(
        db_session: sqlalchemy.orm.Session,
        project: str,
        function_name: str,
        auth_info: mlrun.common.schemas.AuthInfo,
        background_task_name: str,
        delete_v3io_stream: bool,
        access_key: str,
    ):
        """
        Delete the model monitoring function and its resources.

        :param db_session:              A session that manages the current dialog with the database.
        :param project:                 The name of the project.
        :param function_name:           The name of the function to delete.
        :param auth_info:               The auth info of the request.
        :param background_task_name:    The name of the background task.
        :param delete_v3io_stream:      If True, delete the V3IO stream.
        :param access_key:              Model monitoring access key.
        """
        await server.api.api.utils._delete_function(
            db_session=db_session,
            project=project,
            function_name=function_name,
            auth_info=auth_info,
            background_task_name=background_task_name,
        )
        if delete_v3io_stream:
            import v3io.dataplane
            import v3io.dataplane.response

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

            v3io_client = v3io.dataplane.Client(endpoint=mlrun.mlconf.v3io_api)
            stream_paths = server.api.crud.model_monitoring.get_stream_path(
                project=project, function_name=function_name
            )
            for stream_path in stream_paths:
                _, container, stream_path = (
                    mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
                        stream_path
                    )
                )

                try:
                    v3io_client.stream.delete(
                        container, stream_path, access_key=access_key
                    )
                except v3io.dataplane.response.HttpResponseError as e:
                    logger.warning(
                        f"Can't delete {function_name}'s stream",
                        stream_path=stream_path,
                        error=e,
                    )


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
        if feature_stats is not None and name not in feature_stats:
            logger.warn("Feature missing from 'feature_stats'", name=name)
        if current_stats is not None and name not in current_stats:
            logger.warn("Feature missing from 'current_stats'", name=name)
        f = mlrun.common.schemas.Features.new(
            name, safe_feature_stats.get(name), safe_current_stats.get(name)
        )
        features.append(f)
    return features
