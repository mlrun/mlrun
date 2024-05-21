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
import typing
from pathlib import Path

import fastapi
import nuclio
import sqlalchemy.orm

import mlrun.common.schemas
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.model_monitoring.api
import mlrun.model_monitoring.application
import mlrun.model_monitoring.applications
import mlrun.model_monitoring.controller_handler
import mlrun.model_monitoring.stream_processing
import mlrun.model_monitoring.writer
import mlrun.serving.states
import server.api.api.endpoints.nuclio
import server.api.api.utils
import server.api.crud.model_monitoring.helpers
import server.api.utils.background_tasks
import server.api.utils.functions
import server.api.utils.scheduler
import server.api.utils.singletons.db
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

        # the image manifest acts as a cache for deployed model monitoring function images.
        # this allows us to reuse these images for future deployments across projects, improving efficiency.
        # the manifest stores a unique image for each combination of three factors:
        # mlrun version, nuclio version, and base image.
        self._image_manifest = {}
        self._read_image_manifest()

    def deploy_monitoring_functions(
        self,
        background_tasks: fastapi.BackgroundTasks,
        db_session: sqlalchemy.orm.Session,
        base_period: int = 10,
        image: str = "mlrun/mlrun",
        deploy_histogram_data_drift_app: bool = True,
        force_build: bool = False,
    ) -> None:
        """
        Deploy model monitoring application controller, writer and stream functions.

        :param background_tasks: Background task manager.
        :param db_session:       A session that manages the current dialog with the database.
        :param base_period:      The time period in minutes in which the model monitoring controller function
                                 triggers. By default, the base period is 10 minutes.
        :param image:            The image of the model monitoring controller, writer & monitoring
                                 stream functions, which are real time nuclio functino.
                                 By default, the image is mlrun/mlrun.
        :param deploy_histogram_data_drift_app: If true, deploy the default histogram-based data drift application.
        :param force_build:      If true, force the build of the functions images. Default is False.
        """
        extra_functions = []
        self.deploy_model_monitoring_controller(
            controller_image=image, base_period=base_period, force_build=force_build
        )
        self.deploy_model_monitoring_writer_application(
            writer_image=image, force_build=force_build
        )
        self.deploy_model_monitoring_stream_processing(
            stream_image=image, force_build=force_build
        )
        if deploy_histogram_data_drift_app:
            self.deploy_histogram_data_drift_app(image=image, force_build=force_build)
            extra_functions.append(
                mm_constants.HistogramDataDriftApplicationConstants.NAME
            )

        # Update the image manifest cache in a background task, so the invoking client will not wait for it
        server.api.utils.background_tasks.ProjectBackgroundTasksHandler().create_background_task(
            db_session,
            self.project,
            background_tasks,
            self._update_image_manifest,
            config.background_tasks.default_timeouts.operations.update_model_monitoring_manifest,
            None,
            # args for _update_image_manifest
            image,
            extra_functions,
        )

    def deploy_model_monitoring_stream_processing(
        self,
        stream_image: str = "mlrun/mlrun",
        overwrite: bool = False,
        force_build: bool = False,
    ) -> None:
        """
        Deploying model monitoring stream real time nuclio function. The goal of this real time function is
        to monitor the log of the data stream. It is triggered when a new log entry is detected.
        It processes the new events into statistics that are then written to statistics databases.

        :param stream_image:                The image of the model monitoring stream function.
                                            By default, the image is mlrun/mlrun.
        :param overwrite:                   If true, overwrite the existing model monitoring stream. Default is False.
        :param force_build:                 If true, force the build of the function image. Default is False.
        """

        if not self._check_if_already_deployed(
            function_name=mm_constants.MonitoringFunctionNames.STREAM,
            overwrite=overwrite,
        ):
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
                stream_image=stream_image,
                parquet_target=parquet_target,
                force_build=force_build,
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
        force_build: bool = False,
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
        :param force_build:                 If true, force the build of the function image. Default is False.
        """
        if not self._check_if_already_deployed(
            function_name=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            overwrite=overwrite,
        ):
            fn = self._get_model_monitoring_controller_function(
                image=controller_image, force_build=force_build
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
        self,
        writer_image: str = "mlrun/mlrun",
        overwrite: bool = False,
        force_build: bool = False,
    ) -> None:
        """
        Deploying model monitoring writer real time nuclio function. The goal of this real time function is
        to write all the monitoring application result to the databases. It is triggered by those applications.
        It processes and writes the result to the databases.

        :param writer_image:                The image of the model monitoring writer function.
                                            By default, the image is mlrun/mlrun.
        :param overwrite:                   If true, overwrite the existing model monitoring writer. Default is False.
        :param force_build:                 If true, force the build of the function image. Default is False.
        """

        if not self._check_if_already_deployed(
            function_name=mm_constants.MonitoringFunctionNames.WRITER,
            overwrite=overwrite,
        ):
            fn = self._initial_model_monitoring_writer_function(
                writer_image=writer_image, force_build=force_build
            )

            # Adding label to the function - will be used to identify the writer pod
            fn.metadata.labels = {"type": "model-monitoring-writer"}

            fn, ready = server.api.utils.functions.build_function(
                db_session=self.db_session, auth_info=self.auth_info, function=fn
            )

            logger.debug(
                "Submitted the writer deployment",
                writer_data=fn.to_dict(),
                writer_ready=ready,
            )
            # Create tsdb table for model monitoring application results
            self._create_tsdb_application_tables(project=fn.metadata.project)

    def deploy_histogram_data_drift_app(
        self, image: str, force_build: bool = False
    ) -> None:
        """
        Deploy the histogram data drift application.

        :param image: The image on with the function will run.
        :param force_build: If true, force the build of the function image. Default is False.
        """
        logger.info("Preparing the histogram data drift function")
        func = mlrun.model_monitoring.api._create_model_monitoring_function_base(
            project=self.project,
            func=_HISTOGRAM_DATA_DRIFT_APP_PATH,
            name=mm_constants.HistogramDataDriftApplicationConstants.NAME,
            application_class="HistogramDataDriftApplication",
            image=image,
        )

        if not force_build:
            self._reuse_image(
                function=func,
                name=mm_constants.HistogramDataDriftApplicationConstants.NAME,
                base_image=image,
            )

        if not mlrun.mlconf.is_ce_mode():
            logger.info("Setting the access key for the histogram data drift function")
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
                        access_key = os.environ.get("V3IO_ACCESS_KEY")
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

    def _initial_model_monitoring_stream_processing_function(
        self,
        stream_image: str,
        parquet_target: str,
        force_build: bool = False,
    ):
        """
        Initialize model monitoring stream processing function.

        :param stream_image:   The image of the model monitoring stream function.
        :param parquet_target: Path to model monitoring parquet file that will be generated by the
                               monitoring stream nuclio function.
        :param force_build:    If true, force the build of the function image. Default is False.

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

        if not force_build:
            self._reuse_image(
                function=function,
                name=mm_constants.MonitoringFunctionNames.STREAM,
                base_image=stream_image,
            )

        function.set_db_connection(
            server.api.api.utils.get_run_db_instance(self.db_session)
        )

        # Create monitoring serving graph
        stream_processor.apply_monitoring_serving_graph(function)

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

    def _get_model_monitoring_controller_function(
        self, image: str, force_build: bool = False
    ):
        """
        Initialize model monitoring controller function.

        :param image:         Base docker image to use for building the function container
        :param force_build:   If true, force the build of the function image. Default is False.
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

        if not force_build:
            self._reuse_image(
                function=function,
                name=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
                base_image=image,
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

    def _initial_model_monitoring_writer_function(
        self, writer_image: str, force_build: bool = False
    ):
        """
        Initialize model monitoring writer function.

        :param writer_image:                The image of the model monitoring writer function.
        :param force_build:                 If true, force the build of the function image. Default is False.

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

        if not force_build:
            self._reuse_image(
                function=function,
                name=mm_constants.MonitoringFunctionNames.WRITER,
                base_image=writer_image,
            )

        function.set_db_connection(
            server.api.api.utils.get_run_db_instance(self.db_session)
        )

        # Create writer monitoring serving graph
        graph = function.set_topology(mlrun.serving.states.StepKinds.flow)
        graph.to(ModelMonitoringWriter(project=self.project)).respond()  # writer

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
        logger.info(f"Deploying {function_name} function", project=self.project)
        return False

    @staticmethod
    def _create_tsdb_application_tables(project: str):
        """Each project writer service writes the application results into a single TSDB table and therefore the
        target table is created during the writer deployment"""

        tsdb_connector: mlrun.model_monitoring.db.TSDBConnector = (
            mlrun.model_monitoring.get_tsdb_connector(
                project=project,
            )
        )

        tsdb_connector.create_tsdb_application_tables()

    def _read_image_manifest(self):
        """Read the image manifest file that contains the information about the deployed model monitoring
        functions images"""
        manifest_path = self._get_manifest_path()
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                self._image_manifest = json.load(f)

    def _reuse_image(
        self,
        function: mlrun.runtimes.RemoteRuntime,
        name: str,
        base_image: str,
    ):
        # Get existing image of the controller function
        nuclio_image = self._get_existing_nuclio_image(
            name=name,
            base_image=base_image,
        )
        if nuclio_image:
            self._set_nuclio_image_config(function, nuclio_image)

    def _get_existing_nuclio_image(self, name: str, base_image: str):
        """Get the nuclio image that was built with the same base image for the given function name"""
        version_hash_key = self._get_version_hash_key()

        # if the image manifest is empty, read the image manifest as it might be updated
        if version_hash_key not in self._image_manifest:
            self._read_image_manifest()

        function_image_info = self._image_manifest.get(version_hash_key, {}).get(name)

        # we want to get the nuclio image that was built with the same base image
        if function_image_info and base_image in function_image_info:
            return function_image_info.get(base_image)

        return None

    def _update_image_manifest(
        self, base_image: str, extra_functions: list[str] = None
    ):
        """Update the image manifest file that contains the information about the deployed model monitoring"""
        extra_functions = extra_functions or []
        functions = mm_constants.MonitoringFunctionNames.list() + extra_functions
        version_hash_key = self._get_version_hash_key()
        save = False

        for function_name in functions:
            # if the function is already in the manifest with the same base image, don't override it
            if (
                function_name in self._image_manifest.get(version_hash_key, {})
                and base_image in self._image_manifest[version_hash_key][function_name]
            ):
                continue

            def _get_function_status():
                state, _, _, _, _, function_status = (
                    mlrun.runtimes.nuclio.function.get_nuclio_deploy_status(
                        name=function_name,
                        project=self.project,
                        tag="",
                        auth_info=self.auth_info,
                    )
                )

                if state != "ready":
                    raise Exception(f"Function {function_name} is not ready yet")

                return function_status

            try:
                status = mlrun.utils.helpers.retry_until_successful(
                    config.model_endpoint_monitoring.image_manifest.update_retry_interval,
                    config.model_endpoint_monitoring.image_manifest.update_retry_timeout,
                    logger,
                    True,
                    _get_function_status,
                )
            except Exception as exc:
                # we don't want to fail the entire process if one of the functions is not ready yet
                logger.error(
                    f"Failed to get the status of the function {function_name}",
                    exc=str(exc),
                )
                continue

            if "containerImage" not in status:
                logger.error(
                    f"Failed to get the container image of the function {function_name}",
                )
                continue

            if version_hash_key not in self._image_manifest:
                self._image_manifest[version_hash_key] = {}
            if function_name not in self._image_manifest[version_hash_key]:
                self._image_manifest[version_hash_key][function_name] = {}

            self._image_manifest[version_hash_key][function_name][base_image] = status[
                "containerImage"
            ]
            save = True

        # write the updated image manifest
        if save:
            manifest_path = self._get_manifest_path()
            mlrun.utils.ensure_file_path_exists(manifest_path)
            with open(manifest_path, "w") as f:
                json.dump(self._image_manifest, f)

    @staticmethod
    def _get_manifest_path():
        return os.path.join(
            config.httpdb.dirpath, config.model_endpoint_monitoring.image_manifest.path
        )

    @staticmethod
    def _get_version_hash_key():
        """Get the version hash key that will be used to store the model monitoring functions images in the manifest,
        the key is a combination of the mlrun version and the nuclio version"""
        return f"{config.version}-{config.nuclio_version}".encode().hex()

    @staticmethod
    def _set_nuclio_image_config(function, nuclio_image):
        function.set_config("spec.image", nuclio_image)
        function.set_config("spec.build.codeEntryType", "image")

        # make sure the image won't be built in nuclio by clearing up build values
        # TODO: remove this once the api will support deploying functions from an existing image directly
        function.spec.build.functionSourceCode = ""
        function.spec.build.source = ""
        function.spec.build.code_origin = ""
        function.spec.build.origin_filename = ""
        function.set_config("spec.build.functionSourceCode", "")
        function.set_config("spec.build.path", "")
        function.set_config("spec.build.image", "")


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
