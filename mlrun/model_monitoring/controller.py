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

import collections
import concurrent.futures
import datetime
import json
import os
import traceback
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, NamedTuple, Optional, Union, cast

import nuclio_sdk
import pandas as pd

import mlrun
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.feature_store as fstore
import mlrun.model_monitoring
import mlrun.model_monitoring.db._schedules as schedules
import mlrun.model_monitoring.helpers
import mlrun.platforms.iguazio
from mlrun.common.schemas import EndpointType
from mlrun.common.schemas.model_monitoring.constants import (
    ControllerEvent,
    ControllerEventEndpointPolicy,
    ControllerEventKind,
)
from mlrun.errors import err_to_str
from mlrun.model_monitoring.helpers import batch_dict2timedelta
from mlrun.utils import datetime_now, logger

_SECONDS_IN_DAY = int(datetime.timedelta(days=1).total_seconds())
_SECONDS_IN_MINUTE = 60


class _Interval(NamedTuple):
    start: datetime.datetime
    end: datetime.datetime


class _BatchWindow:
    def __init__(
        self,
        *,
        schedules_file: schedules.ModelMonitoringSchedulesFileEndpoint,
        application: str,
        timedelta_seconds: int,
        last_updated: int,
        first_request: int,
    ) -> None:
        """
        Initialize a batch window object that handles the batch interval time range
        for the monitoring functions.
        All the time values are in seconds.
        The start and stop time are in seconds since the epoch.
        """
        self._application = application
        self._first_request = first_request
        self._stop = last_updated
        self._step = timedelta_seconds
        self._db = schedules_file
        self._start = self._get_last_analyzed()

    def _get_saved_last_analyzed(self) -> Optional[int]:
        return cast(int, self._db.get_application_time(self._application))

    def _update_last_analyzed(self, last_analyzed: int) -> None:
        self._db.update_application_time(
            application=self._application, timestamp=last_analyzed
        )

    def _get_initial_last_analyzed(self) -> int:
        logger.info(
            "No last analyzed time was found for this endpoint and application, as this is "
            "probably the first time this application is running. Initializing last analyzed "
            "to the latest between first request time or last update time minus one day",
            application=self._application,
            first_request=self._first_request,
            last_updated=self._stop,
        )
        # max between one day and the base period
        first_period_in_seconds = max(_SECONDS_IN_DAY, self._step)
        return max(
            self._first_request,
            self._stop - first_period_in_seconds,
        )

    def _get_last_analyzed(self) -> int:
        saved_last_analyzed = self._get_saved_last_analyzed()
        if saved_last_analyzed is not None:
            return saved_last_analyzed
        else:
            last_analyzed = self._get_initial_last_analyzed()
            # Update the in-memory DB to avoid duplicate initializations
            self._update_last_analyzed(last_analyzed)
        return last_analyzed

    def get_intervals(self) -> Iterator[_Interval]:
        """Generate the batch interval time ranges."""
        entered = False
        # Iterate timestamp from start until timestamp <= stop - step
        # so that the last interval will end at (timestamp + step) <= stop.
        # Add 1 to stop - step to get <= and not <.
        for timestamp in range(self._start, self._stop - self._step + 1, self._step):
            entered = True
            start_time = datetime.datetime.fromtimestamp(
                timestamp, tz=datetime.timezone.utc
            )
            end_time = datetime.datetime.fromtimestamp(
                timestamp + self._step, tz=datetime.timezone.utc
            )
            yield _Interval(start_time, end_time)

            last_analyzed = timestamp + self._step
            self._update_last_analyzed(last_analyzed)
            logger.debug(
                "Updated the last analyzed time for this endpoint and application",
                application=self._application,
                last_analyzed=last_analyzed,
            )

        if not entered:
            logger.debug(
                "All the data is set, but no complete intervals were found. "
                "Wait for last_updated to be updated",
                application=self._application,
                start=self._start,
                stop=self._stop,
                step=self._step,
            )


class _BatchWindowGenerator(AbstractContextManager):
    def __init__(
        self, project: str, endpoint_id: str, window_length: Optional[int] = None
    ) -> None:
        """
        Initialize a batch window generator object that generates batch window objects
        for the monitoring functions.
        """
        self.batch_window: _BatchWindow = None
        self._project = project
        self._endpoint_id = endpoint_id
        self._timedelta = window_length
        self._schedules_file = schedules.ModelMonitoringSchedulesFileEndpoint(
            project=project, endpoint_id=endpoint_id
        )

    def __enter__(self) -> "_BatchWindowGenerator":
        self._schedules_file.__enter__()
        return super().__enter__()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        self._schedules_file.__exit__(
            exc_type=exc_type, exc_value=exc_value, traceback=traceback
        )

    def get_application_list(self) -> set[str]:
        return self._schedules_file.get_application_list()

    def get_min_last_analyzed(self) -> Optional[int]:
        return self._schedules_file.get_min_timestamp()

    @classmethod
    def _get_last_updated_time(
        cls, last_request: datetime.datetime, not_batch_endpoint: bool
    ) -> int:
        """
        Get the last updated time of a model endpoint.
        """
        last_updated = int(
            last_request.timestamp()
            - cast(
                float,
                mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs,
            )
        )
        if not not_batch_endpoint:
            # If the endpoint does not have a stream, `last_updated` should be
            # the minimum between the current time and the last updated time.
            # This compensates for the bumping mechanism - see
            # `update_model_endpoint_last_request`.
            last_updated = min(int(datetime_now().timestamp()), last_updated)
            logger.debug(
                "The endpoint does not have a stream", last_updated=last_updated
            )
        return last_updated

    def get_intervals(
        self,
        *,
        application: str,
        first_request: datetime.datetime,
        last_request: datetime.datetime,
        not_batch_endpoint: bool,
    ) -> Iterator[_Interval]:
        """
        Get the batch window for a specific endpoint and application.
        `first_request` and `last_request` are the timestamps of the first request and last
        request to the endpoint, respectively. They are guaranteed to be nonempty at this point.
        """
        self.batch_window = _BatchWindow(
            schedules_file=self._schedules_file,
            application=application,
            timedelta_seconds=self._timedelta,
            last_updated=self._get_last_updated_time(last_request, not_batch_endpoint),
            first_request=int(first_request.timestamp()),
        )
        yield from self.batch_window.get_intervals()


def _get_window_length() -> int:
    """Get the timedelta in seconds from the batch dictionary"""
    return int(
        batch_dict2timedelta(
            json.loads(
                cast(str, os.getenv(mm_constants.EventFieldType.BATCH_INTERVALS_DICT))
            )
        ).total_seconds()
    )


class MonitoringApplicationController:
    """
    The main object to handle the monitoring processing job. This object is used to get the required configurations and
    to manage the main monitoring drift detection process based on the current batch.
    Note that the MonitoringApplicationController object requires access keys along with valid project configurations.
    """

    _MAX_FEATURE_SET_PER_WORKER = 1000

    def __init__(self) -> None:
        """Initialize Monitoring Application Controller"""
        self.project = cast(str, mlrun.mlconf.default_project)
        self.project_obj = mlrun.get_run_db().get_project(name=self.project)
        logger.debug(f"Initializing {self.__class__.__name__}", project=self.project)

        self._window_length = _get_window_length()

        self.model_monitoring_access_key = self._get_model_monitoring_access_key()
        self.v3io_access_key = mlrun.mlconf.get_v3io_access_key()
        store, _, _ = mlrun.store_manager.get_or_create_store(
            mlrun.mlconf.artifact_path
        )
        self.storage_options = store.get_storage_options()
        self._controller_stream: Optional[
            Union[
                mlrun.platforms.iguazio.OutputStream,
                mlrun.platforms.iguazio.KafkaOutputStream,
            ]
        ] = None
        self._model_monitoring_stream: Optional[
            Union[
                mlrun.platforms.iguazio.OutputStream,
                mlrun.platforms.iguazio.KafkaOutputStream,
            ]
        ] = None
        self.applications_streams: dict[
            str,
            Union[
                mlrun.platforms.iguazio.OutputStream,
                mlrun.platforms.iguazio.KafkaOutputStream,
            ],
        ] = {}
        self.feature_sets: OrderedDict[str, mlrun.feature_store.FeatureSet] = (
            collections.OrderedDict()
        )
        self.tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
            project=self.project
        )

    @property
    def controller_stream(
        self,
    ) -> Union[
        mlrun.platforms.iguazio.OutputStream,
        mlrun.platforms.iguazio.KafkaOutputStream,
    ]:
        if self._controller_stream is None:
            self._controller_stream = mlrun.model_monitoring.helpers.get_output_stream(
                project=self.project,
                function_name=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
                v3io_access_key=self.v3io_access_key,
            )
        return self._controller_stream

    @property
    def model_monitoring_stream(
        self,
    ) -> Union[
        mlrun.platforms.iguazio.OutputStream,
        mlrun.platforms.iguazio.KafkaOutputStream,
    ]:
        if self._model_monitoring_stream is None:
            self._model_monitoring_stream = (
                mlrun.model_monitoring.helpers.get_output_stream(
                    project=self.project,
                    function_name=mm_constants.MonitoringFunctionNames.STREAM,
                    v3io_access_key=self.model_monitoring_access_key,
                )
            )
        return self._model_monitoring_stream

    @staticmethod
    def _get_model_monitoring_access_key() -> Optional[str]:
        access_key = os.getenv(mm_constants.ProjectSecretKeys.ACCESS_KEY)
        # allow access key to be empty and don't fetch v3io access key if not needed
        if access_key is None:
            access_key = mlrun.mlconf.get_v3io_access_key()
        return access_key

    def _should_monitor_endpoint(
        self,
        endpoint: mlrun.common.schemas.ModelEndpoint,
        application_names: set,
        base_period_minutes: int,
        schedules_file: schedules.ModelMonitoringSchedulesFileChief,
    ) -> bool:
        """
        checks if there is a need to monitor the given endpoint, we should monitor endpoint if it stands in the
        next conditions:
            1.  monitoring_mode is enabled
            2.  first request exists
            3.  last request exists
            4.  endpoint_type is not ROUTER
        if the four above conditions apply we require one of the two condition monitor:
            1.  never monitored the one of the endpoint applications meaning min_last_analyzed is None
            2.  min_last_analyzed stands in the condition for sending NOP event and this the first time regular event
            is sent with the combination of  current last_request  & current last_analyzed  per endpoint.
        """
        last_timestamp_sent = schedules_file.get_endpoint_last_request(
            endpoint.metadata.uid
        )
        last_analyzed_sent = schedules_file.get_endpoint_last_analyzed(
            endpoint.metadata.uid
        )
        logger.debug(
            "Chief should monitor endpoint check",
            last_timestamp_sent=last_timestamp_sent,
            last_analyzed_sent=last_analyzed_sent,
            uid=endpoint.metadata.uid,
        )
        if (
            # Is the model endpoint monitored?
            endpoint.status.monitoring_mode == mm_constants.ModelMonitoringMode.enabled
            # Was the model endpoint called? I.e., are the first and last requests nonempty?
            and endpoint.status.first_request
            and endpoint.status.last_request
            # Is the model endpoint not a router endpoint? Router endpoint has no feature stats
            and endpoint.metadata.endpoint_type.value
            != mm_constants.EndpointType.ROUTER.value
        ):
            with _BatchWindowGenerator(
                project=endpoint.metadata.project,
                endpoint_id=endpoint.metadata.uid,
            ) as batch_window_generator:
                current_time = mlrun.utils.datetime_now()
                current_min_last_analyzed = (
                    batch_window_generator.get_min_last_analyzed()
                )
                if (
                    # Different application names, or last analyzed never updated while there are application to monitor
                    application_names
                    and (
                        application_names
                        != batch_window_generator.get_application_list()
                        or not current_min_last_analyzed
                    )
                ):
                    return True
                elif (
                    # Does nop event will be sent to close the relevant window
                    self._should_send_nop_event(
                        base_period_minutes, current_min_last_analyzed, current_time
                    )
                    and (
                        int(endpoint.status.last_request.timestamp())
                        != last_timestamp_sent
                        or current_min_last_analyzed != last_analyzed_sent
                    )
                ):
                    # Write to schedule chief file the last_request, min_last_analyzed we pushed event to stream
                    schedules_file.update_endpoint_timestamps(
                        endpoint_uid=endpoint.metadata.uid,
                        last_request=int(endpoint.status.last_request.timestamp()),
                        last_analyzed=current_min_last_analyzed,
                    )
                    return True
                else:
                    logger.info(
                        "All the possible intervals were already analyzed, didn't push regular event",
                        endpoint_id=endpoint.metadata.uid,
                        last_analyzed=current_min_last_analyzed,
                        last_request=endpoint.status.last_request,
                    )
        else:
            logger.info(
                "Should not monitor model endpoint, didn't push regular event",
                endpoint_id=endpoint.metadata.uid,
                endpoint_name=endpoint.metadata.name,
                last_request=endpoint.status.last_request,
                first_request=endpoint.status.first_request,
                endpoint_type=endpoint.metadata.endpoint_type,
                feature_set_uri=endpoint.spec.monitoring_feature_set_uri,
            )
        return False

    @staticmethod
    def _should_send_nop_event(
        base_period_minutes: int,
        min_last_analyzed: int,
        current_time: datetime.datetime,
    ):
        if min_last_analyzed:
            return (
                current_time.timestamp() - min_last_analyzed
                >= datetime.timedelta(minutes=base_period_minutes).total_seconds()
                + mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs
            )
        else:
            return True

    def run(self, event: nuclio_sdk.Event) -> None:
        """
        Main method for controller chief, runs all the relevant monitoring applications for a single endpoint.
        Handles nop events logic.
        This method handles the following:
        1. Read applications from the event (endpoint_policy)
        2. Check model monitoring windows
        3. Send data to applications
        4. Pushes nop event to main stream if needed
        """
        logger.info("Start running monitoring controller worker")
        try:
            body = json.loads(event.body.decode("utf-8"))
        except Exception as e:
            logger.error(
                "Failed to decode event",
                exc=err_to_str(e),
            )
            return
        # Run single endpoint process
        self.model_endpoint_process(event=body)

    def model_endpoint_process(
        self,
        event: Optional[dict] = None,
    ) -> None:
        """
        Process a model endpoint and trigger the monitoring applications. This function running on different process
        for each endpoint.

        :param event:                       (dict) Event that triggered the monitoring process.
        """
        logger.info("Model endpoint process started", event=event)

        try:
            project_name = event[ControllerEvent.PROJECT]
            endpoint_id = event[ControllerEvent.ENDPOINT_ID]
            endpoint_name = event[ControllerEvent.ENDPOINT_NAME]
            applications_names = event[ControllerEvent.ENDPOINT_POLICY][
                ControllerEventEndpointPolicy.MONITORING_APPLICATIONS
            ]

            not_batch_endpoint = (
                event[ControllerEvent.ENDPOINT_TYPE] != EndpointType.BATCH_EP
            )

            logger.info(
                "Starting analyzing for", timestamp=event[ControllerEvent.TIMESTAMP]
            )
            last_stream_timestamp = datetime.datetime.fromisoformat(
                event[ControllerEvent.TIMESTAMP]
            )
            first_request = datetime.datetime.fromisoformat(
                event[ControllerEvent.FIRST_REQUEST]
            )
            with _BatchWindowGenerator(
                project=project_name,
                endpoint_id=endpoint_id,
                window_length=self._window_length,
            ) as batch_window_generator:
                for application in applications_names:
                    for (
                        start_infer_time,
                        end_infer_time,
                    ) in batch_window_generator.get_intervals(
                        application=application,
                        not_batch_endpoint=not_batch_endpoint,
                        first_request=first_request,
                        last_request=last_stream_timestamp,
                    ):
                        data_in_window = False
                        if not_batch_endpoint:
                            # Serving endpoint - get the relevant window data from the TSDB
                            prediction_metric = self.tsdb_connector.read_predictions(
                                start=start_infer_time,
                                end=end_infer_time,
                                endpoint_id=endpoint_id,
                            )
                            if prediction_metric.data:
                                data_in_window = True
                        else:
                            if endpoint_id not in self.feature_sets:
                                self.feature_sets[endpoint_id] = fstore.get_feature_set(
                                    event[ControllerEvent.FEATURE_SET_URI]
                                )
                            self.feature_sets.move_to_end(endpoint_id, last=False)
                            if (
                                len(self.feature_sets)
                                > self._MAX_FEATURE_SET_PER_WORKER
                            ):
                                self.feature_sets.popitem(last=True)
                            m_fs = self.feature_sets.get(endpoint_id)

                            # Batch endpoint - get the relevant window data from the parquet target
                            df = m_fs.to_dataframe(
                                start_time=start_infer_time,
                                end_time=end_infer_time,
                                time_column=mm_constants.EventFieldType.TIMESTAMP,
                                storage_options=self.storage_options,
                            )
                            if len(df) > 0:
                                data_in_window = True
                        if not data_in_window:
                            logger.info(
                                "No data found for the given interval",
                                start=start_infer_time,
                                end=end_infer_time,
                                endpoint_id=endpoint_id,
                            )
                        else:
                            logger.info(
                                "Data found for the given interval",
                                start=start_infer_time,
                                end=end_infer_time,
                                endpoint_id=endpoint_id,
                            )
                            self._push_to_applications(
                                start_infer_time=start_infer_time,
                                end_infer_time=end_infer_time,
                                endpoint_id=endpoint_id,
                                endpoint_name=endpoint_name,
                                project=project_name,
                                applications_names=[application],
                                model_monitoring_access_key=self.model_monitoring_access_key,
                                endpoint_updated=event[ControllerEvent.ENDPOINT_POLICY][
                                    ControllerEventEndpointPolicy.ENDPOINT_UPDATED
                                ],
                            )
                base_period = event[ControllerEvent.ENDPOINT_POLICY][
                    ControllerEventEndpointPolicy.BASE_PERIOD
                ]
                current_time = mlrun.utils.datetime_now()
                if (
                    self._should_send_nop_event(
                        base_period,
                        batch_window_generator.get_min_last_analyzed(),
                        current_time,
                    )
                    and event[ControllerEvent.KIND] != ControllerEventKind.NOP_EVENT
                ):
                    event = {
                        ControllerEvent.KIND: mm_constants.ControllerEventKind.NOP_EVENT,
                        ControllerEvent.PROJECT: project_name,
                        ControllerEvent.ENDPOINT_ID: endpoint_id,
                        ControllerEvent.ENDPOINT_NAME: endpoint_name,
                        ControllerEvent.TIMESTAMP: current_time.isoformat(
                            timespec="microseconds"
                        ),
                        ControllerEvent.ENDPOINT_POLICY: event[
                            ControllerEvent.ENDPOINT_POLICY
                        ],
                        ControllerEvent.ENDPOINT_TYPE: event[
                            ControllerEvent.ENDPOINT_TYPE
                        ],
                        ControllerEvent.FEATURE_SET_URI: event[
                            ControllerEvent.FEATURE_SET_URI
                        ],
                        ControllerEvent.FIRST_REQUEST: event[
                            ControllerEvent.FIRST_REQUEST
                        ],
                    }
                    self._push_to_main_stream(
                        event=event,
                        endpoint_id=endpoint_id,
                    )
            logger.info(
                "Finish analyze for", timestamp=event[ControllerEvent.TIMESTAMP]
            )

        except Exception:
            logger.exception(
                "Encountered an exception",
                endpoint_id=event[ControllerEvent.ENDPOINT_ID],
            )

    def _push_to_applications(
        self,
        start_infer_time: datetime.datetime,
        end_infer_time: datetime.datetime,
        endpoint_id: str,
        endpoint_name: str,
        project: str,
        applications_names: list[str],
        model_monitoring_access_key: str,
        endpoint_updated: str,
    ):
        """
        Pushes data to multiple stream applications.

        :param start_infer_time:            The beginning of the infer interval window.
        :param end_infer_time:              The end of the infer interval window.
        :param endpoint_id:                 Identifier for the model endpoint.
        :param project: mlrun               Project name.
        :param applications_names:          List of application names to which data will be pushed.
        :param model_monitoring_access_key: Access key to apply the model monitoring process.
        :param endpoint_updated:            str isoformet for the timestamp the model endpoint was updated
        """
        data = {
            mm_constants.ApplicationEvent.START_INFER_TIME: start_infer_time.isoformat(
                sep=" ", timespec="microseconds"
            ),
            mm_constants.ApplicationEvent.END_INFER_TIME: end_infer_time.isoformat(
                sep=" ", timespec="microseconds"
            ),
            mm_constants.ApplicationEvent.ENDPOINT_ID: endpoint_id,
            mm_constants.ApplicationEvent.ENDPOINT_NAME: endpoint_name,
            mm_constants.ApplicationEvent.ENDPOINT_UPDATED: endpoint_updated,
        }
        for app_name in applications_names:
            data.update({mm_constants.ApplicationEvent.APPLICATION_NAME: app_name})
            if app_name not in self.applications_streams:
                self.applications_streams[app_name] = (
                    mlrun.model_monitoring.helpers.get_output_stream(
                        project=project,
                        function_name=app_name,
                        v3io_access_key=model_monitoring_access_key,
                    )
                )
            app_stream = self.applications_streams.get(app_name)

            logger.info(
                "Pushing data to application stream",
                endpoint_id=endpoint_id,
                app_name=app_name,
                app_stream_type=str(type(app_stream)),
            )
            app_stream.push([data], partition_key=endpoint_id)

    def push_regular_event_to_controller_stream(self) -> None:
        """
        pushes a regular event to the controller stream.
        """
        logger.info("Starting monitoring controller chief")
        applications_names = []
        endpoints = self.project_obj.list_model_endpoints(tsdb_metrics=False).endpoints
        last_request_dict = self.tsdb_connector.get_last_request(
            endpoint_ids=[mep.metadata.uid for mep in endpoints]
        )
        if isinstance(last_request_dict, pd.DataFrame):
            last_request_dict = last_request_dict.set_index(
                mm_constants.EventFieldType.ENDPOINT_ID
            )[mm_constants.ModelEndpointSchema.LAST_REQUEST].to_dict()

        if not endpoints:
            logger.info("No model endpoints found", project=self.project)
            return
        monitoring_functions = self.project_obj.list_model_monitoring_functions()
        if monitoring_functions:
            # if monitoring_functions: - TODO : ML-7700
            #   Gets only application in ready state
            #   applications_names = list(
            #       {
            #           app.metadata.name
            #           for app in monitoring_functions
            #           if (
            #               app.status.state == "ready"
            #               # workaround for the default app, as its `status.state` is `None`
            #               or app.metadata.name
            #               == mm_constants.HistogramDataDriftApplicationConstants.NAME
            #           )
            #       }
            #   )
            applications_names = list(
                {app.metadata.name for app in monitoring_functions}
            )
        if not applications_names:
            logger.info("No monitoring functions found", project=self.project)
            return
        policy = {
            ControllerEventEndpointPolicy.MONITORING_APPLICATIONS: applications_names,
            ControllerEventEndpointPolicy.BASE_PERIOD: int(
                batch_dict2timedelta(
                    json.loads(
                        cast(
                            str,
                            os.getenv(mm_constants.EventFieldType.BATCH_INTERVALS_DICT),
                        )
                    )
                ).total_seconds()
                // _SECONDS_IN_MINUTE
            ),
        }
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(endpoints), 10)
        ) as pool:
            with schedules.ModelMonitoringSchedulesFileChief(
                self.project
            ) as schedule_file:
                for endpoint in endpoints:
                    last_request = last_request_dict.get(endpoint.metadata.uid, None)
                    if isinstance(last_request, float):
                        last_request = pd.to_datetime(last_request, unit="s", utc=True)
                    endpoint.status.last_request = (
                        last_request or endpoint.status.last_request
                    )
                    futures = {
                        pool.submit(
                            self.endpoint_to_regular_event,
                            endpoint,
                            policy,
                            set(applications_names),
                            schedule_file,
                        ): endpoint
                    }
                for future in concurrent.futures.as_completed(futures):
                    if future.exception():
                        exception = future.exception()
                        error = (
                            f"Failed to push event. Endpoint name: {futures[future].metadata.name}, "
                            f"endpoint uid: {futures[future].metadata.uid}, traceback:\n"
                        )
                        error += "".join(
                            traceback.format_exception(
                                None, exception, exception.__traceback__
                            )
                        )
                        logger.error(error)
        logger.info("Finishing monitoring controller chief")

    def endpoint_to_regular_event(
        self,
        endpoint: mlrun.common.schemas.ModelEndpoint,
        policy: dict,
        applications_names: set,
        schedule_file: schedules.ModelMonitoringSchedulesFileChief,
    ) -> None:
        if self._should_monitor_endpoint(
            endpoint,
            set(applications_names),
            policy.get(ControllerEventEndpointPolicy.BASE_PERIOD, 10),
            schedule_file,
        ):
            logger.debug(
                "Endpoint data is being prepared for regular event",
                endpoint_id=endpoint.metadata.uid,
                endpoint_name=endpoint.metadata.name,
                timestamp=endpoint.status.last_request.isoformat(
                    sep=" ", timespec="microseconds"
                ),
                first_request=endpoint.status.first_request.isoformat(
                    sep=" ", timespec="microseconds"
                ),
                endpoint_type=endpoint.metadata.endpoint_type,
                feature_set_uri=endpoint.spec.monitoring_feature_set_uri,
                endpoint_policy=json.dumps(policy),
            )
            policy[ControllerEventEndpointPolicy.ENDPOINT_UPDATED] = (
                endpoint.metadata.updated.isoformat()
            )
            self.push_to_controller_stream(
                kind=mm_constants.ControllerEventKind.REGULAR_EVENT,
                project=endpoint.metadata.project,
                endpoint_id=endpoint.metadata.uid,
                endpoint_name=endpoint.metadata.name,
                timestamp=endpoint.status.last_request.isoformat(
                    sep=" ", timespec="microseconds"
                ),
                first_request=endpoint.status.first_request.isoformat(
                    sep=" ", timespec="microseconds"
                ),
                endpoint_type=endpoint.metadata.endpoint_type.value,
                feature_set_uri=endpoint.spec.monitoring_feature_set_uri,
                endpoint_policy=policy,
            )

    def push_to_controller_stream(
        self,
        kind: str,
        project: str,
        endpoint_id: str,
        endpoint_name: str,
        timestamp: str,
        first_request: str,
        endpoint_type: int,
        feature_set_uri: str,
        endpoint_policy: dict[str, Any],
    ) -> None:
        """
        Pushes event data to controller stream.
        :param timestamp: the event timestamp str isoformat utc timezone
        :param first_request: the first request str isoformat utc timezone
        :param endpoint_policy: dictionary hold the monitoring policy
        :param kind: str event kind
        :param project: project name
        :param endpoint_id: endpoint id string
        :param endpoint_name: the endpoint name string
        :param endpoint_type: Enum of the endpoint type
        :param feature_set_uri: the feature set uri string
        """
        event = {
            ControllerEvent.KIND.value: kind,
            ControllerEvent.PROJECT.value: project,
            ControllerEvent.ENDPOINT_ID.value: endpoint_id,
            ControllerEvent.ENDPOINT_NAME.value: endpoint_name,
            ControllerEvent.TIMESTAMP.value: timestamp,
            ControllerEvent.FIRST_REQUEST.value: first_request,
            ControllerEvent.ENDPOINT_TYPE.value: endpoint_type,
            ControllerEvent.FEATURE_SET_URI.value: feature_set_uri,
            ControllerEvent.ENDPOINT_POLICY.value: endpoint_policy,
        }
        logger.info(
            "Pushing data to controller stream",
            event=event,
            endpoint_id=endpoint_id,
            controller_stream_type=str(type(self.controller_stream)),
        )
        self.controller_stream.push([event], partition_key=endpoint_id)

    def _push_to_main_stream(self, event: dict, endpoint_id: str) -> None:
        """
        Pushes the given event to model monitoring stream
        :param event: event dictionary to push to stream
        :param endpoint_id: endpoint id string
        """
        logger.info(
            "Pushing data to main stream, NOP event is been generated",
            event=json.dumps(event),
            endpoint_id=endpoint_id,
            mm_stream_type=str(type(self.model_monitoring_stream)),
        )
        self.model_monitoring_stream.push([event], partition_key=endpoint_id)


def handler(context: nuclio_sdk.Context, event: nuclio_sdk.Event) -> None:
    """
    Run model monitoring application processor

    :param context: the Nuclio context
    :param event:   trigger event
    """
    logger.info(
        "Controller got event",
        trigger=event.trigger,
        trigger_kind=event.trigger.kind,
    )

    if event.trigger.kind in mm_constants.CRON_TRIGGER_KINDS:
        # Runs controller chief:
        context.user_data.monitor_app_controller.push_regular_event_to_controller_stream()
    elif event.trigger.kind in mm_constants.STREAM_TRIGGER_KINDS:
        # Runs controller worker:
        context.user_data.monitor_app_controller.run(event)
    else:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Wrong trigger kind for model monitoring controller"
        )


def init_context(context):
    monitor_app_controller = MonitoringApplicationController()
    setattr(context.user_data, "monitor_app_controller", monitor_app_controller)
    context.logger.info("Monitoring application controller initialized")
