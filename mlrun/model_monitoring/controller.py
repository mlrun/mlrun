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

import concurrent.futures
import datetime
import json
import os
from collections.abc import Iterator
from contextlib import AbstractContextManager
from types import TracebackType
from typing import NamedTuple, Optional, cast

import nuclio_sdk

import mlrun
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.feature_store as fstore
import mlrun.model_monitoring
from mlrun.common.schemas import EndpointType
from mlrun.datastore import get_stream_pusher
from mlrun.errors import err_to_str
from mlrun.model_monitoring.db._schedules import ModelMonitoringSchedulesFile
from mlrun.model_monitoring.helpers import batch_dict2timedelta, get_stream_path
from mlrun.utils import datetime_now, logger

_SECONDS_IN_DAY = int(datetime.timedelta(days=1).total_seconds())


class _Interval(NamedTuple):
    start: datetime.datetime
    end: datetime.datetime


class _BatchWindow:
    def __init__(
        self,
        *,
        schedules_file: ModelMonitoringSchedulesFile,
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
    def __init__(self, project: str, endpoint_id: str, window_length: int) -> None:
        """
        Initialize a batch window generator object that generates batch window objects
        for the monitoring functions.
        """
        self._project = project
        self._endpoint_id = endpoint_id
        self._timedelta = window_length
        self._schedules_file = ModelMonitoringSchedulesFile(
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
        batch_window = _BatchWindow(
            schedules_file=self._schedules_file,
            application=application,
            timedelta_seconds=self._timedelta,
            last_updated=self._get_last_updated_time(last_request, not_batch_endpoint),
            first_request=int(first_request.timestamp()),
        )
        yield from batch_window.get_intervals()


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

    def __init__(self) -> None:
        """Initialize Monitoring Application Controller"""
        self.project = cast(str, mlrun.mlconf.default_project)
        self.project_obj = mlrun.load_project(name=self.project, url=self.project)

        logger.debug(f"Initializing {self.__class__.__name__}", project=self.project)

        self._window_length = _get_window_length()

        self.model_monitoring_access_key = self._get_model_monitoring_access_key()
        self.storage_options = None
        if mlrun.mlconf.artifact_path.startswith("s3://"):
            self.storage_options = mlrun.mlconf.get_s3_storage_options()

    @staticmethod
    def _get_model_monitoring_access_key() -> Optional[str]:
        access_key = os.getenv(mm_constants.ProjectSecretKeys.ACCESS_KEY)
        # allow access key to be empty and don't fetch v3io access key if not needed
        if access_key is None:
            access_key = mlrun.mlconf.get_v3io_access_key()
        return access_key

    @staticmethod
    def _should_monitor_endpoint(endpoint: mlrun.common.schemas.ModelEndpoint) -> bool:
        return (
            # Is the model endpoint monitored?
            endpoint.status.monitoring_mode == mm_constants.ModelMonitoringMode.enabled
            # Was the model endpoint called? I.e., are the first and last requests nonempty?
            and endpoint.status.first_request
            and endpoint.status.last_request
            # Is the model endpoint not a router endpoint? Router endpoint has no feature stats
            and endpoint.metadata.endpoint_type.value
            != mm_constants.EndpointType.ROUTER.value
        )

    def run(self) -> None:
        """
        Main method for run all the relevant monitoring applications on each endpoint.
        This method handles the following:
        1. List model endpoints
        2. List applications
        3. Check model monitoring windows
        4. Send data to applications
        5. Delete old parquets
        """
        logger.info("Start running monitoring controller")
        try:
            applications_names = []
            endpoints_list = mlrun.db.get_run_db().list_model_endpoints(
                project=self.project, tsdb_metrics=True
            )
            endpoints = endpoints_list.endpoints
            if not endpoints:
                logger.info("No model endpoints found", project=self.project)
                return
            monitoring_functions = self.project_obj.list_model_monitoring_functions()
            if monitoring_functions:
                applications_names = list(
                    {app.metadata.name for app in monitoring_functions}
                )
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
            if not applications_names:
                logger.info("No monitoring functions found", project=self.project)
                return
            logger.info(
                "Starting to iterate over the applications",
                applications=applications_names,
            )

        except Exception as e:
            logger.error(
                "Failed to list endpoints and monitoring applications",
                exc=err_to_str(e),
            )
            return
        # Initialize a thread pool that will be used to monitor each endpoint on a dedicated thread
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(endpoints), 10)
        ) as pool:
            for endpoint in endpoints:
                if self._should_monitor_endpoint(endpoint):
                    pool.submit(
                        MonitoringApplicationController.model_endpoint_process,
                        project=self.project,
                        endpoint=endpoint,
                        applications_names=applications_names,
                        window_length=self._window_length,
                        model_monitoring_access_key=self.model_monitoring_access_key,
                        storage_options=self.storage_options,
                    )
                else:
                    logger.debug(
                        "Skipping endpoint, not ready or not suitable for monitoring",
                        endpoint_id=endpoint.metadata.uid,
                        endpoint_name=endpoint.metadata.name,
                    )
        logger.info("Finished running monitoring controller")

    @classmethod
    def model_endpoint_process(
        cls,
        project: str,
        endpoint: mlrun.common.schemas.ModelEndpoint,
        applications_names: list[str],
        window_length: int,
        model_monitoring_access_key: str,
        storage_options: Optional[dict] = None,
    ) -> None:
        """
        Process a model endpoint and trigger the monitoring applications. This function running on different process
        for each endpoint. In addition, this function will generate a parquet file that includes the relevant data
        for a specific time range.

        :param endpoint:                    (dict) Model endpoint record.
        :param applications_names:          (list[str]) List of application names to push results to.
        :param batch_window_generator:      (_BatchWindowGenerator) An object that generates _BatchWindow objects.
        :param project:                     (str) Project name.
        :param model_monitoring_access_key: (str) Access key to apply the model monitoring process.
        :param storage_options:             (dict) Storage options for reading the infer parquet files.
        """
        endpoint_id = endpoint.metadata.uid
        not_batch_endpoint = not (
            endpoint.metadata.endpoint_type == EndpointType.BATCH_EP
        )
        m_fs = fstore.get_feature_set(endpoint.spec.monitoring_feature_set_uri)
        try:
            with _BatchWindowGenerator(
                project=project, endpoint_id=endpoint_id, window_length=window_length
            ) as batch_window_generator:
                for application in applications_names:
                    for (
                        start_infer_time,
                        end_infer_time,
                    ) in batch_window_generator.get_intervals(
                        application=application,
                        first_request=endpoint.status.first_request,
                        last_request=endpoint.status.last_request,
                        not_batch_endpoint=not_batch_endpoint,
                    ):
                        df = m_fs.to_dataframe(
                            start_time=start_infer_time,
                            end_time=end_infer_time,
                            time_column=mm_constants.EventFieldType.TIMESTAMP,
                            storage_options=storage_options,
                        )
                        if len(df) == 0:
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
                            cls._push_to_applications(
                                start_infer_time=start_infer_time,
                                end_infer_time=end_infer_time,
                                endpoint_id=endpoint_id,
                                endpoint_name=endpoint.metadata.name,
                                project=project,
                                applications_names=[application],
                                model_monitoring_access_key=model_monitoring_access_key,
                            )
                logger.info("Finished processing endpoint", endpoint_id=endpoint_id)

        except Exception:
            logger.exception(
                "Encountered an exception",
                endpoint_id=endpoint.metadata.uid,
            )

    @staticmethod
    def _push_to_applications(
        start_infer_time: datetime.datetime,
        end_infer_time: datetime.datetime,
        endpoint_id: str,
        endpoint_name: str,
        project: str,
        applications_names: list[str],
        model_monitoring_access_key: str,
    ):
        """
        Pushes data to multiple stream applications.

        :param start_infer_time:            The beginning of the infer interval window.
        :param end_infer_time:              The end of the infer interval window.
        :param endpoint_id:                 Identifier for the model endpoint.
        :param project: mlrun               Project name.
        :param applications_names:          List of application names to which data will be pushed.
        :param model_monitoring_access_key: Access key to apply the model monitoring process.

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
            mm_constants.ApplicationEvent.OUTPUT_STREAM_URI: get_stream_path(
                project=project,
                function_name=mm_constants.MonitoringFunctionNames.WRITER,
            ),
        }
        for app_name in applications_names:
            data.update({mm_constants.ApplicationEvent.APPLICATION_NAME: app_name})
            stream_uri = get_stream_path(project=project, function_name=app_name)

            logger.info(
                "Pushing data to application stream",
                endpoint_id=endpoint_id,
                app_name=app_name,
                stream_uri=stream_uri,
            )
            get_stream_pusher(stream_uri, access_key=model_monitoring_access_key).push(
                [data]
            )


def handler(context: nuclio_sdk.Context, event: nuclio_sdk.Event) -> None:
    """
    Run model monitoring application processor

    :param context: the Nuclio context
    :param event:   trigger event
    """
    MonitoringApplicationController().run()
