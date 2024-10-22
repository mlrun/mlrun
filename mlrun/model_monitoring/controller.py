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
from typing import NamedTuple, NewType, Optional, cast

import nuclio

import mlrun
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.data_types.infer
import mlrun.model_monitoring.db
from mlrun.datastore import DataItem, get_stream_pusher
from mlrun.errors import err_to_str
from mlrun.model_monitoring.helpers import (
    _BatchDict,
    batch_dict2timedelta,
    get_monitoring_schedules_data,
    get_stream_path,
)
from mlrun.utils import datetime_now, logger

_UTF_8 = "utf-8"
_SECONDS_IN_DAY = int(datetime.timedelta(days=1).total_seconds())


# `Schedules` is a dictionary of registered application name as key and Unix timestamp as value
_Schedules = NewType("_Schedules", dict[str, int])


class _Interval(NamedTuple):
    start: datetime.datetime
    end: datetime.datetime


class _BatchWindow:
    def __init__(
        self,
        *,
        endpoint_app_schedules: DataItem,
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
        # `db` is the persistent version of the monitoring schedules.
        self._db = endpoint_app_schedules
        # `schedules` is an in-memory copy of the DB for all the applications for
        # the same model endpoint.
        self._schedules, self._start = self._get_schedules_and_last_analyzed()

    def _update_db(self, schedules: _Schedules) -> None:
        self._db.put(json.dumps(schedules))

    def _init_last_analyzed(self) -> int:
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

    def _get_schedules_and_last_analyzed(self) -> tuple[_Schedules, int]:
        content = self._db.get(encoding=_UTF_8)
        try:
            schedules = json.loads(content)
            if self._application in schedules:
                return _Schedules(schedules), schedules[self._application]
            else:
                last_analyzed = self._init_last_analyzed()
                schedules.update(_Schedules({self._application: last_analyzed}))
        except json.JSONDecodeError:
            # Using the earliest safe time to avoid TSDB misorders
            last_analyzed = self._stop
            schedules = _Schedules({self._application: last_analyzed})
            logger.warning(
                "The monitoring schedules file is corrupted, resetting it "
                "with the last request as last_analyzed",
                path=self._db.url,
                content=content,
                last_analyzed=last_analyzed,
            )

        # Updating the DB to avoid reoccurrence of missing keys, corrupted data, or a missing file
        self._update_db(schedules)
        return schedules, last_analyzed

    def _update_last_analyzed(self, last_analyzed: int) -> None:
        logger.debug(
            "Updating the last analyzed time for this endpoint and application",
            application=self._application,
            last_analyzed=last_analyzed,
        )
        self._schedules.update(_Schedules({self._application: last_analyzed}))
        self._update_db(self._schedules)

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
            self._update_last_analyzed(timestamp + self._step)
        if not entered:
            logger.debug(
                "All the data is set, but no complete intervals were found. "
                "Wait for last_updated to be updated",
                application=self._application,
                start=self._start,
                stop=self._stop,
                step=self._step,
            )


class _BatchWindowGenerator:
    def __init__(self, batch_dict: dict[str, int]) -> None:
        """
        Initialize a batch window generator object that generates batch window objects
        for the monitoring functions.
        """
        self._batch_dict = batch_dict
        self._timedelta = self._get_timedelta()

    def _get_timedelta(self) -> int:
        """Get the timedelta in seconds from the batch dictionary"""
        return int(
            batch_dict2timedelta(cast(_BatchDict, self._batch_dict)).total_seconds()
        )

    @classmethod
    def _get_last_updated_time(cls, last_request: str, has_stream: bool) -> int:
        """
        Get the last updated time of a model endpoint.
        """
        last_updated = int(
            cls._date_string2timestamp(last_request)
            - cast(
                float,
                mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs,
            )
        )
        if not has_stream:
            # If the endpoint does not have a stream, `last_updated` should be
            # the minimum between the current time and the last updated time.
            # This compensates for the bumping mechanism - see
            # `update_model_endpoint_last_request`.
            last_updated = min(int(datetime_now().timestamp()), last_updated)
            logger.debug(
                "The endpoint does not have a stream", last_updated=last_updated
            )
        return last_updated

    @staticmethod
    def _date_string2timestamp(date_string: str) -> int:
        return int(datetime.datetime.fromisoformat(date_string).timestamp())

    def get_batch_window(
        self,
        *,
        project: str,
        endpoint: str,
        application: str,
        first_request: str,
        last_request: str,
        has_stream: bool,
    ) -> _BatchWindow:
        """
        Get the batch window for a specific endpoint and application.
        first_request is the first request time to the endpoint.
        """

        return _BatchWindow(
            endpoint_app_schedules=get_monitoring_schedules_data(
                project=project, endpoint_id=endpoint
            ),
            application=application,
            timedelta_seconds=self._timedelta,
            last_updated=self._get_last_updated_time(last_request, has_stream),
            first_request=self._date_string2timestamp(first_request),
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

        self.db = mlrun.model_monitoring.get_store_object(project=self.project)

        self._batch_window_generator = _BatchWindowGenerator(
            batch_dict=json.loads(
                cast(str, os.getenv(mm_constants.EventFieldType.BATCH_INTERVALS_DICT))
            )
        )

        self.model_monitoring_access_key = self._get_model_monitoring_access_key()
        self.tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
            project=self.project
        )

    @staticmethod
    def _get_model_monitoring_access_key() -> Optional[str]:
        access_key = os.getenv(mm_constants.ProjectSecretKeys.ACCESS_KEY)
        # allow access key to be empty and don't fetch v3io access key if not needed
        if access_key is None:
            access_key = mlrun.mlconf.get_v3io_access_key()
        return access_key

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
            endpoints = self.db.list_model_endpoints(include_stats=True)
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
        # Initialize a process pool that will be used to run each endpoint applications on a dedicated process
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(endpoints), 10),
        ) as pool:
            for endpoint in endpoints:
                if (
                    endpoint[mm_constants.EventFieldType.ACTIVE]
                    and endpoint[mm_constants.EventFieldType.MONITORING_MODE]
                    == mm_constants.ModelMonitoringMode.enabled.value
                ):
                    # Skip router endpoint:
                    if (
                        int(endpoint[mm_constants.EventFieldType.ENDPOINT_TYPE])
                        == mm_constants.EndpointType.ROUTER
                    ):
                        # Router endpoint has no feature stats
                        logger.info(
                            f"{endpoint[mm_constants.EventFieldType.UID]} is router, skipping"
                        )
                        continue
                    pool.submit(
                        MonitoringApplicationController.model_endpoint_process,
                        endpoint=endpoint,
                        applications_names=applications_names,
                        batch_window_generator=self._batch_window_generator,
                        project=self.project,
                        model_monitoring_access_key=self.model_monitoring_access_key,
                        tsdb_connector=self.tsdb_connector,
                    )

    @classmethod
    def model_endpoint_process(
        cls,
        endpoint: dict,
        applications_names: list[str],
        batch_window_generator: _BatchWindowGenerator,
        project: str,
        model_monitoring_access_key: str,
        tsdb_connector: mlrun.model_monitoring.db.tsdb.TSDBConnector,
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
        :param tsdb_connector:              (mlrun.model_monitoring.db.tsdb.TSDBConnector) TSDB connector
        """
        endpoint_id = endpoint[mm_constants.EventFieldType.UID]
        # if false the endpoint represent batch infer step.
        has_stream = endpoint[mm_constants.EventFieldType.STREAM_PATH] != ""
        try:
            for application in applications_names:
                batch_window = batch_window_generator.get_batch_window(
                    project=project,
                    endpoint=endpoint_id,
                    application=application,
                    first_request=endpoint[mm_constants.EventFieldType.FIRST_REQUEST],
                    last_request=endpoint[mm_constants.EventFieldType.LAST_REQUEST],
                    has_stream=has_stream,
                )

                for start_infer_time, end_infer_time in batch_window.get_intervals():
                    prediction_metric = tsdb_connector.read_predictions(
                        endpoint_id=endpoint_id,
                        start=start_infer_time,
                        end=end_infer_time,
                    )
                    if not prediction_metric.data and has_stream:
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
                            project=project,
                            applications_names=[application],
                            model_monitoring_access_key=model_monitoring_access_key,
                        )
        except Exception:
            logger.exception(
                "Encountered an exception",
                endpoint_id=endpoint[mm_constants.EventFieldType.UID],
            )

    @staticmethod
    def _push_to_applications(
        start_infer_time: datetime.datetime,
        end_infer_time: datetime.datetime,
        endpoint_id: str,
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


def handler(context: nuclio.Context, event: nuclio.Event) -> None:
    """
    Run model monitoring application processor

    :param context: the Nuclio context
    :param event:   trigger event
    """
    MonitoringApplicationController().run()
