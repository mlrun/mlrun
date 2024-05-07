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
import re
from collections.abc import Iterator
from typing import Any, NamedTuple, Optional, Union, cast

import nuclio

import mlrun
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.data_types.infer
import mlrun.feature_store as fstore
import mlrun.model_monitoring.db.stores
from mlrun.common.model_monitoring.helpers import FeatureStats, pad_features_hist
from mlrun.datastore import get_stream_pusher
from mlrun.datastore.targets import ParquetTarget
from mlrun.errors import err_to_str
from mlrun.model_monitoring.helpers import (
    _BatchDict,
    batch_dict2timedelta,
    calculate_inputs_statistics,
    get_monitoring_parquet_path,
    get_stream_path,
)
from mlrun.utils import datetime_now, logger


class _Interval(NamedTuple):
    start: datetime.datetime
    end: datetime.datetime


class _BatchWindow:
    def __init__(
        self,
        project: str,
        endpoint: str,
        application: str,
        timedelta_seconds: int,
        last_updated: Optional[int],
        first_request: Optional[int],
    ) -> None:
        """
        Initialize a batch window object that handles the batch interval time range
        for the monitoring functions.
        All the time values are in seconds.
        The start and stop time are in seconds since the epoch.
        """
        self.project = project
        self._endpoint = endpoint
        self._application = application
        self._first_request = first_request
        self._stop = last_updated
        self._step = timedelta_seconds
        self._db = mlrun.model_monitoring.get_store_object(project=self.project)
        self._start = self._get_last_analyzed()

    def _get_last_analyzed(self) -> Optional[int]:
        try:
            last_analyzed = self._db.get_last_analyzed(
                endpoint_id=self._endpoint,
                application_name=self._application,
            )
        except mlrun.errors.MLRunNotFoundError:
            logger.info(
                "No last analyzed time was found for this endpoint and "
                "application, as this is probably the first time this "
                "application is running. Using the latest between first "
                "request time or last update time minus one day instead",
                endpoint=self._endpoint,
                application=self._application,
                first_request=self._first_request,
                last_updated=self._stop,
            )

            if self._first_request and self._stop:
                # TODO : Change the timedelta according to the policy.
                first_period_in_seconds = max(
                    int(datetime.timedelta(days=1).total_seconds()), self._step
                )  # max between one day and the base period
                return max(
                    self._first_request,
                    self._stop - first_period_in_seconds,
                )
            return self._first_request

        logger.info(
            "Got the last analyzed time for this endpoint and application",
            endpoint=self._endpoint,
            application=self._application,
            last_analyzed=last_analyzed,
        )
        return last_analyzed

    def _update_last_analyzed(self, last_analyzed: int) -> None:
        logger.info(
            "Updating the last analyzed time for this endpoint and application",
            endpoint=self._endpoint,
            application=self._application,
            last_analyzed=last_analyzed,
        )

        self._db.update_last_analyzed(
            endpoint_id=self._endpoint,
            application_name=self._application,
            last_analyzed=last_analyzed,
        )

    def get_intervals(
        self,
    ) -> Iterator[_Interval]:
        """Generate the batch interval time ranges."""
        if self._start is not None and self._stop is not None:
            entered = False
            # Iterate timestamp from start until timestamp <= stop - step
            # so that the last interval will end at (timestamp + step) <= stop.
            # Add 1 to stop - step to get <= and not <.
            for timestamp in range(
                self._start, self._stop - self._step + 1, self._step
            ):
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
                logger.info(
                    "All the data is set, but no complete intervals were found. "
                    "Wait for last_updated to be updated",
                    endpoint=self._endpoint,
                    application=self._application,
                    start=self._start,
                    stop=self._stop,
                    step=self._step,
                )
        else:
            logger.warn(
                "The first request time is not found for this endpoint. "
                "No intervals will be generated",
                endpoint=self._endpoint,
                application=self._application,
                start=self._start,
                stop=self._stop,
            )


class _BatchWindowGenerator:
    def __init__(self, batch_dict: Union[dict, str]) -> None:
        """
        Initialize a batch window generator object that generates batch window objects
        for the monitoring functions.
        """
        self._batch_dict = batch_dict
        self._norm_batch_dict()
        self._timedelta = self._get_timedelta()

    def _norm_batch_dict(self) -> None:
        # TODO: This will be removed once the job params can be parsed with different types
        # Convert batch dict string into a dictionary
        if isinstance(self._batch_dict, str):
            self._parse_batch_dict_str()

    def _parse_batch_dict_str(self) -> None:
        """Convert batch dictionary string into a valid dictionary"""
        characters_to_remove = "{} "
        pattern = "[" + characters_to_remove + "]"
        # Remove unnecessary characters from the provided string
        batch_list = re.sub(pattern, "", self._batch_dict).split(",")
        # Initialize the dictionary of batch interval ranges
        self._batch_dict = {}
        for pair in batch_list:
            pair_list = pair.split(":")
            self._batch_dict[pair_list[0]] = float(pair_list[1])

    def _get_timedelta(self) -> int:
        """Get the timedelta in seconds from the batch dictionary"""
        return int(
            batch_dict2timedelta(cast(_BatchDict, self._batch_dict)).total_seconds()
        )

    @classmethod
    def _get_last_updated_time(
        cls, last_request: Optional[str], has_stream: bool
    ) -> Optional[int]:
        """
        Get the last updated time of a model endpoint.
        """
        if not last_request:
            return None
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
            # `bump_model_endpoint_last_request`.
            last_updated = min(int(datetime_now().timestamp()), last_updated)
            logger.debug(
                "The endpoint does not have a stream", last_updated=last_updated
            )
        return last_updated

    @classmethod
    def _normalize_first_request(
        cls, first_request: Optional[str], endpoint: str
    ) -> Optional[int]:
        if not first_request:
            logger.debug(
                "There is no first request time for this endpoint.",
                endpoint=endpoint,
                first_request=first_request,
            )
            return None
        return cls._date_string2timestamp(first_request)

    @staticmethod
    def _date_string2timestamp(date_string: str) -> int:
        return int(datetime.datetime.fromisoformat(date_string).timestamp())

    def get_batch_window(
        self,
        project: str,
        endpoint: str,
        application: str,
        first_request: Optional[str],
        last_request: Optional[str],
        has_stream: bool,
    ) -> _BatchWindow:
        """
        Get the batch window for a specific endpoint and application.
        first_request is the first request time to the endpoint.
        """

        return _BatchWindow(
            project=project,
            endpoint=endpoint,
            application=application,
            timedelta_seconds=self._timedelta,
            last_updated=self._get_last_updated_time(last_request, has_stream),
            first_request=self._normalize_first_request(first_request, endpoint),
        )


class MonitoringApplicationController:
    """
    The main object to handle the monitoring processing job. This object is used to get the required configurations and
    to manage the main monitoring drift detection process based on the current batch.
    Note that the MonitoringApplicationController object requires access keys along with valid project configurations.
    """

    def __init__(
        self,
        mlrun_context: mlrun.run.MLClientCtx,
        project: str,
    ):
        """
        Initialize Monitoring Application Processor object.

        :param mlrun_context:               An MLRun context.
        :param project:                     Project name.
        """
        self.context = mlrun_context
        self.project = project
        self.project_obj = mlrun.get_or_create_project(project)

        mlrun_context.logger.debug(
            f"Initializing {self.__class__.__name__}", project=project
        )

        self.db = mlrun.model_monitoring.get_store_object(project=project)

        self._batch_window_generator = _BatchWindowGenerator(
            batch_dict=json.loads(
                mlrun.get_secret_or_env(
                    mm_constants.EventFieldType.BATCH_INTERVALS_DICT
                )
            )
        )

        self.model_monitoring_access_key = self._get_model_monitoring_access_key()
        self.parquet_directory = get_monitoring_parquet_path(
            self.project_obj,
            kind=mm_constants.FileTargetKind.APPS_PARQUET,
        )
        self.storage_options = None
        if not mlrun.mlconf.is_ce_mode():
            self._initialize_v3io_configurations()
        elif self.parquet_directory.startswith("s3://"):
            self.storage_options = mlrun.mlconf.get_s3_storage_options()

    @staticmethod
    def _get_model_monitoring_access_key() -> Optional[str]:
        access_key = os.getenv(mm_constants.ProjectSecretKeys.ACCESS_KEY)
        # allow access key to be empty and don't fetch v3io access key if not needed
        if access_key is None:
            access_key = mlrun.mlconf.get_v3io_access_key()
        return access_key

    def _initialize_v3io_configurations(self) -> None:
        self.v3io_framesd = mlrun.mlconf.v3io_framesd
        self.v3io_api = mlrun.mlconf.v3io_api
        self.storage_options = dict(
            v3io_access_key=self.model_monitoring_access_key, v3io_api=self.v3io_api
        )

    def run(self, event: nuclio.Event):
        """
        Main method for run all the relevant monitoring applications on each endpoint

        :param event:   trigger event
        """
        logger.info("Start running monitoring controller")
        try:
            applications_names = []
            endpoints = self.db.list_model_endpoints()
            if not endpoints:
                self.context.logger.info(
                    "No model endpoints found", project=self.project
                )
                return
            monitoring_functions = self.project_obj.list_model_monitoring_functions()
            if monitoring_functions:
                # Gets only application in ready state
                applications_names = list(
                    {
                        app.metadata.name
                        for app in monitoring_functions
                        if (
                            app.status.state == "ready"
                            # workaround for the default app, as its `status.state` is `None`
                            or app.metadata.name
                            == mm_constants.HistogramDataDriftApplicationConstants.NAME
                        )
                    }
                )
            if not applications_names:
                self.context.logger.info(
                    "No monitoring functions found", project=self.project
                )
                return
            self.context.logger.info(
                "Starting to iterate over the applications",
                applications=applications_names,
            )

        except Exception as e:
            self.context.logger.error(
                "Failed to list endpoints and monitoring applications",
                exc=err_to_str(e),
            )
            return
        # Initialize a process pool that will be used to run each endpoint applications on a dedicated process
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(len(endpoints), 10),
        )
        futures = []
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
                        f"{endpoint[mm_constants.EventFieldType.UID]} is router skipping"
                    )
                    continue
                future = pool.submit(
                    MonitoringApplicationController.model_endpoint_process,
                    endpoint=endpoint,
                    applications_names=applications_names,
                    batch_window_generator=self._batch_window_generator,
                    project=self.project,
                    parquet_directory=self.parquet_directory,
                    storage_options=self.storage_options,
                    model_monitoring_access_key=self.model_monitoring_access_key,
                )
                futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                self.context.log_results(result)

        self._delete_old_parquet(endpoints=endpoints)

    @classmethod
    def model_endpoint_process(
        cls,
        endpoint: dict,
        applications_names: list[str],
        batch_window_generator: _BatchWindowGenerator,
        project: str,
        parquet_directory: str,
        storage_options: dict,
        model_monitoring_access_key: str,
    ) -> Optional[dict[str, list[str]]]:
        """
        Process a model endpoint and trigger the monitoring applications. This function running on different process
        for each endpoint. In addition, this function will generate a parquet file that includes the relevant data
        for a specific time range.

        :param endpoint:                    (dict) Model endpoint record.
        :param applications_names:          (list[str]) List of application names to push results to.
        :param batch_window_generator:      (_BatchWindowGenerator) An object that generates _BatchWindow objects.
        :param project:                     (str) Project name.
        :param parquet_directory:           (str) Directory to store application parquet files
        :param storage_options:             (dict) Storage options for writing ParquetTarget.
        :param model_monitoring_access_key: (str) Access key to apply the model monitoring process.

        """
        endpoint_id = endpoint[mm_constants.EventFieldType.UID]
        start_times: set[datetime.datetime] = set()
        try:
            m_fs = fstore.get_feature_set(
                endpoint[mm_constants.EventFieldType.FEATURE_SET_URI]
            )

            for application in applications_names:
                batch_window = batch_window_generator.get_batch_window(
                    project=project,
                    endpoint=endpoint_id,
                    application=application,
                    first_request=endpoint[mm_constants.EventFieldType.FIRST_REQUEST],
                    last_request=endpoint[mm_constants.EventFieldType.LAST_REQUEST],
                    has_stream=endpoint[mm_constants.EventFieldType.STREAM_PATH] != "",
                )

                for start_infer_time, end_infer_time in batch_window.get_intervals():
                    # start - TODO : delete in 1.9.0 (V1 app deprecation)
                    try:
                        # Get application sample data
                        offline_response = cls._get_sample_df(
                            feature_set=m_fs,
                            endpoint_id=endpoint_id,
                            start_infer_time=start_infer_time,
                            end_infer_time=end_infer_time,
                            parquet_directory=parquet_directory,
                            storage_options=storage_options,
                            application_name=application,
                        )

                        df = offline_response.to_dataframe()
                        parquet_target_path = offline_response.vector.get_target_path()

                        if len(df) == 0:
                            logger.info(
                                "During this time window, the endpoint has not received any data",
                                endpoint=endpoint[mm_constants.EventFieldType.UID],
                                start_time=start_infer_time,
                                end_time=end_infer_time,
                            )
                            continue

                    except FileNotFoundError:
                        logger.warn(
                            "No parquets were written yet",
                            endpoint=endpoint[mm_constants.EventFieldType.UID],
                        )
                        continue

                    # Get the timestamp of the latest request:
                    latest_request = df[mm_constants.EventFieldType.TIMESTAMP].iloc[-1]

                    # Get the feature stats from the model endpoint for reference data
                    feature_stats = json.loads(
                        endpoint[mm_constants.EventFieldType.FEATURE_STATS]
                    )

                    # Pad the original feature stats to accommodate current
                    # data out of the original range (unless already padded)
                    pad_features_hist(FeatureStats(feature_stats))

                    # Get the current stats:
                    current_stats = calculate_inputs_statistics(
                        sample_set_statistics=feature_stats, inputs=df
                    )
                    # end - TODO : delete in 1.9.0 (V1 app deprecation)
                    cls._push_to_applications(
                        current_stats=current_stats,
                        feature_stats=feature_stats,
                        start_infer_time=start_infer_time,
                        end_infer_time=end_infer_time,
                        endpoint_id=endpoint_id,
                        latest_request=latest_request,
                        project=project,
                        applications_names=[application],
                        model_monitoring_access_key=model_monitoring_access_key,
                        parquet_target_path=parquet_target_path,
                    )
                    start_times.add(start_infer_time)
        except Exception:
            logger.exception(
                "Encountered an exception",
                endpoint_id=endpoint[mm_constants.EventFieldType.UID],
            )

        if start_times:
            return {endpoint_id: [str(t) for t in sorted(list(start_times))]}

    def _delete_old_parquet(self, endpoints: list[dict[str, Any]], days: int = 1):
        """
        Delete application parquets older than the argument days.

        :param endpoints: A list of dictionaries of model endpoints records.
        """
        if self.parquet_directory.startswith("v3io:///"):
            # create fs with access to the user side (under projects)
            store, _, _ = mlrun.store_manager.get_or_create_store(
                self.parquet_directory,
                {"V3IO_ACCESS_KEY": self.model_monitoring_access_key},
            )
            fs = store.filesystem

            # calculate time threshold (keep only files from the last 24 hours)
            time_to_keep = (
                datetime.datetime.now(tz=datetime.timezone.utc)
                - datetime.timedelta(days=days)
            ).timestamp()

            for endpoint in endpoints:
                try:
                    apps_parquet_directories = fs.listdir(
                        path=f"{self.parquet_directory}"
                        f"/key={endpoint[mm_constants.EventFieldType.UID]}"
                    )
                    for directory in apps_parquet_directories:
                        if directory["mtime"] < time_to_keep:
                            # Delete files
                            fs.rm(path=directory["name"], recursive=True)
                            # Delete directory
                            fs.rmdir(path=directory["name"])
                except FileNotFoundError:
                    logger.info(
                        "Application parquet directory is empty, "
                        "probably parquets have not yet been created for this app",
                        endpoint=endpoint[mm_constants.EventFieldType.UID],
                        path=f"{self.parquet_directory}"
                        f"/key={endpoint[mm_constants.EventFieldType.UID]}",
                    )

    @staticmethod
    def _push_to_applications(
        current_stats,
        feature_stats,
        start_infer_time,
        end_infer_time,
        endpoint_id,
        latest_request,
        project,
        applications_names,
        model_monitoring_access_key,
        parquet_target_path,
    ):
        """
        Pushes data to multiple stream applications.

        :param current_stats:       Current statistics of input data.
        :param feature_stats:       Statistics of train features.
        :param start_infer_time:    The beginning of the infer interval window.
        :param end_infer_time:      The end of the infer interval window.
        :param endpoint_id:         Identifier for the model endpoint.
        :param latest_request:      Timestamp of the latest model request.
        :param project: mlrun       Project name.
        :param applications_names:  List of application names to which data will be pushed.

        """

        data = {
            mm_constants.ApplicationEvent.CURRENT_STATS: json.dumps(current_stats),
            mm_constants.ApplicationEvent.FEATURE_STATS: json.dumps(feature_stats),
            mm_constants.ApplicationEvent.SAMPLE_PARQUET_PATH: parquet_target_path,
            mm_constants.ApplicationEvent.START_INFER_TIME: start_infer_time.isoformat(
                sep=" ", timespec="microseconds"
            ),
            mm_constants.ApplicationEvent.END_INFER_TIME: end_infer_time.isoformat(
                sep=" ", timespec="microseconds"
            ),
            mm_constants.ApplicationEvent.LAST_REQUEST: latest_request.isoformat(
                sep=" ", timespec="microseconds"
            ),
            mm_constants.ApplicationEvent.ENDPOINT_ID: endpoint_id,
            mm_constants.ApplicationEvent.OUTPUT_STREAM_URI: get_stream_path(
                project=project,
                function_name=mm_constants.MonitoringFunctionNames.WRITER,
            ),
            mm_constants.ApplicationEvent.MLRUN_CONTEXT: {},  # TODO : for future use by ad-hoc batch infer
        }
        for app_name in applications_names:
            data.update({mm_constants.ApplicationEvent.APPLICATION_NAME: app_name})
            stream_uri = get_stream_path(project=project, function_name=app_name)

            logger.info(
                f"push endpoint_id {endpoint_id} to {app_name} by stream :{stream_uri}"
            )
            get_stream_pusher(stream_uri, access_key=model_monitoring_access_key).push(
                [data]
            )

    @staticmethod
    def _get_sample_df(
        feature_set: mlrun.common.schemas.FeatureSet,
        endpoint_id: str,
        start_infer_time: datetime.datetime,
        end_infer_time: datetime.datetime,
        parquet_directory: str,
        storage_options: dict,
        application_name: str,
    ) -> mlrun.feature_store.OfflineVectorResponse:
        """
        Retrieves a sample DataFrame of the current input according to the provided infer interval window.

        :param feature_set:         The main feature set.
        :param endpoint_id:         Identifier for the model endpoint.
        :param start_infer_time:    The beginning of the infer interval window.
        :param end_infer_time:      The end of the infer interval window.
        :param parquet_directory:   Directory where Parquet files are stored.
        :param storage_options:     Storage options for accessing the data.
        :param application_name:    Current application name.

        :return: OfflineVectorResponse that can be used for generating a sample DataFrame for the specified endpoint.

        """
        features = [f"{feature_set.metadata.name}.*"]
        vector = fstore.FeatureVector(
            name=f"{endpoint_id}_vector",
            features=features,
            with_indexes=True,
        )
        vector.metadata.tag = application_name
        vector.feature_set_objects = {feature_set.metadata.name: feature_set}

        # get offline features based on application start and end time.
        # store the result parquet by partitioning by controller end processing time
        offline_response = vector.get_offline_features(
            start_time=start_infer_time,
            end_time=end_infer_time,
            timestamp_for_filtering=mm_constants.EventFieldType.TIMESTAMP,
            target=ParquetTarget(
                path=parquet_directory
                + f"/key={endpoint_id}/{int(start_infer_time.timestamp())}/{application_name}.parquet",
                storage_options=storage_options,
            ),
        )
        return offline_response
