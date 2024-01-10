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
from typing import Any, Iterator, Optional, Tuple, Union, cast

from v3io.dataplane.response import HttpResponseError

import mlrun
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.data_types.infer
import mlrun.feature_store as fstore
from mlrun.common.model_monitoring.helpers import FeatureStats, pad_features_hist
from mlrun.datastore import get_stream_pusher
from mlrun.datastore.targets import ParquetTarget
from mlrun.model_monitoring.batch import calculate_inputs_statistics
from mlrun.model_monitoring.helpers import get_monitoring_parquet_path, get_stream_path
from mlrun.utils import logger
from mlrun.utils.v3io_clients import get_v3io_client


class _BatchWindow:
    V3IO_CONTAINER_FORMAT = "users/pipelines/{project}/monitoring-schedules/functions"

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
        self._endpoint = endpoint
        self._application = application
        self._first_request = first_request
        self._kv_storage = get_v3io_client(endpoint=mlrun.mlconf.v3io_api).kv
        self._v3io_container = self.V3IO_CONTAINER_FORMAT.format(project=project)
        self._start = self._get_last_analyzed()
        self._stop = last_updated
        self._step = timedelta_seconds

    def _get_last_analyzed(self) -> Optional[int]:
        try:
            data = self._kv_storage.get(
                container=self._v3io_container,
                table_path=self._endpoint,
                key=self._application,
            )
        except HttpResponseError as err:
            logger.warn(
                "Failed to get the last analyzed time for this endpoint and application, "
                "as this is probably the first time this application is running. "
                "Using the first request time instead.",
                endpoint=self._endpoint,
                application=self._application,
                first_request=self._first_request,
                error=err,
            )
            return self._first_request

        last_analyzed = data.output.item[mm_constants.SchedulingKeys.LAST_ANALYZED]
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
        self._kv_storage.put(
            container=self._v3io_container,
            table_path=self._endpoint,
            key=self._application,
            attributes={mm_constants.SchedulingKeys.LAST_ANALYZED: last_analyzed},
        )

    def get_intervals(
        self,
    ) -> Iterator[Tuple[datetime.datetime, datetime.datetime]]:
        """Generate the batch interval time ranges."""
        if self._start is not None and self._stop is not None:
            entered = False
            for timestamp in range(self._start, self._stop, self._step):
                entered = True
                start_time = datetime.datetime.utcfromtimestamp(timestamp)
                end_time = datetime.datetime.utcfromtimestamp(timestamp + self._step)
                yield start_time, end_time
                self._update_last_analyzed(timestamp + self._step)
            if not entered:
                logger.info(
                    "All the data is set, but no complete intervals were found. "
                    "Wait for last_updated to be updated.",
                    endpoint=self._endpoint,
                    application=self._application,
                    start=self._start,
                    stop=self._stop,
                    step=self._step,
                )
        else:
            logger.warn(
                "The first request time is not not found for this endpoint. "
                "No intervals will be generated.",
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
        """Get the timedelta from a batch dictionary"""
        self._batch_dict = cast(dict[str, int], self._batch_dict)
        minutes, hours, days = (
            self._batch_dict[mm_constants.EventFieldType.MINUTES],
            self._batch_dict[mm_constants.EventFieldType.HOURS],
            self._batch_dict[mm_constants.EventFieldType.DAYS],
        )
        return int(
            datetime.timedelta(minutes=minutes, hours=hours, days=days).total_seconds()
        )

    @classmethod
    def _get_last_updated_time(cls, last_request: Optional[str]) -> Optional[int]:
        """
        Get the last updated time of a model endpoint.
        """
        if not last_request:
            return None
        return int(
            cls._date_string2timestamp(last_request)
            - cast(
                float,
                mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs,
            )
        )

    @classmethod
    def _normalize_first_request(
        cls, first_request: Optional[str], endpoint: str
    ) -> Optional[int]:
        if not first_request:
            logger.warn(
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
            last_updated=self._get_last_updated_time(last_request),
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
        context: mlrun.run.MLClientCtx,
        project: str,
    ):
        """
        Initialize Monitoring Application Processor object.

        :param context:                     An MLRun context.
        :param project:                     Project name.
        """
        self.context = context
        self.project = project

        logger.info(
            "Initializing MonitoringApplicationController",
            project=project,
        )

        # Get a runtime database

        self.db = mlrun.model_monitoring.get_model_endpoint_store(project=project)

        # If an error occurs, it will be raised using the following argument
        self.endpoints_exceptions = {}

        # The batch window
        self._batch_window_generator = _BatchWindowGenerator(
            batch_dict=context.parameters[
                mm_constants.EventFieldType.BATCH_INTERVALS_DICT
            ]
        )

        # If provided, only model endpoints in that that list will be analyzed
        self.model_endpoints = context.parameters.get(
            mm_constants.EventFieldType.MODEL_ENDPOINTS, None
        )
        self.model_monitoring_access_key = self._get_model_monitoring_access_key()
        self.parquet_directory = get_monitoring_parquet_path(
            project=project,
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

    def run(self):
        """
        Main method for run all the relevant monitoring application on each endpoint
        """
        try:
            endpoints = self.db.list_model_endpoints(uids=self.model_endpoints)
            application = mlrun.get_or_create_project(
                self.project
            ).list_model_monitoring_functions()
            if application:
                applications_names = list({app.metadata.name for app in application})
            else:
                logger.info("There are no monitoring application found in this project")
                applications_names = []

        except Exception as e:
            logger.error("Failed to list endpoints", exc=e)
            return
        if endpoints and applications_names:
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
                res = future.result()
                if res:
                    self.endpoints_exceptions[res[0]] = res[1]

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
    ) -> Optional[Tuple[str, Exception]]:
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
        try:
            m_fs = fstore.get_feature_set(
                endpoint[mm_constants.EventFieldType.FEATURE_SET_URI]
            )
            labels = endpoint[mm_constants.EventFieldType.LABEL_NAMES]
            if labels:
                if isinstance(labels, str):
                    labels = json.loads(labels)
                for label in labels:
                    if label not in list(m_fs.spec.features.keys()):
                        m_fs.add_feature(fstore.Feature(name=label, value_type="float"))

            for application in applications_names:
                batch_window = batch_window_generator.get_batch_window(
                    project=project,
                    endpoint=endpoint_id,
                    application=application,
                    first_request=endpoint[mm_constants.EventFieldType.FIRST_REQUEST],
                    last_request=endpoint[mm_constants.EventFieldType.LAST_REQUEST],
                )

                for start_infer_time, end_infer_time in batch_window.get_intervals():
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
                            logger.warn(
                                "Not enough model events since the beginning of the batch interval",
                                featureset_name=m_fs.metadata.name,
                                endpoint=endpoint[mm_constants.EventFieldType.UID],
                                min_required_events=mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events,
                                start_time=start_infer_time,
                                end_time=end_infer_time,
                            )
                            continue

                    # Continue if not enough events provided since the deployment of the model endpoint
                    except FileNotFoundError:
                        logger.warn(
                            "Parquet not found, probably due to not enough model events",
                            endpoint=endpoint[mm_constants.EventFieldType.UID],
                            min_required_events=mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events,
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
                        sample_set_statistics=feature_stats,
                        inputs=df,
                    )

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
        except Exception as e:
            logger.error(
                "Encountered an exception",
                endpoint_id=endpoint[mm_constants.EventFieldType.UID],
            )
            return endpoint_id, e

    def _delete_old_parquet(self, endpoints: list[dict[str, Any]], days: int = 1):
        """
        Delete application parquets older than the argument days.

        :param endpoints: A list of dictionaries of model endpoints records.
        """
        if self.parquet_directory.startswith("v3io:///"):
            # create fs with access to the user side (under projects)
            store, _ = mlrun.store_manager.get_or_create_store(
                self.parquet_directory,
                {"V3IO_ACCESS_KEY": self.model_monitoring_access_key},
            )
            fs = store.get_filesystem()

            # calculate time threshold (keep only files from the last 24 hours)
            time_to_keep = float(
                (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%s")
            )
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
                application_name=mm_constants.MonitoringFunctionNames.WRITER,
            ),
        }
        for app_name in applications_names:
            data.update({mm_constants.ApplicationEvent.APPLICATION_NAME: app_name})
            stream_uri = get_stream_path(project=project, application_name=app_name)

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
        offline_response = fstore.get_offline_features(
            feature_vector=vector,
            start_time=start_infer_time,
            end_time=end_infer_time,
            timestamp_for_filtering=mm_constants.EventFieldType.TIMESTAMP,
            target=ParquetTarget(
                path=parquet_directory
                + f"/key={endpoint_id}/{start_infer_time.strftime('%s')}/{application_name}.parquet",
                storage_options=storage_options,
            ),
        )
        return offline_response
