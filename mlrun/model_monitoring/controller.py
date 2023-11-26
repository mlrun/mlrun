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
from typing import Any, Callable, Optional, Tuple, Union, cast

import mlrun
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.data_types.infer
import mlrun.feature_store as fstore
from mlrun.datastore import get_stream_pusher
from mlrun.datastore.targets import ParquetTarget
from mlrun.model_monitoring.batch import calculate_inputs_statistics
from mlrun.model_monitoring.helpers import get_monitoring_parquet_path, get_stream_path
from mlrun.utils import logger


class _BatchWindow:
    def __init__(self, batch_dict: Union[dict, str]) -> None:
        """
        Initialize a batch window object that handles the batch interval time range
        for the monitoring functions.
        """
        self._batch_dict = batch_dict
        self._norm_batch_dict()

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

    def get_interval_range(
        self,
        now_func: Callable[[], datetime.datetime] = datetime.datetime.now,
    ) -> Tuple[datetime.datetime, datetime.datetime]:
        """Getting batch interval time range"""
        self._batch_dict = cast(dict[str, int], self._batch_dict)
        minutes, hours, days = (
            self._batch_dict[mm_constants.EventFieldType.MINUTES],
            self._batch_dict[mm_constants.EventFieldType.HOURS],
            self._batch_dict[mm_constants.EventFieldType.DAYS],
        )
        end_time = now_func() - datetime.timedelta(
            seconds=mlrun.mlconf.model_endpoint_monitoring.parquet_batching_timeout_secs
        )
        start_time = end_time - datetime.timedelta(
            minutes=minutes, hours=hours, days=days
        )
        return start_time, end_time


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
        self._batch_window = _BatchWindow(
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
                        batch_interval_dict=self._batch_window.get_interval_range,
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
        get_interval_range: Callable[[], Tuple[datetime.datetime, datetime.datetime]],
        project: str,
        parquet_directory: str,
        storage_options: dict,
        model_monitoring_access_key: str,
    ):
        """
        Process a model endpoint and trigger the monitoring applications. This function running on different process
        for each endpoint. In addition, this function will generate a parquet file that includes the relevant data
        for a specific time range.

        :param endpoint:                    (dict) Model endpoint record.
        :param applications_names:          (list[str]) List of application names to push results to.
        :param get_interval_range:          (callable) A callable returning the batch interval start and end times.
        :param project:                     (str) Project name.
        :param parquet_directory:           (str) Directory to store application parquet files
        :param storage_options:             (dict) Storage options for writing ParquetTarget.
        :param model_monitoring_access_key: (str) Access key to apply the model monitoring process.

        """
        endpoint_id = endpoint[mm_constants.EventFieldType.UID]
        try:
            # Get the monitoring feature set of the current model endpoint.
            # Will be used later to retrieve the infer data through the feature set target.
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

            # Getting batch interval start time and end time
            # TODO: Once implemented, use the monitoring policy to generate time range for each application
            start_infer_time, end_infer_time = get_interval_range()
            for application in applications_names:
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
                            min_rqeuired_events=mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events,
                            start_time=start_infer_time,
                            end_time=end_infer_time,
                        )
                        return

                # Continue if not enough events provided since the deployment of the model endpoint
                except FileNotFoundError:
                    logger.warn(
                        "Parquet not found, probably due to not enough model events",
                        endpoint=endpoint[mm_constants.EventFieldType.UID],
                        min_rqeuired_events=mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events,
                    )
                    return

                # Get the timestamp of the latest request:
                latest_request = df[mm_constants.EventFieldType.TIMESTAMP].iloc[-1]

                # Get the feature stats from the model endpoint for reference data
                feature_stats = json.loads(
                    endpoint[mm_constants.EventFieldType.FEATURE_STATS]
                )

                # Get the current stats:
                current_stats = calculate_inputs_statistics(
                    sample_set_statistics=feature_stats,
                    inputs=df,
                )

                # create and push data to all applications
                cls._push_to_applications(
                    current_stats=current_stats,
                    feature_stats=feature_stats,
                    start_infer_time=start_infer_time,
                    end_infer_time=end_infer_time,
                    endpoint_id=endpoint_id,
                    latest_request=latest_request,
                    project=project,
                    applications_names=applications_names,
                    model_monitoring_access_key=model_monitoring_access_key,
                    parquet_target_path=parquet_target_path,
                )
        except FileNotFoundError as e:
            logger.error(
                f"Exception for endpoint {endpoint[mm_constants.EventFieldType.UID]}"
            )
            return endpoint_id, e

    def _delete_old_parquet(self, endpoints: List[Dict[str, Any]], days: int = 1):
        """Delete application parquets that are older than 24 hours.
        now_func: Callable[[], datetime.datetime] = datetime.datetime.now,
        batch_dict: dict[str, int],
        # Initialize the dictionary of batch interval ranges
        pattern = "[" + characters_to_remove + "]"
        self.batch_dict = {}


        :param endpoints: List of dictionaries of model endpoints records.
        """
        if self.parquet_directory.startswith("v3io:///"):

            target = mlrun.datastore.targets.BaseStoreTarget(
                path=self.parquet_directory
            )
            store, _ = target._get_store_and_path()
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
                partition_cols=[
                    mm_constants.EventFieldType.ENDPOINT_ID,
                ],
                storage_options=storage_options,
            ),
        )
        return offline_response
