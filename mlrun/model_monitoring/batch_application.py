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
#
import concurrent.futures
import datetime
import json
import os
import re
import typing
from typing import List, Tuple

import numpy as np

import mlrun
import mlrun.common.helpers
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.data_types.infer
import mlrun.feature_store as fstore
import mlrun.utils.v3io_clients
from mlrun.datastore import get_stream_pusher
from mlrun.datastore.targets import ParquetTarget
from mlrun.model_monitoring.batch import calculate_inputs_statistics
from mlrun.model_monitoring.helpers import get_monitoring_parquet_path, get_stream_path
from mlrun.utils import logger


class BatchApplicationProcessor:
    """
    The main object to handle the batch processing job. This object is used to get the required configurations and
    to manage the main monitoring drift detection process based on the current batch.
    Note that the BatchProcessor object requires access keys along with valid project configurations.
    """

    def __init__(
        self,
        context: mlrun.run.MLClientCtx,
        project: str,
    ):
        """
        Initialize Batch Processor object.

        :param context:                     An MLRun context.
        :param project:                     Project name.
        """
        self.context = context
        self.project = project

        logger.info(
            "Initializing BatchProcessor",
            project=project,
        )

        # Get a runtime database

        self.db = mlrun.model_monitoring.get_model_endpoint_store(project=project)

        # If an error occurs, it will be raised using the following argument
        self.endpoints_exceptions = {}

        # Get the batch interval range
        self.batch_dict = context.parameters[
            mlrun.common.schemas.model_monitoring.EventFieldType.BATCH_INTERVALS_DICT
        ]

        # TODO: This will be removed in 1.5.0 once the job params can be parsed with different types
        # Convert batch dict string into a dictionary
        if isinstance(self.batch_dict, str):
            self._parse_batch_dict_str()
        # If provided, only model endpoints in that that list will be analyzed
        self.model_endpoints = context.parameters.get(
            mlrun.common.schemas.model_monitoring.EventFieldType.MODEL_ENDPOINTS, None
        )
        self.v3io_access_key = os.environ.get("V3IO_ACCESS_KEY")
        self.model_monitoring_access_key = (
            os.environ.get("MODEL_MONITORING_ACCESS_KEY") or self.v3io_access_key
        )
        self.parquet_directory = get_monitoring_parquet_path(
            project=project,
            kind=mlrun.common.schemas.model_monitoring.FileTargetKind.APPS_PARQUET,
        )
        self.storage_options = None
        if not mlrun.mlconf.is_ce_mode():
            self._initialize_v3io_configurations(
                model_monitoring_access_key=self.model_monitoring_access_key
            )
        elif self.parquet_directory.startswith("s3://"):
            self.storage_options = mlrun.mlconf.get_s3_storage_options()

    def _initialize_v3io_configurations(
        self,
        v3io_access_key: str = None,
        v3io_framesd: str = None,
        v3io_api: str = None,
        model_monitoring_access_key: str = None,
    ):
        # Get the V3IO configurations
        self.v3io_framesd = v3io_framesd or mlrun.mlconf.v3io_framesd
        self.v3io_api = v3io_api or mlrun.mlconf.v3io_api

        self.v3io_access_key = v3io_access_key or os.environ.get("V3IO_ACCESS_KEY")
        self.model_monitoring_access_key = model_monitoring_access_key
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
                applications_names = np.unique(
                    [app.metadata.name for app in application]
                ).tolist()
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
                    endpoint[
                        mlrun.common.schemas.model_monitoring.EventFieldType.ACTIVE
                    ]
                    and endpoint[
                        mlrun.common.schemas.model_monitoring.EventFieldType.MONITORING_MODE
                    ]
                    == mlrun.common.schemas.model_monitoring.ModelMonitoringMode.enabled.value
                ):
                    # Skip router endpoint:
                    if (
                        int(
                            endpoint[
                                mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_TYPE
                            ]
                        )
                        == mlrun.common.schemas.model_monitoring.EndpointType.ROUTER
                    ):
                        # Router endpoint has no feature stats
                        logger.info(
                            f"{endpoint[mlrun.common.schemas.model_monitoring.EventFieldType.UID]} is router skipping"
                        )
                        continue
                    future = pool.submit(
                        BatchApplicationProcessor.model_endpoint_process,
                        endpoint=endpoint,
                        applications_names=applications_names,
                        batch_interval_dict=self.batch_dict,
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

    @staticmethod
    def model_endpoint_process(
        endpoint: dict,
        applications_names: List[str],
        batch_interval_dict: dict,
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
        :param applications_names:          (List[str]) List of application names to push results to.
        :param batch_interval_dict:         (dict) Batch interval start and end times.
        :param project:                     (str) Project name.
        :param parquet_directory:           (str) Directory to store application parquet files
        :param storage_options:             (dict) Storage options for writing ParquetTarget.
        :param model_monitoring_access_key: (str) Access key to apply the model monitoring process.

        """
        endpoint_id = endpoint[mlrun.common.schemas.model_monitoring.EventFieldType.UID]
        try:
            # Get the monitoring feature set of the current model endpoint.
            # Will be used later to retrieve the infer data through the feature set target.
            m_fs = fstore.get_feature_set(
                endpoint[
                    mlrun.common.schemas.model_monitoring.EventFieldType.FEATURE_SET_URI
                ]
            )
            labels = endpoint[
                mlrun.common.schemas.model_monitoring.EventFieldType.LABEL_NAMES
            ]
            if labels:
                if isinstance(labels, str):
                    labels = json.loads(labels)
                for label in labels:
                    if label not in list(m_fs.spec.features.keys()):
                        m_fs.add_feature(fstore.Feature(name=label, value_type="float"))

            # Getting batch interval start time and end time
            # TODO: Once implemented, use the monitoring policy to generate time range for each application
            start_time, end_time = BatchApplicationProcessor._get_interval_range(
                batch_interval_dict
            )
            for application in applications_names:
                try:
                    # Get application sample data
                    offline_response = BatchApplicationProcessor._get_sample_df(
                        feature_set=m_fs,
                        endpoint_id=endpoint_id,
                        end_time=end_time,
                        start_time=start_time,
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
                            endpoint=endpoint[
                                mlrun.common.schemas.model_monitoring.EventFieldType.UID
                            ],
                            min_rqeuired_events=mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events,
                            start_time=str(
                                datetime.datetime.now() - datetime.timedelta(hours=1)
                            ),
                            end_time=str(datetime.datetime.now()),
                        )
                        return

                # Continue if not enough events provided since the deployment of the model endpoint
                except FileNotFoundError:
                    logger.warn(
                        "Parquet not found, probably due to not enough model events",
                        # parquet_target=m_fs.status.targets[0].path,
                        endpoint=endpoint[
                            mlrun.common.schemas.model_monitoring.EventFieldType.UID
                        ],
                        min_rqeuired_events=mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events,
                    )
                    return

                # Get the timestamp of the latest request:
                latest_request = df[
                    mlrun.common.schemas.model_monitoring.EventFieldType.TIMESTAMP
                ].iloc[-1]

                # Get the feature stats from the model endpoint for reference data
                feature_stats = json.loads(
                    endpoint[
                        mlrun.common.schemas.model_monitoring.EventFieldType.FEATURE_STATS
                    ]
                )

                # Get the current stats:
                current_stats = calculate_inputs_statistics(
                    sample_set_statistics=feature_stats,
                    inputs=df,
                )

                # create and push data to all applications
                BatchApplicationProcessor._push_to_applications(
                    current_stats=current_stats,
                    feature_stats=feature_stats,
                    start_time=start_time,
                    end_time=end_time,
                    endpoint_id=endpoint_id,
                    latest_request=latest_request,
                    project=project,
                    applications_names=applications_names,
                    model_monitoring_access_key=model_monitoring_access_key,
                    parquet_target_path=parquet_target_path,
                )
        except FileNotFoundError as e:
            logger.error(
                f"Exception for endpoint {endpoint[mlrun.common.schemas.model_monitoring.EventFieldType.UID]}"
            )
            return endpoint_id, e

    @staticmethod
    def _get_interval_range(
        batch_dict: dict[str, int]
    ) -> Tuple[datetime.datetime, datetime.datetime]:
        """Getting batch interval time range"""
        minutes, hours, days = (
            batch_dict[mlrun.common.schemas.model_monitoring.EventFieldType.MINUTES],
            batch_dict[mlrun.common.schemas.model_monitoring.EventFieldType.HOURS],
            batch_dict[mlrun.common.schemas.model_monitoring.EventFieldType.DAYS],
        )
        start_time = datetime.datetime.now() - datetime.timedelta(
            minutes=minutes, hours=hours, days=days
        )
        end_time = datetime.datetime.now()
        return start_time, end_time

    def _parse_batch_dict_str(self):
        """Convert batch dictionary string into a valid dictionary"""
        characters_to_remove = "{} "
        pattern = "[" + characters_to_remove + "]"
        # Remove unnecessary characters from the provided string
        batch_list = re.sub(pattern, "", self.batch_dict).split(",")
        # Initialize the dictionary of batch interval ranges
        self.batch_dict = {}
        for pair in batch_list:
            pair_list = pair.split(":")
            self.batch_dict[pair_list[0]] = float(pair_list[1])

    def _delete_old_parquet(
        self, endpoints: typing.List[typing.Dict[str, typing.Any]], days: int = 1
    ):
        """Delete application parquets that are older than 24 hours.

        :param endpoints: List of dictionaries of model endpoints records.
        """

        target = ParquetTarget(path=self.parquet_directory)
        fs = target._get_store().get_filesystem()

        # calculate time threshold (keep only files from the last 24 hours)
        time_to_keep = float(
            (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%s")
        )
        for endpoint in endpoints:
            parquet_files = fs.listdir(
                path=f"{self.parquet_directory}"
                f"/key={endpoint[mlrun.common.schemas.model_monitoring.EventFieldType.UID]}"
            )
            for file in parquet_files:
                if file["mktime"] < time_to_keep:
                    # Delete files
                    fs.rm(path=file["name"], recursive=True)
                    # Delete directory
                    fs.rmdir(path=file["name"])

    @staticmethod
    def _push_to_applications(
        current_stats,
        feature_stats,
        start_time,
        end_time,
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
        :param start_time:          Start time of the monitoring schedule.
        :param end_time:            End time of the monitoring schedule.
        :param endpoint_id:         Identifier for the model endpoint.
        :param latest_request:      Timestamp of the latest model request.
        :param project: mlrun       Project name.
        :param applications_names:  List of application names to which data will be pushed.

        """

        data = {
            mm_constants.ApplicationEvent.CURRENT_STATS: json.dumps(current_stats),
            mm_constants.ApplicationEvent.FEATURE_STATS: json.dumps(feature_stats),
            mm_constants.ApplicationEvent.SAMPLE_PARQUET_PATH: parquet_target_path,
            mm_constants.ApplicationEvent.START_PROCESSING_TIME: start_time.isoformat(
                sep=" ", timespec="microseconds"
            ),
            mm_constants.ApplicationEvent.END_PROCESSING_TIME: end_time.isoformat(
                sep=" ", timespec="microseconds"
            ),
            mm_constants.ApplicationEvent.LAST_REQUEST: latest_request.isoformat(
                sep=" ", timespec="microseconds"
            ),
            mm_constants.ApplicationEvent.ENDPOINT_ID: endpoint_id,
            mm_constants.ApplicationEvent.OUTPUT_STREAM_URI: get_stream_path(
                project=project,
                application_name=mlrun.common.schemas.model_monitoring.constants.MonitoringFunctionNames.WRITER,
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
        end_time: datetime.datetime,
        start_time: datetime.datetime,
        parquet_directory: str,
        storage_options: dict,
        application_name: str,
    ) -> mlrun.feature_store.OfflineVectorResponse:
        """
        Retrieves a sample DataFrame of the current input.

        :param feature_set:         The main feature set.
        :param endpoint_id:         Identifier for the model endpoint.
        :param end_time:            End time of the monitoring schedule.
        :param start_time:          Start time of the monitoring schedule.
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
            start_time=start_time,
            end_time=end_time,
            timestamp_for_filtering=mlrun.common.schemas.model_monitoring.EventFieldType.TIMESTAMP,
            target=ParquetTarget(
                path=parquet_directory
                + f"/key={endpoint_id}/processing_time={end_time}/{application_name}.parquet",
                partition_cols=[
                    mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID,
                ],
                storage_options=storage_options,
            ),
        )
        return offline_response
