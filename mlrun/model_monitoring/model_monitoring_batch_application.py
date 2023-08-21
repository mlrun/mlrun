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
import concurrent
import datetime
import json
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

import mlrun
import mlrun.common.helpers
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring
import mlrun.data_types.infer
import mlrun.feature_store as fstore
import mlrun.utils.v3io_clients
from mlrun.datastore import get_stream_pusher
from mlrun.datastore.targets import ParquetTarget
from mlrun.model_monitoring import MODEL_MONITORING_WRITER_FUNCTION_NAME
from mlrun.model_monitoring.helpers import get_monitoring_parquet_path, get_stream_path
from mlrun.utils import logger


def calculate_inputs_statistics(
    sample_set_statistics: dict, inputs: pd.DataFrame
) -> dict:
    """
    Calculate the inputs data statistics for drift monitoring purpose.

    :param sample_set_statistics: The sample set (stored end point's dataset to reference) statistics. The bins of the
                                  histograms of each feature will be used to recalculate the histograms of the inputs.
    :param inputs:                The inputs to calculate their statistics and later on - the drift with respect to the
                                  sample set.

    :returns: The calculated statistics of the inputs data.
    """

    # Use `DFDataInfer` to calculate the statistics over the inputs:
    inputs_statistics = mlrun.data_types.infer.DFDataInfer.get_stats(
        df=inputs,
        options=mlrun.data_types.infer.InferOptions.Histogram,
    )

    # Recalculate the histograms over the bins that are set in the sample-set of the end point:
    for feature in inputs_statistics.keys():
        if feature in sample_set_statistics:
            counts, bins = np.histogram(
                inputs[feature].to_numpy(),
                bins=sample_set_statistics[feature]["hist"][1],
            )
            inputs_statistics[feature]["hist"] = [
                counts.tolist(),
                bins.tolist(),
            ]

    return inputs_statistics


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
            kind=mlrun.common.schemas.model_monitoring.FileTargetKind.BATCH_CONTROLLER_PARQUET,
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
        Main method for manage the drift analysis and write the results into tsdb and KV table.
        """
        # Get model endpoints (each deployed project has at least 1 serving model):

        try:
            endpoints = self.db.list_model_endpoints(uids=self.model_endpoints)
            application = mlrun.get_or_create_project(
                self.project
            ).list_model_monitoring_applications()
            if application:
                applications_names = np.unique(
                    [app.metadata.name for app in application]
                ).tolist()
            else:
                applications_names = []

        except Exception as e:
            logger.error("Failed to list endpoints", exc=e)
            return
        logger.info(
            f"[DAVID] starting for loop with applications_names="
            f"{applications_names} with len endpoint = {len(endpoints)}"
        )
        if len(endpoints):  # TODO add application
            pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=len(endpoints),
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
                    logger.info(f"[DAVID] apply process")
                    future = pool.submit(
                        BatchApplicationProcessor.endpoint_process,
                        endpoint,
                        applications_names,
                        self.batch_dict,
                        self.project,
                        self.parquet_directory,
                        self.storage_options,
                    )
                    futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    self.endpoints_exceptions[res[0]] = res[1]

            self._delete_old_parquet()

    @staticmethod
    def endpoint_process(
        endpoint: dict,
        applications_names: List[str],
        bath_dict: dict,
        project: str,
        parquet_directory: str,
        storage_options: dict,
    ):
        endpoint_id = endpoint[mlrun.common.schemas.model_monitoring.EventFieldType.UID]
        try:
            logger.info("[DAVID] starting application job for endpoint")
            # Getting batch interval start time and end time
            start_time, end_time = BatchApplicationProcessor._get_interval_range(
                bath_dict
            )
            m_fs = fstore.get_feature_set(
                endpoint[
                    mlrun.common.schemas.model_monitoring.EventFieldType.FEATURE_SET_URI
                ]
            )
            fs_name = m_fs.metadata.name
            labels = endpoint[
                mlrun.common.schemas.model_monitoring.EventFieldType.LABEL_NAMES
            ]
            if labels:
                if isinstance(labels, str):
                    labels = json.loads(labels)
                for label in labels:
                    if label not in list(m_fs.spec.features.keys()):
                        m_fs.add_feature(fstore.Feature(name=label, value_type="float"))

            # TODO : add extra feature_sets

            try:
                features = [f"{fs_name}.*"]
                join_graph = fstore.JoinGraph(first_feature_set=fs_name)
                vector = fstore.FeatureVector(
                    name=f"{endpoint_id}_vector",
                    features=features,
                    with_indexes=True,
                    join_graph=join_graph,
                )
                vector.feature_set_objects = {
                    m_fs.metadata.name: m_fs
                }  # to avoid exception when the taf is not latest
                entity_rows = pd.DataFrame(
                    {
                        mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID: [
                            endpoint_id
                        ],
                        "scheduled_time": [end_time],
                    }
                )
                offline_response = fstore.get_offline_features(
                    feature_vector=vector,
                    entity_rows=entity_rows,
                    entity_timestamp_column="scheduled_time",
                    start_time=start_time,
                    end_time=end_time,
                    timestamp_for_filtering=mlrun.common.schemas.model_monitoring.EventFieldType.TIMESTAMP,
                    target=ParquetTarget(
                        path=parquet_directory,
                        time_partitioning_granularity="minute",
                        partition_cols=[
                            mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID,
                        ],
                        storage_options=storage_options,
                    ),
                )
                df = offline_response.to_dataframe()

                if len(df) == 0:
                    logger.warn(
                        "Not enough model events since the beginning of the batch interval",
                        featureset_name=fs_name,
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

            # TODO: The below warn will be removed once the state of the Feature Store target is updated
            #       as expected. In that case, the existence of the file will be checked before trying to get
            #       the offline data from the feature set.
            # Continue if not enough events provided since the deployment of the model endpoint
            except FileNotFoundError:
                logger.warn(
                    "Parquet not found, probably due to not enough model events",
                    # parquet_target=m_fs.status.targets[0].path, TODO:
                    endpoint=endpoint[
                        mlrun.common.schemas.model_monitoring.EventFieldType.UID
                    ],
                    min_rqeuired_events=mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events,
                )
                return

            # Infer feature set stats and schema
            fstore.api._infer_from_static_df(
                df,
                m_fs,
                options=mlrun.data_types.infer.InferOptions.all_stats(),
            )

            # Save feature set to apply changes
            m_fs.save()

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

            data = {
                "current_stats": json.dumps(current_stats),
                "feature_stats": json.dumps(feature_stats),
                "sample_parquet_path": BatchApplicationProcessor._get_parquet_path(
                    parquet_directory=parquet_directory,
                    schedule_time=end_time,
                    endpoint_id=endpoint_id,
                ),
                "schedule_time": end_time.isoformat(sep=" ", timespec="microseconds"),
                "latest_request": latest_request.isoformat(
                    sep=" ", timespec="microseconds"
                ),
                "endpoint_id": mlrun.common.schemas.model_monitoring.EventFieldType.UID,
                "output_stream_uri": get_stream_path(
                    project=project,
                    application_name=MODEL_MONITORING_WRITER_FUNCTION_NAME,
                ),
            }
            for app_name in applications_names:
                stream_uri = get_stream_path(project=project, application_name=app_name)
                logger.info(f"[DAVID] push to {stream_uri}")
                get_stream_pusher(stream_uri).push([data])

            logger.info("[DAVID] Finish application job for endpoint")

        except FileNotFoundError as e:
            logger.error(
                f"Exception for endpoint {endpoint[mlrun.common.schemas.model_monitoring.EventFieldType.UID]}"
            )
            return endpoint_id, e

    @staticmethod
    def _get_interval_range(batch_dict) -> Tuple[datetime.datetime, datetime.datetime]:
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

    @staticmethod
    def _get_parquet_path(
        parquet_directory: str, schedule_time: datetime.datetime, endpoint_id: str
    ):
        schedule_time_str = ""
        for unit, fmt in [
            ("year", "%Y"),
            ("month", "%m"),
            ("day", "%d"),
            ("hour", "%H"),
            ("minute", "%M"),
        ]:
            schedule_time_str += f"{unit}={schedule_time.strftime(fmt)}/"
        endpoint_str = (
            f"{mlrun.common.schemas.model_monitoring.EventFieldType.UID}={endpoint_id}"
        )

        return f"{parquet_directory}/{schedule_time_str}/{endpoint_str}"

    def _delete_old_parquet(self):
        _, schedule_time = BatchApplicationProcessor._get_interval_range(
            self.batch_dict
        )
        threshold_date = schedule_time - datetime.timedelta(days=1)
        threshold_year = threshold_date.year
        threshold_month = threshold_date.month
        threshold_day = threshold_date.day

        base_directory = get_monitoring_parquet_path(
            project=self.project,
            kind=mlrun.common.schemas.model_monitoring.FileTargetKind.BATCH_CONTROLLER_PARQUET,
        )
        target = ParquetTarget(path=base_directory)
        fs = target._get_store().get_filesystem()

        try:
            # List all subdirectories in the base directory
            years_subdirectories = fs.listdir(base_directory)

            for y_subdirectory in years_subdirectories:
                year = int(y_subdirectory["name"].split("/")[-1].split("=")[1])
                if year == threshold_year:
                    month_subdirectories = fs.listdir(y_subdirectory["name"])
                    for m_subdirectory in month_subdirectories:
                        month = int(m_subdirectory["name"].split("/")[-1].split("=")[1])
                        if month == threshold_month:
                            day_subdirectories = fs.listdir(m_subdirectory["name"])
                            for d_subdirectory in day_subdirectories:
                                day = int(
                                    d_subdirectory["name"].split("/")[-1].split("=")[1]
                                )
                                if day == threshold_day - 1:
                                    fs.rm(path=d_subdirectory["name"], recursive=True)
                        elif month == threshold_month - 1 and threshold_day == 1:
                            fs.rm(path=m_subdirectory["name"], recursive=True)
                elif (
                    year == threshold_year - 1
                    and threshold_month == 1
                    and threshold_day == 1
                ):
                    fs.rm(path=y_subdirectory["name"], recursive=True)
        except FileNotFoundError as exc:
            logger.warn(
                f"Batch application process were unsuccessful to remove the old parquets due to {exc}."
            )
