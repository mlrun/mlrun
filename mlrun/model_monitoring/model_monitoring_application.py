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
import collections
import dataclasses
import datetime
import json
import multiprocessing
import re
from typing import Any, Dict, List, Tuple, Union

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
from mlrun.serving.utils import StepToDict
from mlrun.utils import logger


@dataclasses.dataclass
class ModelMonitoringApplicationResult:

    application_name: str
    endpoint_id: str
    schedule_time: pd.Timestamp
    result_name: str
    result_value: float
    result_kind: mlrun.common.schemas.model_monitoring.constants.ResultKindApp
    result_status: mlrun.common.schemas.model_monitoring.constants.ResultKindApp
    result_extra_data: dict

    def to_dict(self):
        return {
            "application_name": self.application_name,
            "endpoint_id": self.endpoint_id,
            "schedule_time": self.schedule_time.isoformat(
                sep=" ", timespec="microseconds"
            ),
            "result_name": self.result_name,
            "result_value": self.result_value,
            "result_kind": self.result_kind.value,
            "result_status": self.result_status.value,
            "result_extra_data": json.dumps(self.result_extra_data),
        }


class ModelMonitoringApplication(StepToDict):
    kind = "monitoring_application"

    def do(self, event):
        return self.run_application(*self._resolve_event(event))

    def run_application(
        self,
        current_stats: pd.DataFrame,
        feature_stats: pd.DataFrame,
        sample_df: pd.DataFrame,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        endpoint_uid: str,
        output_stream_uri: str,
    ) -> Union[
        ModelMonitoringApplicationResult, List[ModelMonitoringApplicationResult]
    ]:
        """

        :param current_stats:
        :param feature_stats:
        :param sample_df:
        :param start_time:
        :param end_time:
        :param endpoint_uid:
        :param output_stream_uri:

        :returns: List[ModelMonitoringApplicationResult]
        """
        raise NotImplementedError

    @staticmethod
    def _resolve_event(
        event,
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp, str, str
    ]:
        data = event["data"]

        return (
            ModelMonitoringApplication._dict_to_histogram(data["current_stats"]),
            ModelMonitoringApplication._dict_to_histogram(data["feature_stats"]),
            ParquetTarget(path=data["sample_parquet_path"]).as_df(),
            pd.Timestamp(data["schedule_time"]),
            pd.Timestamp(data["latest_request"]),
            data["endpoint_uid"],
            data["output_stream_uri"],
        )

    @staticmethod
    def _dict_to_histogram(histogram_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert histogram dictionary to pandas DataFrame with feature histograms as columns

        :param histogram_dict: Histogram dictionary

        :returns: Histogram dataframe
        """

        # Create a dictionary with feature histograms as values
        histograms = {}
        for feature, stats in histogram_dict.items():
            if "hist" in stats:
                # Normalize to probability distribution of each feature
                histograms[feature] = np.array(stats["hist"][0]) / stats["count"]

        # Convert the dictionary to pandas DataFrame
        histograms = pd.DataFrame(histograms)

        return histograms


class PushToMonitoringWriter(StepToDict):
    kind = "monitoring_application_stream_pusher"

    def __init__(
        self,
        project: str = None,
        application_name_to_push: str = None,
        stream_uri: str = None,
    ):
        self.project = project
        self.application_name_to_push = application_name_to_push
        self.stream_uri = stream_uri or get_stream_path(
            project=self.project, application_name=self.application_name_to_push
        )
        self.output_stream = None

    def do(
        self,
        event: Union[
            ModelMonitoringApplicationResult, List[ModelMonitoringApplicationResult]
        ],
    ):
        self._lazy_init()
        event = event if isinstance(event, List) else [event]
        for result in event:
            self.output_stream.push(result.to_dict())

    def _lazy_init(self):
        if self.output_stream is None:
            self.output_stream = get_stream_pusher(self.stream_uri)


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
        self.exception = None

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
                applications_names = [app.metadata.name for app in application]
            else:
                applications_names = None

        except Exception as e:
            logger.error("Failed to list endpoints", exc=e)
            return
        logger.info(f"[DAVID] starting for loop with applications_names="
                    f"{applications_names} with len endpoint = {len(endpoints)}")
        if True:  # TODO
            pool = concurrent.futures.ProcessPoolExecutor(processes=len(endpoints))
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
                        self.endpoint_process,
                        args=(
                            endpoint,
                            applications_names,
                        ),
                    )
                    futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                future.result()

            self._delete_old_parquet()

    def endpoint_process(self, endpoint: dict, applications_names: List[str]):
        try:
            logger.info("[DAVID] starting application job for endpoint")
            # Getting batch interval start time and end time
            start_time, end_time = self._get_interval_range()
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

            endpoint_id = endpoint[
                mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID
            ]

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
                entity_rows = pd.DataFrame(
                    {
                        mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID: [
                            endpoint_id
                        ],
                        "scheduled_time": [end_time],
                    }
                )
                parquet_directory = get_monitoring_parquet_path(
                    project=self.project,
                    kind=mlrun.common.schemas.model_monitoring.FileTargetKind.BATCH_CONTROLLER_PARQUET,
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
                "sample_parquet_path": self._get_parquet_path(
                    parquet_directory=parquet_directory,
                    schedule_time=end_time,
                    endpoint_id=endpoint_id,
                ),
                "schedule_time": end_time.isoformat(sep=" ", timespec="microseconds"),
                "latest_request": latest_request.isoformat(
                    sep=" ", timespec="microseconds"
                ),
                "endpoint_id": mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID,
                "output_stream_uri": get_stream_path(
                    project=self.project,
                    application_name=MODEL_MONITORING_WRITER_FUNCTION_NAME,
                ),
            }
            for app_name in applications_names:
                stream_uri = get_stream_path(
                    project=self.project, application_name=app_name
                )
                get_stream_pusher(stream_uri).push([data])

            logger.info("[DAVID] Finish application job for endpoint")

        except Exception as e:
            logger.error(
                f"Exception for endpoint {endpoint[mlrun.common.schemas.model_monitoring.EventFieldType.UID]}"
            )
            self.exception = e

    def _get_interval_range(self) -> Tuple[datetime.datetime, datetime.datetime]:
        """Getting batch interval time range"""
        minutes, hours, days = (
            self.batch_dict[
                mlrun.common.schemas.model_monitoring.EventFieldType.MINUTES
            ],
            self.batch_dict[mlrun.common.schemas.model_monitoring.EventFieldType.HOURS],
            self.batch_dict[mlrun.common.schemas.model_monitoring.EventFieldType.DAYS],
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
        endpoint_str = f"{mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID}={endpoint_id}"

        return f"{parquet_directory}/{schedule_time_str}/{endpoint_str}"

    def _delete_old_parquet(self):
        _, schedule_time = self._get_interval_range()
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


def handler(context: mlrun.run.MLClientCtx):
    logger.info("[DAVID] starting application job")
    batch_processor = BatchApplicationProcessor(
        context=context,
        project=context.project,
    )
    batch_processor.run()
    logger.info("[DAVID] Finish application job")
    if batch_processor.exception:
        raise batch_processor.exception
