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

import dataclasses
import json
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

import mlrun.common.helpers
import mlrun.common.schemas.model_monitoring
import mlrun.utils.v3io_clients
from mlrun.datastore import get_stream_pusher
from mlrun.datastore.targets import ParquetTarget
from mlrun.model_monitoring.helpers import get_stream_path
from mlrun.serving.utils import StepToDict


@dataclasses.dataclass
class ModelMonitoringApplicationResult:
    application_name: str
    endpoint_id: str
    schedule_time: pd.Timestamp
    result_name: str
    result_value: float
    result_kind: mlrun.common.schemas.model_monitoring.constants.ResultKindApp
    result_status: mlrun.common.schemas.model_monitoring.constants.ResultStatusApp
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

    def __int__(self, name):
        self.name = name

    def do(self, event):
        return self.run_application(*self._resolve_event(event))

    def run_application(
        self,
        current_stats: pd.DataFrame,
        feature_stats: pd.DataFrame,
        sample_df: pd.DataFrame,
        schedule_time: pd.Timestamp,
        latest_request: pd.Timestamp,
        endpoint_uid: str,
        output_stream_uri: str,
    ) -> Union[
        ModelMonitoringApplicationResult, List[ModelMonitoringApplicationResult]
    ]:
        """

        :param current_stats:
        :param feature_stats:
        :param sample_df:
        :param schedule_time:
        :param latest_request:
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
        name: str = None,
    ):
        self.project = project
        self.application_name_to_push = application_name_to_push
        self.stream_uri = stream_uri or get_stream_path(
            project=self.project, application_name=self.application_name_to_push
        )
        self.output_stream = None
        self.name = name or "PushToMonitoringWriter"

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
