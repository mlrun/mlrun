# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This is the application that can include mutiple metrcs from evaluat and llm_judge
# This class should perform the following:
# 1. decide which metrics to use, if there is no y_true, we can only use
#    metrics from llm_judge for single grading and pairwise grading. otherwise,
#    we can use metrics from evaluate and llm_judge for reference grading.
# 2. calculate the metrics values for the given data.
# 3. create a radar chart for the metrics values. (this shoud be logged as an artifact)
# 4. create a report for the metrics values. (this should be logged as an artifact)
# 5. it's even better if we can offer a UI for this


import pandas as pd
from abc import ABC, abstractmethod
from statistics import mean, median
from typing import List, Optional, Union, Dict
from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem
from mlrun.model import ModelObj, ObjectList
from mlrun.utils import logger
from mlrun.model_monitoring.genai.metrics import LLMEvaluateMetric, LLMJudgeBaseMetric


class MonitoringAppResult:
    def __init__(
        self,
        app_name: str,
        tsdb_record: Dict[str, float],
        drift_result: Tuple[DriftStatus, float, DriftKind] = None,
    ):
        self.app_name = app_name
        self.tsdb_record = tsdb_record.update({"record_type": "drift_measures"})
        self.drift_result = drift_result


class DriftStatus(Enum):
    """
    Enum for the drift status values.
    """

    NO_DRIFT = "NO_DRIFT"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    POSSIBLE_DRIFT = "POSSIBLE_DRIFT"
    NONE = "NONE"


class DriftKind(Enum):
    """
    Enum for the drift kind.
    """

    DATA_DRIFT = "DATA_DRIFT"
    CONCEPT_DRIFT = "CONCEPT_DRIFT"
    MODEL_PERFORMANCE_DRIFT = "MODEL_PERFORMANCE_DRIFT"
    SYSTEM_PERFORMANCE_DRIFT = "SYSTEM_PERFORMANCE_DRIFT"


# A type for representing a drift result, a tuple of the status and the drift mean:
DriftResultType = Tuple[DriftStatus, float, DriftKind]


def aggregate(agg_type):
    def decorator(func):
        def wrapper(*args, **kwargs):
            values = func(*args, **kwargs)

            if not isinstance(values, list) or not all(
                isinstance(x, (int, float)) for x in values
            ):
                raise TypeError("Function must return a list of numbers")

            if agg_type == "mean":
                agg_value = mean(values)
            elif agg_type == "median":
                agg_value = median(values)
            else:
                raise ValueError("Invalid aggregation type")

            return agg_value

        return wrapper

    return decorator


class LLMMonitoringApp(ModelObj, ABC):
    kind = "LLM_monitoring_app"

    def __init__(
        self,
        name: Optional[str] = None,
        metrics: Optional[List[Union[LLMEvaluateMetric, LLMJudgeBaseMetric]]] = None,
        possible_drift_threshold: float = None,
        obvious_drift_threshold: float = None,
    ):
        self.name = name
        self.metrics = ObjectList.from_list(metrics)
        self.possible_drift_threshold = possible_drift_threshold
        self.obvious_drift_threshold = obvious_drift_threshold

    def do(
        self,
        train_histograms_path: str = None,
        sample_histograms_path: str = None,
        train_df_path: str = None,
        sample_df_path: str = None,
        **kwargs,
    ) -> MonitoringAppResult:
        pass

    @property
    def metrics(self) -> List[Union[LLMEvaluateMetric, LLMJudgeBaseMetric]]:
        """list of all the metrics in current app"""
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: List[Union[LLMEvaluateMetric, LLMJudgeBaseMetric]]):
        self._metrics = ObjectList.from_list(metrics)

    def compute_metrics_over_data(
        self, sample_df: pd.DataFrame, train_df: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Calculate metrics values - helper for the user .
        """
        for metric in self.metrics:
            res[metric.name] = metric.compute_over_data(sample_df, train_df)

    @aggregate("mean")
    def compute_one_metric_over_data(
        self,
        metric: Union[LLMEvaluateMetric, LLMJudgeBaseMetric],
        sample_df: pd.DataFrame,
        train_df: pd.DataFrame = None,
    ) -> List[Union[int, float]]:
        """
        Calculate one metric value - helper for the user .
        """
        sample_questions = sample_df["question"].tolist()
        sample_answers = sample_df["answer"].tolist()
        res = []
        for i in range(len(sample_questions)):
            res.append(
                metric.compute_one_metric_over_data(
                    sample_questions[i], sample_answers[i], train_df
                )
            )
        return res

    def build_radar_chart(self, metrics_res: Dict[str, Union[float, int]]):
        """
        Create a radar chart for the metrics values.
        """

        pass

    def build_report(self, metrics_names: List[str], metrics_values: List[float]):
        """
        Create a report for the metrics values.
        """
