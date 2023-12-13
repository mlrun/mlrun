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
from typing import List, Optional, Union, Dict
from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem
from mlrun.model import ModelObj, ObjectList
from mlrun.utils import logger
from mlrun.model_monitoring.genai.metrics import LLMEvaluateMetric, LLMJudgeBaseMetric


class LLMMonitoringApp(ModelObj):
    kind = "LLM_monitoring_app"

    def __init__(
        self,
        name: Optional[str] = None,
        metrics: Optional[List[Union[LLMEvaluateMetric, LLMJudgeBaseMetric]]] = None,
        possible_drift_threshold: float = None,
        obvious_drift_threshold: float = None,
    ):
        pass

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
        self, train_df: pd.DataFrame, sample_df: pd.DataFrame, metrics_names: List[str]
    ) -> Dict[str, float]:
        """
        Calculate metrics values - helper for the user .
        """

        pass

    def radar_chart(self, metrics_names: List[str], metrics_values: List[float]):
        """
        Create a radar chart for the metrics values.
        """

        pass
