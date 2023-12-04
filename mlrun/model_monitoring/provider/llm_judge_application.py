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

# This is a wrapper class for the mlrun to interact with the huggingface's evaluate library
import uuid
from typing import Union, List, Optional, Dict, Any
from mlrun.model_monitoring.application import (
    ModelMonitoringApplication,
    ModelMonitoringApplicationResult,
)


class LlmJudgeMonitoringApp(ModelObj):
    kind = "llm_as_judge_monitoring_app"

    def __init__(
        self,
        name: Optional[str] = None,
        metrics: Optional[List[Union[MonitoringMetric]]] = None,
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
    def metrics(self) -> List[MonitoringMetric]:
        """list of all the metrics in current app"""
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: List[Union[MonitoringMetric]]):
        self._metrics = ObjectList.from_list(MonitoringMetric, metrics)

    def compute_metrics_over_data(
        self, train_df: pd.DataFrame, sample_df: pd.DataFrame, metrics_names: List[str]
    ) -> Dict[str, float]:
        """
        Calculate metrics values - helper for the user .
        """
