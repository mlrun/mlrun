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
# This is the application can include mutiple metrcs from evaluat and llm_judge
# This class should perform the following:
# 1. calculate the metrics values for the given data.
# 2. create a radar chart for the metrics values. (this shoud be logged as an artifact)
# 3. create a report for the metrics values. (this should be logged as an artifact)
# 4. it's even better if we can offer a UI for this
# TODO: need to figure out a way to compute the nlp metrics (these need the y_true and y_pred)


import pandas as pd
import json
from abc import ABC, abstractmethod
from functools import wraps
from statistics import mean, median
from typing import List, Optional, Union, Dict
from mlrun.artifacts import Artifact
from mlrun.datastore import DataItem
from mlrun.model import ModelObj, ObjectList
from mlrun.utils import logger
from mlrun.model_monitoring.genai.metrics import LLMEvaluateMetric, LLMJudgeBaseMetric
from mlrun.model_monitoring.genai.radar_plot import radar_plot
from mlrun.model_monitoring.application import (
    MonitoringApplication,
    ModelMonitoringApplicationResult,
)


# A decorator for aggregating the metrics values
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


class LLMMonitoringApp(ModelMonitoringApplication):
    kind = "LLM_monitoring_app"

    def __init__(
        self,
        name: Optional[str] = None,
        metrics: Optional[List[Union[LLMEvaluateMetric, LLMJudgeBaseMetric]]] = None,
        possible_drift_threshold: Union[int, float] = None,
        obvious_drift_threshold: Union[int, float] = None,
    ):
        self.name = name
        self.metrics = ObjectList.from_list(metrics)
        self.possible_drift_threshold = possible_drift_threshold
        self.obvious_drift_threshold = obvious_drift_threshold

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
        Compute the metrics values from the given data this will not do any aggregation.

        :param sample_df:   (pd.DataFrame) The new sample DataFrame.
        :param train_df:    (pd.DataFrame) The train sample DataFrame.
        :return:            (Dict[str, Any]) The metrics values, explanation with metrics name as key.
        """
        res = {}
        for metric in self.metrics:
            if isinstance(metric, LLMEvaluateMetric):
                # TODO need to figure out a way to compute the nlp metrics
                # These need the y_true and y_pred
                logger.info(
                    f"metrics {metric.name} is LLMEvaluateMetric type, need y_true to compute the value"
                )
            else:
                res[metric.name] = metric.compute_over_data(sample_df, train_df)
                res[metric.name]["kind"] = metric.kind
        return res

    @aggregate("mean")
    def compute_one_metric_over_data(
        self,
        metric: Union[LLMEvaluateMetric, LLMJudgeBaseMetric],
        sample_df: pd.DataFrame,
        train_df: pd.DataFrame = None,
    ) -> List[Union[int, float]]:
        """
        Calculate one kind of metric from the given data, and aggregate the values.

        :param metric:      (Union[LLMEvaluateMetric, LLMJudgeBaseMetric]) The metric to calculate.
        :param sample_df:   (pd.DataFrame) The new sample DataFrame.
        :param train_df:    (pd.DataFrame) The train sample DataFrame.
        :return:            (List[Union[int, float]]) The aggregated values.
        """
        if isinstance(metric, LLMEvaluateMetric):
            # TODO need to figure out a way to compute the nlp metrics
            # These need the y_true and y_pred
            logger.info(
                f"metrics {metric.name} is LLMEvaluateMetric type, need y_true to compute the value"
            )
        else:
            res_df = metric.compute_over_data(sample_df, train_df)
            score_cols = [col for col in res_df.columns if "score" in col]
            if len(score_cols) == 1:
                return res_df[score_cols[0]].tolist()
            else:
                return res_df["score_of_assistant_a"].tolist()

    def build_radar_chart(self, metrics_res: Dict[str, Any], **kwargs):
        """
        Create a radar chart for the metrics values with the benchmark model values.

        :param metrics_res: (Dict[str, Any]) The metrics values, explanation with metrics name as key.
        :return:            (Artifact) The radar chart artifact.
        """
        data = [{}, {}]
        for key, value in metrics_res.items():
            if value["kind"] == "llm_judge_single_grading":
                logger.info(
                    f"metrics {key} is llm_judge_single_grading type, no benchmark, the reslut is {value}"
                )
            else:
                data[0][key] = value["score_of_assistant_a"]
                data[1][key] = value["score_of_assistant_b"]
        model_names = [
            kwargs.get("model_name", "custom_model"),
            kwargs.get("benchmark_model_name", "benchmark_model"),
        ]
        plot = radar_chart(data, model_names)
        return plot

    def build_report(self, metrics_res: Dict[str, Any]):
        """
        Create a report for the metrics values.

        :param metrics_res: (Dict[str, Any]) The metrics values, explanation with metrics name as key.
        :return:            (Artifact) The report artifact.
        """
        report = {}

        for key, value in metrics_res.items():
            if type(value) == pd.DataFrame:
                report[key] = value.to_dict()
            else:
                report[key] = value

        return report

    def run_application(
        self,
        application_name: str,
        sample_df_stats: pd.DataFrame,
        feature_stats: pd.DataFrame,
        sample_df: pd.DataFrame,
        start_infer_time: pd.Timestamp,
        end_infer_time: pd.Timestamp,
        latest_request: pd.Timestamp,
        endpoint_id: str,
        output_stream_uri: str,
    ) -> Union[
        ModelMonitoringApplicationResult, list[ModelMonitoringApplicationResult]
    ]:
        """
        Implement this method with your custom monitoring logic.

        :param application_name:         (str) the app name
        :param sample_df_stats:         (pd.DataFrame) The new sample distribution DataFrame.
        :param feature_stats:           (pd.DataFrame) The train sample distribution DataFrame.
        :param sample_df:               (pd.DataFrame) The new sample DataFrame.
        :param start_infer_time:        (pd.Timestamp) Start time of the monitoring schedule.
        :param end_infer_time:          (pd.Timestamp) End time of the monitoring schedule.
        :param latest_request:          (pd.Timestamp) Timestamp of the latest request on this endpoint_id.
        :param endpoint_id:             (str) ID of the monitored model endpoint
        :param output_stream_uri:       (str) URI of the output stream for results

        :returns:                       (ModelMonitoringApplicationResult) or
                                        (list[ModelMonitoringApplicationResult]) of the application results.
        """
        # for the open ended questions, it doesn't make sense to send the aggregated result to the output stream
        # instead, we want to send all the questions and answers that are below the threshold with explanation

        raise NotImplementedError
