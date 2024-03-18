# Copyright 2024 Iguazio
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

from dataclasses import dataclass
from typing import Final, Optional, Protocol

import numpy as np
from pandas import DataFrame, Timestamp

from mlrun.common.schemas.model_monitoring.constants import (
    MLRUN_HISTOGRAM_DATA_DRIFT_APP_NAME,
    ResultKindApp,
    ResultStatusApp,
)
from mlrun.model_monitoring.application import (
    ModelMonitoringApplicationBase,
    ModelMonitoringApplicationResult,
)
from mlrun.model_monitoring.batch import (
    HellingerDistance,
    HistogramDistanceMetric,
    KullbackLeiblerDivergence,
    TotalVarianceDistance,
)


class InvalidMetricValueError(ValueError):
    pass


class InvalidThresholdValueError(ValueError):
    pass


class ValueClassifier(Protocol):
    def value_to_status(self, value: float) -> ResultStatusApp: ...


@dataclass
class DataDriftClassifier:
    """
    Classify data drift numeric values into categorical status.
    """

    potential: float = 0.5
    detected: float = 0.7

    def __post_init__(self) -> None:
        """Catch erroneous threshold values"""
        if not 0 < self.potential < self.detected < 1:
            raise InvalidThresholdValueError(
                "The provided thresholds do not comply with the rules"
            )

    def value_to_status(self, value: float) -> ResultStatusApp:
        """
        Translate the numeric value into status category.

        :param value: The numeric value of the data drift metric, between 0 and 1.
        :returns:     `ResultStatusApp` according to the classification.
        """
        if value > 1 or value < 0:
            raise InvalidMetricValueError(
                f"{value = } is invalid, must be in the range [0, 1]."
            )
        if value >= self.detected:
            return ResultStatusApp.detected
        if value >= self.potential:
            return ResultStatusApp.potential_detection
        return ResultStatusApp.no_detection


class HistogramDataDriftApplication(ModelMonitoringApplicationBase):
    """
    MLRun's default data drift application for model monitoring.

    The application calculates the metrics over the features' histograms.
    Each metric is calculated over all the features, the mean is taken,
    and the status is returned.
    """

    NAME: Final[str] = MLRUN_HISTOGRAM_DATA_DRIFT_APP_NAME
    METRIC_KIND: Final[ResultKindApp] = ResultKindApp.data_drift

    _REQUIRED_METRICS = {HellingerDistance, TotalVarianceDistance}

    metrics: list[type[HistogramDistanceMetric]] = [
        HellingerDistance,
        KullbackLeiblerDivergence,
        TotalVarianceDistance,
    ]

    def __init__(self, value_classifier: Optional[ValueClassifier] = None) -> None:
        """
        Initialize the data drift application.

        :param value_classifier: Classifier object that adheres to the `ValueClassifier` protocol.
                                 If not provided, the default `DataDriftClassifier()` is used.
        """
        self._value_classifier = value_classifier or DataDriftClassifier()
        assert self._REQUIRED_METRICS <= set(
            self.metrics
        ), "TVD and Hellinger distance are required for the general data drift result"

    def _compute_metrics_per_feature(
        self, sample_df_stats: DataFrame, feature_stats: DataFrame
    ) -> dict[type[HistogramDistanceMetric], list[float]]:
        """Compute the metrics for the different features and labels"""
        metrics_per_feature: dict[type[HistogramDistanceMetric], list[float]] = {
            metric_class: [] for metric_class in self.metrics
        }

        for (sample_feat, sample_hist), (reference_feat, reference_hist) in zip(
            sample_df_stats.items(), feature_stats.items()
        ):
            assert sample_feat == reference_feat, "The features do not match"
            self.context.logger.info(
                "Computing metrics for feature", feature_name=sample_feat
            )
            sample_arr = np.asarray(sample_hist)
            reference_arr = np.asarray(reference_hist)
            for metric in self.metrics:
                metric_name = metric.NAME
                self.context.logger.debug(
                    "Computing data drift metric",
                    metric_name=metric_name,
                    feature_name=sample_feat,
                )
                metrics_per_feature[metric].append(
                    metric(distrib_t=sample_arr, distrib_u=reference_arr).compute()
                )
        self.context.logger.info("Finished computing the metrics")

        return metrics_per_feature

    def _add_general_drift_result(
        self, results: list[ModelMonitoringApplicationResult], value: float
    ) -> None:
        results.append(
            ModelMonitoringApplicationResult(
                name="general_drift",
                value=value,
                kind=self.METRIC_KIND,
                status=self._value_classifier.value_to_status(value),
            )
        )

    def _get_results(
        self, metrics_per_feature: dict[type[HistogramDistanceMetric], list[float]]
    ) -> list[ModelMonitoringApplicationResult]:
        """Average the metrics over the features and add the status"""
        results: list[ModelMonitoringApplicationResult] = []
        hellinger_tvd_values: list[float] = []
        for metric_class, metric_values in metrics_per_feature.items():
            self.context.logger.debug(
                "Averaging metric over the features", metric_name=metric_class.NAME
            )
            value = np.mean(metric_values)
            if metric_class == KullbackLeiblerDivergence:
                # This metric is not bounded from above [0, inf).
                # No status is currently reported for KL divergence
                status = ResultStatusApp.irrelevant
            else:
                status = self._value_classifier.value_to_status(value)
            if metric_class in self._REQUIRED_METRICS:
                hellinger_tvd_values.append(value)
            results.append(
                ModelMonitoringApplicationResult(
                    name=f"{metric_class.NAME}_mean",
                    value=value,
                    kind=self.METRIC_KIND,
                    status=status,
                )
            )

        self._add_general_drift_result(
            results=results, value=np.mean(hellinger_tvd_values)
        )

        return results

    def do_tracking(
        self,
        application_name: str,
        sample_df_stats: DataFrame,
        feature_stats: DataFrame,
        sample_df: DataFrame,
        start_infer_time: Timestamp,
        end_infer_time: Timestamp,
        latest_request: Timestamp,
        endpoint_id: str,
        output_stream_uri: str,
    ) -> list[ModelMonitoringApplicationResult]:
        """
        Calculate and return the data drift metrics, averaged over the features.

        Refer to `ModelMonitoringApplicationBase` for the meaning of the
        function arguments.
        """
        self.context.logger.debug("Starting to run the application")
        metrics_per_feature = self._compute_metrics_per_feature(
            sample_df_stats=sample_df_stats, feature_stats=feature_stats
        )
        self.context.logger.debug("Computing average per metric")
        results = self._get_results(metrics_per_feature)
        self.context.logger.debug("Finished running the application", results=results)
        return results
