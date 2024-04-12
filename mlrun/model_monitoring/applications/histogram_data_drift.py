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
from typing import Final, Optional, Protocol, cast

import numpy as np
from pandas import DataFrame, Series, Timestamp

import mlrun.artifacts
import mlrun.common.model_monitoring.helpers
import mlrun.model_monitoring.features_drift_table as mm_drift_table
from mlrun.common.schemas.model_monitoring.constants import (
    MLRUN_HISTOGRAM_DATA_DRIFT_APP_NAME,
    EventFieldType,
    ResultKindApp,
    ResultStatusApp,
)
from mlrun.model_monitoring.application import (
    ModelMonitoringApplicationBase,
    ModelMonitoringApplicationResult,
)
from mlrun.model_monitoring.metrics.histogram_distance import (
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
    ) -> DataFrame:
        """Compute the metrics for the different features and labels"""
        metrics_per_feature = DataFrame(
            columns=[metric_class.NAME for metric_class in self.metrics]
        )

        for feature_name in feature_stats:
            sample_hist = np.asarray(sample_df_stats[feature_name])
            reference_hist = np.asarray(feature_stats[feature_name])
            self.context.logger.info(
                "Computing metrics for feature", feature_name=feature_name
            )
            metrics_per_feature.loc[feature_name] = {  # pyright: ignore[reportCallIssue,reportArgumentType]
                metric.NAME: metric(
                    distrib_t=sample_hist, distrib_u=reference_hist
                ).compute()
                for metric in self.metrics
            }
        self.context.logger.info("Finished computing the metrics")

        return metrics_per_feature

    def _add_general_drift_result(
        self, results: list[ModelMonitoringApplicationResult], value: float
    ) -> None:
        """Add the general drift result to the results list and log it"""
        status = self._value_classifier.value_to_status(value)
        results.append(
            ModelMonitoringApplicationResult(
                name="general_drift",
                value=value,
                kind=self.METRIC_KIND,
                status=status,
            )
        )

    def _get_results(
        self, metrics_per_feature: DataFrame
    ) -> list[ModelMonitoringApplicationResult]:
        """Average the metrics over the features and add the status"""
        results: list[ModelMonitoringApplicationResult] = []

        self.context.logger.debug("Averaging metrics over the features")
        metrics_mean = metrics_per_feature.mean().to_dict()

        self.context.logger.debug("Creating the results")
        for name, value in metrics_mean.items():
            if name == KullbackLeiblerDivergence.NAME:
                # This metric is not bounded from above [0, inf).
                # No status is currently reported for KL divergence
                status = ResultStatusApp.irrelevant
            else:
                status = self._value_classifier.value_to_status(value)
            results.append(
                ModelMonitoringApplicationResult(
                    name=f"{name}_mean",
                    value=value,
                    kind=self.METRIC_KIND,
                    status=status,
                )
            )

        self._add_general_drift_result(
            results=results,
            value=np.mean(
                [
                    metrics_mean[HellingerDistance.NAME],
                    metrics_mean[TotalVarianceDistance.NAME],
                ]
            ),
        )

        self.context.logger.info("Finished with the results")
        return results

    @staticmethod
    def _remove_timestamp_feature(
        sample_set_statistics: mlrun.common.model_monitoring.helpers.FeatureStats,
    ) -> mlrun.common.model_monitoring.helpers.FeatureStats:
        """
        Drop the 'timestamp' feature if it exists, as it is irrelevant
        in the plotly artifact
        """
        sample_set_statistics = mlrun.common.model_monitoring.helpers.FeatureStats(
            sample_set_statistics.copy()
        )
        if EventFieldType.TIMESTAMP in sample_set_statistics:
            del sample_set_statistics[EventFieldType.TIMESTAMP]
        return sample_set_statistics

    def _log_json_artifact(self, drift_per_feature_values: Series) -> None:
        """Log the drift values as a JSON artifact"""
        self.context.logger.debug("Logging drift value per feature JSON artifact")
        self.context.log_artifact(
            mlrun.artifacts.Artifact(
                body=drift_per_feature_values.to_json(),
                format="json",
                key="features_drift_results",
            )
        )
        self.context.logger.debug("Logged JSON artifact successfully")

    def _log_plotly_table_artifact(
        self,
        sample_set_statistics: mlrun.common.model_monitoring.helpers.FeatureStats,
        inputs_statistics: mlrun.common.model_monitoring.helpers.FeatureStats,
        metrics_per_feature: DataFrame,
        drift_per_feature_values: Series,
    ) -> None:
        """Log the Plotly drift table artifact"""
        self.context.logger.debug(
            "Feature stats",
            sample_set_statistics=sample_set_statistics,
            inputs_statistics=inputs_statistics,
        )

        self.context.logger.debug("Computing drift results per feature")
        drift_results = {
            cast(str, key): (self._value_classifier.value_to_status(value), value)
            for key, value in drift_per_feature_values.items()
        }
        self.context.logger.debug("Logging plotly artifact")
        self.context.log_artifact(
            mm_drift_table.FeaturesDriftTablePlot().produce(
                sample_set_statistics=sample_set_statistics,
                inputs_statistics=inputs_statistics,
                metrics=metrics_per_feature.T.to_dict(),
                drift_results=drift_results,
            )
        )
        self.context.logger.debug("Logged plotly artifact successfully")

    def _log_drift_artifacts(
        self,
        sample_set_statistics: mlrun.common.model_monitoring.helpers.FeatureStats,
        inputs_statistics: mlrun.common.model_monitoring.helpers.FeatureStats,
        metrics_per_feature: DataFrame,
        log_json_artifact: bool = True,
    ) -> None:
        """Log JSON and Plotly drift data per feature artifacts"""
        drift_per_feature_values = metrics_per_feature[
            [HellingerDistance.NAME, TotalVarianceDistance.NAME]
        ].mean(axis=1)

        if log_json_artifact:
            self._log_json_artifact(drift_per_feature_values)

        self._log_plotly_table_artifact(
            sample_set_statistics=self._remove_timestamp_feature(sample_set_statistics),
            inputs_statistics=inputs_statistics,
            metrics_per_feature=metrics_per_feature,
            drift_per_feature_values=drift_per_feature_values,
        )

    def do_tracking(
        self,
        application_name: str,
        sample_df_stats: mlrun.common.model_monitoring.helpers.FeatureStats,
        feature_stats: mlrun.common.model_monitoring.helpers.FeatureStats,
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
            sample_df_stats=self.dict_to_histogram(sample_df_stats),
            feature_stats=self.dict_to_histogram(feature_stats),
        )
        self.context.logger.debug("Saving artifacts")
        self._log_drift_artifacts(
            inputs_statistics=feature_stats,
            sample_set_statistics=sample_df_stats,
            metrics_per_feature=metrics_per_feature,
        )
        self.context.logger.debug("Computing average per metric")
        results = self._get_results(metrics_per_feature)
        self.context.logger.debug("Finished running the application", results=results)
        return results
