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

import json
from dataclasses import dataclass
from typing import Final, Optional, Protocol, Union, cast

import numpy as np
from pandas import DataFrame, Series

import mlrun.artifacts
import mlrun.common.model_monitoring.helpers
import mlrun.model_monitoring.applications.context as mm_context
import mlrun.model_monitoring.applications.results as mm_results
import mlrun.model_monitoring.features_drift_table as mm_drift_table
from mlrun.common.schemas.model_monitoring.constants import (
    EventFieldType,
    HistogramDataDriftApplicationConstants,
    ResultKindApp,
    ResultStatusApp,
)
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationBase,
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

    The application expects tabular numerical data, and calculates three metrics over the shared features' histograms.
    The metrics are calculated on features that have reference data from the training dataset. When there is no
    reference data (`feature_stats`), this application send a warning log and does nothing.
    The three metrics are:

    * Hellinger distance.
    * Total variance distance.
    * Kullback-Leibler divergence.

    Each metric is calculated over all the features individually and the mean is taken as the metric value.
    The average of Hellinger and total variance distance is taken as the result.

    The application logs two artifacts:

    * A JSON with the general drift per feature.
    * A plotly table different metrics per feature.

    This application is deployed by default when calling:

    .. code-block:: python

        project.enable_model_monitoring()

    To avoid it, pass `deploy_histogram_data_drift_app=False`.
    """

    NAME: Final[str] = HistogramDataDriftApplicationConstants.NAME

    _REQUIRED_METRICS = {HellingerDistance, TotalVarianceDistance}

    metrics: list[type[HistogramDistanceMetric]] = [
        HellingerDistance,
        KullbackLeiblerDivergence,
        TotalVarianceDistance,
    ]

    def __init__(self, value_classifier: Optional[ValueClassifier] = None) -> None:
        """
        :param value_classifier: Classifier object that adheres to the `ValueClassifier` protocol.
                                 If not provided, the default `DataDriftClassifier()` is used.
        """
        self._value_classifier = value_classifier or DataDriftClassifier()
        assert self._REQUIRED_METRICS <= set(
            self.metrics
        ), "TVD and Hellinger distance are required for the general data drift result"

    def _compute_metrics_per_feature(
        self, monitoring_context: mm_context.MonitoringApplicationContext
    ) -> DataFrame:
        """Compute the metrics for the different features and labels"""
        metrics_per_feature = DataFrame(
            columns=[metric_class.NAME for metric_class in self.metrics]
        )
        feature_stats = monitoring_context.dict_to_histogram(
            monitoring_context.feature_stats
        )
        sample_df_stats = monitoring_context.dict_to_histogram(
            monitoring_context.sample_df_stats
        )
        for feature_name in feature_stats:
            sample_hist = np.asarray(sample_df_stats[feature_name])
            reference_hist = np.asarray(feature_stats[feature_name])
            monitoring_context.logger.info(
                "Computing metrics for feature", feature_name=feature_name
            )
            metrics_per_feature.loc[feature_name] = {  # pyright: ignore[reportCallIssue,reportArgumentType]
                metric.NAME: metric(
                    distrib_t=sample_hist, distrib_u=reference_hist
                ).compute()
                for metric in self.metrics
            }
        monitoring_context.logger.info("Finished computing the metrics")

        return metrics_per_feature

    def _get_general_drift_result(
        self,
        metrics: list[mm_results.ModelMonitoringApplicationMetric],
        monitoring_context: mm_context.MonitoringApplicationContext,
        metrics_per_feature: DataFrame,
    ) -> mm_results.ModelMonitoringApplicationResult:
        """Get the general drift result from the metrics list"""
        value = cast(
            float,
            np.mean(
                [
                    metric.value
                    for metric in metrics
                    if metric.name
                    in [
                        f"{HellingerDistance.NAME}_mean",
                        f"{TotalVarianceDistance.NAME}_mean",
                    ]
                ]
            ),
        )

        status = self._value_classifier.value_to_status(value)
        return mm_results.ModelMonitoringApplicationResult(
            name=HistogramDataDriftApplicationConstants.GENERAL_RESULT_NAME,
            value=value,
            kind=ResultKindApp.data_drift,
            status=status,
            extra_data={
                EventFieldType.CURRENT_STATS: json.dumps(
                    monitoring_context.sample_df_stats
                ),
                EventFieldType.DRIFT_MEASURES: json.dumps(
                    metrics_per_feature.T.to_dict()
                    | {metric.name: metric.value for metric in metrics}
                ),
                EventFieldType.DRIFT_STATUS: status.value,
            },
        )

    @staticmethod
    def _get_metrics(
        metrics_per_feature: DataFrame,
    ) -> list[mm_results.ModelMonitoringApplicationMetric]:
        """Average the metrics over the features and add the status"""
        metrics: list[mm_results.ModelMonitoringApplicationMetric] = []

        metrics_mean = metrics_per_feature.mean().to_dict()

        for name, value in metrics_mean.items():
            metrics.append(
                mm_results.ModelMonitoringApplicationMetric(
                    name=f"{name}_mean",
                    value=value,
                )
            )

        return metrics

    @staticmethod
    def _get_shared_features_sample_stats(
        monitoring_context: mm_context.MonitoringApplicationContext,
    ) -> mlrun.common.model_monitoring.helpers.FeatureStats:
        """
        Filter out features without reference data in `feature_stats`, e.g. `timestamp`.
        """
        return mlrun.common.model_monitoring.helpers.FeatureStats(
            {
                key: monitoring_context.sample_df_stats[key]
                for key in monitoring_context.feature_stats
            }
        )

    @staticmethod
    def _log_json_artifact(
        drift_per_feature_values: Series,
        monitoring_context: mm_context.MonitoringApplicationContext,
    ) -> None:
        """Log the drift values as a JSON artifact"""
        monitoring_context.logger.debug("Logging drift value per feature JSON artifact")
        monitoring_context.log_artifact(
            mlrun.artifacts.Artifact(
                body=drift_per_feature_values.to_json(),
                format="json",
                key="features_drift_results",
            )
        )
        monitoring_context.logger.debug("Logged JSON artifact successfully")

    def _log_plotly_table_artifact(
        self,
        sample_set_statistics: mlrun.common.model_monitoring.helpers.FeatureStats,
        inputs_statistics: mlrun.common.model_monitoring.helpers.FeatureStats,
        metrics_per_feature: DataFrame,
        drift_per_feature_values: Series,
        monitoring_context: mm_context.MonitoringApplicationContext,
    ) -> None:
        """Log the Plotly drift table artifact"""
        monitoring_context.logger.debug(
            "Feature stats",
            sample_set_statistics=sample_set_statistics,
            inputs_statistics=inputs_statistics,
        )

        monitoring_context.logger.debug("Computing drift results per feature")
        drift_results = {
            cast(str, key): (self._value_classifier.value_to_status(value), value)
            for key, value in drift_per_feature_values.items()
        }
        monitoring_context.logger.debug("Logging plotly artifact")
        monitoring_context.log_artifact(
            mm_drift_table.FeaturesDriftTablePlot().produce(
                sample_set_statistics=sample_set_statistics,
                inputs_statistics=inputs_statistics,
                metrics=metrics_per_feature.T.to_dict(),  # pyright: ignore[reportArgumentType]
                drift_results=drift_results,
            )
        )
        monitoring_context.logger.debug("Logged plotly artifact successfully")

    def _log_drift_artifacts(
        self,
        monitoring_context: mm_context.MonitoringApplicationContext,
        metrics_per_feature: DataFrame,
        log_json_artifact: bool = True,
    ) -> None:
        """Log JSON and Plotly drift data per feature artifacts"""
        drift_per_feature_values = metrics_per_feature[
            [HellingerDistance.NAME, TotalVarianceDistance.NAME]
        ].mean(axis=1)

        if log_json_artifact:
            self._log_json_artifact(drift_per_feature_values, monitoring_context)

        self._log_plotly_table_artifact(
            sample_set_statistics=self._get_shared_features_sample_stats(
                monitoring_context
            ),
            inputs_statistics=monitoring_context.feature_stats,
            metrics_per_feature=metrics_per_feature,
            drift_per_feature_values=drift_per_feature_values,
            monitoring_context=monitoring_context,
        )

    def do_tracking(
        self,
        monitoring_context: mm_context.MonitoringApplicationContext,
    ) -> list[
        Union[
            mm_results.ModelMonitoringApplicationResult,
            mm_results.ModelMonitoringApplicationMetric,
        ]
    ]:
        """
        Calculate and return the data drift metrics, averaged over the features.

        Refer to `ModelMonitoringApplicationBaseV2` for the meaning of the
        function arguments.
        """
        monitoring_context.logger.debug("Starting to run the application")
        if not monitoring_context.feature_stats:
            monitoring_context.logger.warning(
                "No feature statistics found, skipping the application. \n"
                "In order to run the application, training set must be provided when logging the model."
            )
            return []
        metrics_per_feature = self._compute_metrics_per_feature(
            monitoring_context=monitoring_context
        )
        monitoring_context.logger.debug("Saving artifacts")
        self._log_drift_artifacts(
            monitoring_context=monitoring_context,
            metrics_per_feature=metrics_per_feature,
        )
        monitoring_context.logger.debug("Computing average per metric")
        metrics = self._get_metrics(metrics_per_feature)
        result = self._get_general_drift_result(
            metrics=metrics,
            monitoring_context=monitoring_context,
            metrics_per_feature=metrics_per_feature,
        )
        metrics_and_result = metrics + [result]
        monitoring_context.logger.debug(
            "Finished running the application", results=metrics_and_result
        )
        return metrics_and_result
