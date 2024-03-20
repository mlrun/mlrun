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

import collections
import datetime
import json
import os
import re
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import requests
import v3io
import v3io.dataplane
import v3io_frames
from v3io_frames.frames_pb2 import IGNORE

import mlrun.common.helpers
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring
import mlrun.data_types.infer
import mlrun.feature_store as fstore
import mlrun.utils.v3io_clients
from mlrun.model_monitoring.helpers import calculate_inputs_statistics
from mlrun.model_monitoring.metrics.histogram_distance import (
    HellingerDistance,
    HistogramDistanceMetric,
    KullbackLeiblerDivergence,
    TotalVarianceDistance,
)
from mlrun.utils import logger

# A type for representing a drift result, a tuple of the status and the drift mean:
DriftResultType = tuple[mlrun.common.schemas.model_monitoring.DriftStatus, float]


class VirtualDrift:
    """
    Virtual Drift object is used for handling the drift calculations.
    It contains the metrics objects and the related methods for the detection of potential drift.
    """

    def __init__(
        self,
        prediction_col: Optional[str] = None,
        label_col: Optional[str] = None,
        feature_weights: Optional[list[float]] = None,
        inf_capping: Optional[float] = 10,
    ):
        """
        Initialize a Virtual Drift object.

        :param prediction_col:  The name of the dataframe column which represents the predictions of the model. If
                                provided, it will be used for calculating drift over the predictions. The name of the
                                dataframe column which represents the labels of the model. If provided, it will be used
                                for calculating drift over the labels.
        :param feature_weights: Weights that can be applied to the features and to be considered during the drift
                                analysis.
        :param inf_capping:     A bounded value for the results of the statistical metric. For example, when calculating
                                KL divergence and getting infinite distance between the two distributions, the result
                                will be replaced with the capping value.
        """
        self.prediction_col = prediction_col
        self.label_col = label_col
        self.feature_weights = feature_weights
        self.capping = inf_capping

        # Initialize objects of the current metrics
        self.metrics: dict[str, type[HistogramDistanceMetric]] = {
            metric_class.NAME: metric_class
            for metric_class in (
                TotalVarianceDistance,
                HellingerDistance,
                KullbackLeiblerDivergence,
            )
        }

    @staticmethod
    def dict_to_histogram(histogram_dict: dict[str, dict[str, Any]]) -> pd.DataFrame:
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

    def compute_metrics_over_df(
        self,
        base_histogram: dict[str, dict[str, Any]],
        latest_histogram: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """
        Calculate metrics values for each feature.

        For example:
        {tvd: {feature_1: 0.001, feature_2: 0.2: ,...}}

        :param base_histogram:   histogram dataframe that represents the distribution of the features from the original
                                 training set.
        :param latest_histogram: Histogram dataframe that represents the distribution of the features from the latest
                                 input batch.

        :returns: A dictionary in which for each metric (key) we assign the values for each feature.
        """

        # compute the different metrics for each feature distribution and store the results in dictionary
        drift_measures = {}
        for metric_name, metric in self.metrics.items():
            drift_measures[metric_name] = {
                feature: metric(
                    base_histogram.loc[:, feature], latest_histogram.loc[:, feature]
                ).compute()
                for feature in base_histogram
            }

        return drift_measures

    def compute_drift_from_histograms(
        self,
        feature_stats: dict[str, dict[str, Any]],
        current_stats: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """
        Compare the distributions of both the original features data and the latest input data
        :param feature_stats: Histogram dictionary of the original feature dataset that was used in the model training.
        :param current_stats: Histogram dictionary of the recent input data

        :returns: A dictionary that includes the drift results for each feature.

        """

        # convert histogram dictionaries to DataFrame of the histograms
        # with feature histogram as cols
        base_histogram = self.dict_to_histogram(feature_stats)
        latest_histogram = self.dict_to_histogram(current_stats)

        # verify all the features exist between datasets
        base_features = set(base_histogram.columns)
        latest_features = set(latest_histogram.columns)
        features_common = list(base_features.intersection(latest_features))
        feature_difference = list(base_features ^ latest_features)
        if not features_common:
            raise ValueError(
                f"No common features found: {base_features} <> {latest_features}"
            )

        # drop columns of non-exist features
        base_histogram = base_histogram.drop(
            feature_difference, axis=1, errors="ignore"
        )
        latest_histogram = latest_histogram.drop(
            feature_difference, axis=1, errors="ignore"
        )

        # compute the statistical metrics per feature
        features_drift_measures = self.compute_metrics_over_df(
            base_histogram.loc[:, features_common],
            latest_histogram.loc[:, features_common],
        )

        # compute total value for each metric
        for metric_name in self.metrics.keys():
            feature_values = list(features_drift_measures[metric_name].values())
            features_drift_measures[metric_name]["total_sum"] = np.sum(feature_values)
            features_drift_measures[metric_name]["total_mean"] = np.mean(feature_values)

            # add weighted mean by given feature weights if provided
            if self.feature_weights:
                features_drift_measures[metric_name]["total_weighted_mean"] = np.dot(
                    feature_values, self.feature_weights
                )

        # define drift result dictionary with values as a dictionary
        drift_result = collections.defaultdict(dict)

        # fill drift result dictionary with the statistical metrics results per feature
        # and the total sum and mean of each metric
        for feature in features_common:
            for metric, values in features_drift_measures.items():
                drift_result[feature][metric] = values[feature]
                sum = features_drift_measures[metric]["total_sum"]
                mean = features_drift_measures[metric]["total_mean"]
                drift_result[f"{metric}_sum"] = sum
                drift_result[f"{metric}_mean"] = mean
                if self.feature_weights:
                    metric_measure = features_drift_measures[metric]
                    weighted_mean = metric_measure["total_weighted_mean"]
                    drift_result[f"{metric}_weighted_mean"] = weighted_mean

        # compute the drift metric over the labels
        if self.label_col:
            label_drift_measures = self.compute_metrics_over_df(
                base_histogram.loc[:, self.label_col],
                latest_histogram.loc[:, self.label_col],
            )
            for metric, values in label_drift_measures.items():
                drift_result[self.label_col][metric] = values[metric]

        # compute the drift metric over the predictions
        if self.prediction_col:
            prediction_drift_measures = self.compute_metrics_over_df(
                base_histogram.loc[:, self.prediction_col],
                latest_histogram.loc[:, self.prediction_col],
            )
            for metric, values in prediction_drift_measures.items():
                drift_result[self.prediction_col][metric] = values[metric]

        return drift_result

    @staticmethod
    def check_for_drift_per_feature(
        metrics_results_dictionary: dict[str, Union[float, dict]],
        possible_drift_threshold: float = 0.5,
        drift_detected_threshold: float = 0.7,
    ) -> dict[str, DriftResultType]:
        """
        Check for drift based on the defined decision rule and the calculated results of the statistical metrics per
        feature.

        :param metrics_results_dictionary: Dictionary of statistical metrics results per feature and the total means of
                                           all features.
        :param possible_drift_threshold:   Threshold for the calculated result to be in a possible drift status.
                                           Default: 0.5.
        :param drift_detected_threshold:   Threshold for the calculated result to be in a drift detected status.
                                           Default: 0.7.

        :returns: A dictionary of all the features and their drift status and results tuples, tuple of:
                  [0] = Drift status enum based on the thresholds given.
                  [1] = The drift result (float) based on the mean of the Total Variance Distance and the Hellinger
                        distance.
        """
        # Initialize the drift results dictionary:
        drift_results = {}

        # Calculate the result per feature:
        for feature, results in metrics_results_dictionary.items():
            # A feature result must be a dictionary, otherwise it's the total mean (float):
            if not isinstance(results, dict):
                continue
            # Calculate the feature's drift mean:
            tvd = results[TotalVarianceDistance.NAME]
            hellinger = results[HellingerDistance.NAME]
            if tvd is None or hellinger is None:
                logger.warning(
                    "Can't calculate drift for this feature because at least one of the required "
                    "statistical metrics is missing",
                    feature=feature,
                    tvd=tvd,
                    hellinger=hellinger,
                )
                continue
            metrics_results_dictionary = (tvd + hellinger) / 2
            # Decision rule for drift detection:
            drift_status = VirtualDrift._get_drift_status(
                drift_result=metrics_results_dictionary,
                possible_drift_threshold=possible_drift_threshold,
                drift_detected_threshold=drift_detected_threshold,
            )
            # Collect the drift result:
            drift_results[feature] = (drift_status, metrics_results_dictionary)

        return drift_results

    @staticmethod
    def check_for_drift(
        metrics_results_dictionary: dict[str, Union[float, dict]],
        possible_drift_threshold: float = 0.5,
        drift_detected_threshold: float = 0.7,
    ) -> DriftResultType:
        """
        Check for drift based on the defined decision rule and the calculated results of the statistical metrics by the
        mean of all features.

        :param metrics_results_dictionary: Dictionary of statistical metrics results per feature and the total means of
                                           all features.
        :param possible_drift_threshold:   Threshold for the calculated result to be in a possible drift status.
                                           Default: 0.5.
        :param drift_detected_threshold:   Threshold for the calculated result to be in a drift detected status.
                                           Default: 0.7.

        :returns: A tuple of:
                  [0] = Drift status enum based on the thresholds given.
                  [1] = The drift result (float) based on the mean of the Total Variance Distance and the Hellinger
                        distance.
        """
        # Calculate the mean drift result:
        tvd_mean = metrics_results_dictionary[f"{TotalVarianceDistance.NAME}_mean"]
        hellinger_mean = metrics_results_dictionary.get(
            f"{HellingerDistance.NAME}_mean"
        )
        drift_result = 0.0
        if tvd_mean and hellinger_mean:
            drift_result = (tvd_mean + hellinger_mean) / 2

        # Decision rule for drift detection:
        drift_status = VirtualDrift._get_drift_status(
            drift_result=drift_result,
            possible_drift_threshold=possible_drift_threshold,
            drift_detected_threshold=drift_detected_threshold,
        )

        return drift_status, drift_result

    @staticmethod
    def _get_drift_status(
        drift_result: float,
        possible_drift_threshold: float,
        drift_detected_threshold: float,
    ) -> mlrun.common.schemas.model_monitoring.DriftStatus:
        """
        Get the drift status according to the result and thresholds given.

        :param drift_result:             The drift result.
        :param possible_drift_threshold: Threshold for the calculated result to be in a possible drift status.
        :param drift_detected_threshold: Threshold for the calculated result to be in a drift detected status.

        :returns: The figured drift status.
        """
        drift_status = mlrun.common.schemas.model_monitoring.DriftStatus.NO_DRIFT
        if drift_result >= drift_detected_threshold:
            drift_status = (
                mlrun.common.schemas.model_monitoring.DriftStatus.DRIFT_DETECTED
            )
        elif drift_result >= possible_drift_threshold:
            drift_status = (
                mlrun.common.schemas.model_monitoring.DriftStatus.POSSIBLE_DRIFT
            )

        return drift_status
