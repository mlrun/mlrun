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
import abc
import collections
import dataclasses
import datetime
import json
import os
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

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
from mlrun.utils import logger

# A type for representing a drift result, a tuple of the status and the drift mean:
DriftResultType = Tuple[mlrun.common.schemas.model_monitoring.DriftStatus, float]


@dataclasses.dataclass
class HistogramDistanceMetric(abc.ABC):
    """
    An abstract base class for distance metrics between histograms.

    :args distrib_t: array of distribution t (usually the latest dataset distribution)
    :args distrib_u: array of distribution u (usually the sample dataset distribution)

    Each distribution must contain nonnegative floats that sum up to 1.0.
    """

    distrib_t: np.ndarray
    distrib_u: np.ndarray

    NAME: ClassVar[str]

    # noinspection PyMethodOverriding
    def __init_subclass__(cls, *, metric_name: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.NAME = metric_name

    @abc.abstractmethod
    def compute(self) -> float:
        raise NotImplementedError


class TotalVarianceDistance(HistogramDistanceMetric, metric_name="tvd"):
    """
    Provides a symmetric drift distance between two periods t and u
    Z - vector of random variables
    Pt - Probability distribution over time span t
    """

    def compute(self) -> float:
        """
        Calculate Total Variance distance.

        :returns:  Total Variance Distance.
        """
        return np.sum(np.abs(self.distrib_t - self.distrib_u)) / 2


class HellingerDistance(HistogramDistanceMetric, metric_name="hellinger"):
    """
    Hellinger distance is an f divergence measure, similar to the Kullback-Leibler (KL) divergence.
    It used to quantify the difference between two probability distributions.
    However, unlike KL Divergence the Hellinger divergence is symmetric and bounded over a probability space.
    The output range of Hellinger distance is [0,1]. The closer to 0, the more similar the two distributions.
    """

    def compute(self) -> float:
        """
        Calculate Hellinger Distance

        :returns: Hellinger Distance
        """
        return np.sqrt(
            max(
                1 - np.sum(np.sqrt(self.distrib_u * self.distrib_t)),
                0,  # numerical errors may produce small negative numbers, e.g. -1e-16.
                # However, Cauchy-Schwarz inequality assures this number is in the range [0, 1]
            )
        )


class KullbackLeiblerDivergence(HistogramDistanceMetric, metric_name="kld"):
    """
    KL Divergence (or relative entropy) is a measure of how one probability distribution differs from another.
    It is an asymmetric measure (thus it's not a metric) and it doesn't satisfy the triangle inequality.
    KL Divergence of 0, indicates two identical distributions.
    """

    @staticmethod
    def _calc_kl_div(
        actual_dist: np.array, expected_dist: np.array, kld_scaling: float
    ) -> float:
        """Return the assymetric KL divergence"""
        return np.sum(
            np.where(
                actual_dist != 0,
                (actual_dist)
                * np.log(
                    actual_dist
                    / np.where(expected_dist != 0, expected_dist, kld_scaling)
                ),
                0,
            )
        )

    def compute(self, capping: float = None, kld_scaling: float = 1e-4) -> float:
        """
        :param capping:      A bounded value for the KL Divergence. For infinite distance, the result is replaced with
                             the capping value which indicates a huge differences between the distributions.
        :param kld_scaling:  Will be used to replace 0 values for executing the logarithmic operation.

        :returns: symmetric KL Divergence
        """
        t_u = self._calc_kl_div(self.distrib_t, self.distrib_u, kld_scaling)
        u_t = self._calc_kl_div(self.distrib_u, self.distrib_t, kld_scaling)
        result = t_u + u_t
        if capping:
            return capping if result == float("inf") else result
        return result


class VirtualDrift:
    """
    Virtual Drift object is used for handling the drift calculations.
    It contains the metrics objects and the related methods for the detection of potential drift.
    """

    def __init__(
        self,
        prediction_col: Optional[str] = None,
        label_col: Optional[str] = None,
        feature_weights: Optional[List[float]] = None,
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
        self.metrics: Dict[str, Type[HistogramDistanceMetric]] = {
            metric_class.NAME: metric_class
            for metric_class in (
                TotalVarianceDistance,
                HellingerDistance,
                KullbackLeiblerDivergence,
            )
        }

    @staticmethod
    def dict_to_histogram(histogram_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
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
        base_histogram: Dict[str, Dict[str, Any]],
        latest_histogram: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
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
        feature_stats: Dict[str, Dict[str, Any]],
        current_stats: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
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
        metrics_results_dictionary: Dict[str, Union[float, dict]],
        possible_drift_threshold: float = 0.5,
        drift_detected_threshold: float = 0.7,
    ) -> Dict[str, DriftResultType]:
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
                    "Can't calculate drift for this feature because at least one of the required"
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
        metrics_results_dictionary: Dict[str, Union[float, dict]],
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
        elif "hist" in inputs_statistics[feature]:
            # Comply with the other common features' histogram length
            mlrun.common.model_monitoring.helpers.pad_hist(
                mlrun.common.model_monitoring.helpers.Histogram(
                    inputs_statistics[feature]["hist"]
                )
            )

    return inputs_statistics


class BatchProcessor:
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

        # Initialize virtual drift object
        self.virtual_drift = VirtualDrift(inf_capping=10)

        logger.info(
            "Initializing BatchProcessor",
            project=project,
        )

        # Get drift thresholds from the model monitoring configuration
        # fmt: off
        self.default_possible_drift_threshold = (
            mlrun.mlconf.model_endpoint_monitoring.drift_thresholds.default.possible_drift
        )
        self.default_drift_detected_threshold = (
            mlrun.mlconf.model_endpoint_monitoring.drift_thresholds.default.drift_detected
        )
        # fmt: on

        # Get a runtime database

        self.db = mlrun.model_monitoring.get_model_endpoint_store(project=project)

        if not mlrun.mlconf.is_ce_mode():
            # TODO: Once there is a time series DB alternative in a non-CE deployment, we need to update this if
            #  statement to be applied only for V3IO TSDB
            self._initialize_v3io_configurations()

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

    def _initialize_v3io_configurations(self):
        self.v3io_access_key = os.environ.get("V3IO_ACCESS_KEY")
        self.model_monitoring_access_key = (
            os.environ.get("MODEL_MONITORING_ACCESS_KEY") or self.v3io_access_key
        )

        # Define the required paths for the project objects
        tsdb_path = mlrun.mlconf.get_model_monitoring_file_target_path(
            project=self.project,
            kind=mlrun.common.schemas.model_monitoring.FileTargetKind.EVENTS,
        )
        (
            _,
            self.tsdb_container,
            self.tsdb_path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            tsdb_path
        )
        # stream_path = template.format(project=self.project, kind="log_stream")
        stream_path = mlrun.mlconf.get_model_monitoring_file_target_path(
            project=self.project,
            kind=mlrun.common.schemas.model_monitoring.FileTargetKind.LOG_STREAM,
        )
        (
            _,
            self.stream_container,
            self.stream_path,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            stream_path
        )

        # Get the frames clients based on the v3io configuration
        # it will be used later for writing the results into the tsdb
        self.v3io = mlrun.utils.v3io_clients.get_v3io_client(
            access_key=self.v3io_access_key
        )
        self.frames = mlrun.utils.v3io_clients.get_frames_client(
            address=mlrun.mlconf.v3io_framesd,
            container=self.tsdb_container,
            token=self.v3io_access_key,
        )
        logger.info(
            "Creating table in TSDB if it does not already exist", table=self.tsdb_path
        )
        self.frames.create(
            backend="tsdb",
            table=self.tsdb_path,
            if_exists=IGNORE,
            rate="1/s",
        )

    def post_init(self):
        """
        Preprocess of the batch processing.
        """

        if not mlrun.mlconf.is_ce_mode():
            # Create v3io stream based on the input stream
            response = self.v3io.stream.create(
                container=self.stream_container,
                stream_path=self.stream_path,
                shard_count=1,
                raise_for_status=v3io.dataplane.RaiseForStatus.never,
                access_key=self.v3io_access_key,
            )

            if not (
                response.status_code == 400 and "ResourceInUse" in str(response.body)
            ):
                response.raise_for_status([409, 204, 403])
        pass

    def run(self):
        """
        Main method for manage the drift analysis and write the results into tsdb and KV table.
        """
        # Get model endpoints (each deployed project has at least 1 serving model):

        try:
            endpoints = self.db.list_model_endpoints(uids=self.model_endpoints)

        except Exception as e:
            logger.error("Failed to list endpoints", exc=e)
            return

        for endpoint in endpoints:
            if (
                endpoint[mlrun.common.schemas.model_monitoring.EventFieldType.ACTIVE]
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
                self.update_drift_metrics(endpoint=endpoint)

    def update_drift_metrics(self, endpoint: dict):
        try:
            m_fs = fstore.get_feature_set(
                endpoint[
                    mlrun.common.schemas.model_monitoring.EventFieldType.FEATURE_SET_URI
                ]
            )

            # Getting batch interval start time and end time
            start_time, end_time = self._get_interval_range()

            try:
                df = m_fs.to_dataframe(
                    start_time=start_time,
                    end_time=end_time,
                    time_column=mlrun.common.schemas.model_monitoring.EventFieldType.TIMESTAMP,
                )

                if len(df) == 0:
                    logger.warn(
                        "Not enough model events since the beginning of the batch interval",
                        parquet_target=m_fs.status.targets[0].path,
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
                    parquet_target=m_fs.status.targets[0].path,
                    endpoint=endpoint[
                        mlrun.common.schemas.model_monitoring.EventFieldType.UID
                    ],
                    min_rqeuired_events=mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events,
                )
                return

            # Get feature names from monitoring feature set
            feature_names = [
                feature_name["name"] for feature_name in m_fs.spec.features.to_dict()
            ]

            # Create DataFrame based on the input features
            stats_columns = [
                mlrun.common.schemas.model_monitoring.EventFieldType.TIMESTAMP,
                *feature_names,
            ]

            # Add label names if provided
            if endpoint[
                mlrun.common.schemas.model_monitoring.EventFieldType.LABEL_NAMES
            ]:
                labels = endpoint[
                    mlrun.common.schemas.model_monitoring.EventFieldType.LABEL_NAMES
                ]
                if isinstance(labels, str):
                    labels = json.loads(labels)
                for label in labels:
                    if label not in stats_columns:
                        stats_columns.append(label)
            named_features_df = df[stats_columns].copy()

            # Infer feature set stats and schema
            fstore.api._infer_from_static_df(
                named_features_df,
                m_fs,
                options=mlrun.data_types.infer.InferOptions.all_stats(),
            )

            # Save feature set to apply changes
            m_fs.save()

            # Get the timestamp of the latest request:
            timestamp = df[
                mlrun.common.schemas.model_monitoring.EventFieldType.TIMESTAMP
            ].iloc[-1]

            # Get the feature stats from the model endpoint for reference data
            feature_stats = json.loads(
                endpoint[
                    mlrun.common.schemas.model_monitoring.EventFieldType.FEATURE_STATS
                ]
            )
            # Pad the original feature stats to accommodate current data out
            # of the original range (unless already padded)
            mlrun.common.model_monitoring.helpers.pad_features_hist(
                mlrun.common.model_monitoring.helpers.FeatureStats(feature_stats)
            )

            # Get the current stats:
            current_stats = calculate_inputs_statistics(
                sample_set_statistics=feature_stats,
                inputs=named_features_df,
            )

            # Compute the drift based on the histogram of the current stats and the histogram of the original
            # feature stats that can be found in the model endpoint object:
            drift_result = self.virtual_drift.compute_drift_from_histograms(
                feature_stats=feature_stats,
                current_stats=current_stats,
            )
            logger.info("Drift result", drift_result=drift_result)

            # Get drift thresholds from the model configuration:
            monitor_configuration = (
                json.loads(
                    endpoint[
                        mlrun.common.schemas.model_monitoring.EventFieldType.MONITOR_CONFIGURATION
                    ]
                )
                or {}
            )

            # For backwards compatibility first check if the old drift thresholds
            # (both `possible drift and `drift_detected`) keys exist in the monitor configuration dict
            # TODO: Remove the first get in 1.7.0
            possible_drift = monitor_configuration.get(
                "possible_drift",
                monitor_configuration.get(
                    mlrun.common.schemas.model_monitoring.EventFieldType.POSSIBLE_DRIFT_THRESHOLD,
                    self.default_possible_drift_threshold,
                ),
            )

            drift_detected = monitor_configuration.get(
                "drift_detected",
                monitor_configuration.get(
                    mlrun.common.schemas.model_monitoring.EventFieldType.DRIFT_DETECTED_THRESHOLD,
                    self.default_drift_detected_threshold,
                ),
            )

            # Check for possible drift based on the results of the statistical metrics defined above:
            drift_status, drift_measure = self.virtual_drift.check_for_drift(
                metrics_results_dictionary=drift_result,
                possible_drift_threshold=possible_drift,
                drift_detected_threshold=drift_detected,
            )
            logger.info(
                "Drift status",
                endpoint_id=endpoint[
                    mlrun.common.schemas.model_monitoring.EventFieldType.UID
                ],
                drift_status=drift_status.value,
                drift_measure=drift_measure,
            )

            attributes = {
                "current_stats": json.dumps(current_stats),
                "drift_measures": json.dumps(drift_result),
                "drift_status": drift_status.value,
            }

            self.db.update_model_endpoint(
                endpoint_id=endpoint[
                    mlrun.common.schemas.model_monitoring.EventFieldType.UID
                ],
                attributes=attributes,
            )

            if not mlrun.mlconf.is_ce_mode():
                # Generate V3IO KV schema if not exist
                self._infer_kv_schema()

                # Update drift results in TSDB
                self._update_drift_in_v3io_tsdb(
                    endpoint_id=endpoint[
                        mlrun.common.schemas.model_monitoring.EventFieldType.UID
                    ],
                    drift_status=drift_status,
                    drift_measure=drift_measure,
                    drift_result=drift_result,
                    timestamp=timestamp,
                )

            else:
                # Update drift results in Prometheus
                self._update_drift_in_prometheus(
                    endpoint_id=endpoint[
                        mlrun.common.schemas.model_monitoring.EventFieldType.UID
                    ],
                    drift_status=drift_status,
                    drift_result=drift_result,
                )

        except Exception as e:
            logger.error(
                f"Exception for endpoint {endpoint[mlrun.common.schemas.model_monitoring.EventFieldType.UID]}"
            )
            self.exception = e
        logger.info(
            "Done updating drift measures",
            endpoint_id=endpoint[
                mlrun.common.schemas.model_monitoring.EventFieldType.UID
            ],
        )

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

    def _update_drift_in_v3io_tsdb(
        self,
        endpoint_id: str,
        drift_status: mlrun.common.schemas.model_monitoring.DriftStatus,
        drift_measure: float,
        drift_result: Dict[str, Dict[str, Any]],
        timestamp: pd.Timestamp,
    ):
        """Update drift results in input stream.

        :param endpoint_id:   The unique id of the model endpoint.
        :param drift_status:  Drift status result. Possible values can be found under DriftStatus enum class.
        :param drift_measure: The drift result (float) based on the mean of the Total Variance Distance and the
                              Hellinger distance.
        :param drift_result:  A dictionary that includes the drift results for each feature.
        :param timestamp:     Pandas Timestamp value.

        """

        if (
            drift_status
            == mlrun.common.schemas.model_monitoring.DriftStatus.POSSIBLE_DRIFT
            or drift_status
            == mlrun.common.schemas.model_monitoring.DriftStatus.DRIFT_DETECTED
        ):
            self.v3io.stream.put_records(
                container=self.stream_container,
                stream_path=self.stream_path,
                records=[
                    {
                        "data": json.dumps(
                            {
                                "endpoint_id": endpoint_id,
                                "drift_status": drift_status.value,
                                "drift_measure": drift_measure,
                                "drift_per_feature": {**drift_result},
                            }
                        )
                    }
                ],
            )

        # Update the results in tsdb:
        tsdb_drift_measures = {
            "endpoint_id": endpoint_id,
            "timestamp": timestamp,
            "record_type": "drift_measures",
            "tvd_mean": drift_result["tvd_mean"],
            "kld_mean": drift_result["kld_mean"],
            "hellinger_mean": drift_result["hellinger_mean"],
        }

        try:
            self.frames.write(
                backend="tsdb",
                table=self.tsdb_path,
                dfs=pd.DataFrame.from_records([tsdb_drift_measures]),
                index_cols=["timestamp", "endpoint_id", "record_type"],
            )
        except v3io_frames.errors.Error as err:
            logger.warn(
                "Could not write drift measures to TSDB",
                err=err,
                tsdb_path=self.tsdb_path,
                endpoint=endpoint_id,
            )

    def _update_drift_in_prometheus(
        self,
        endpoint_id: str,
        drift_status: mlrun.common.schemas.model_monitoring.DriftStatus,
        drift_result: Dict[str, Dict[str, Any]],
    ):
        """Push drift metrics to Prometheus registry. Please note that the metrics are being pushed through HTTP
        to the monitoring stream pod that writes them into a local registry. Afterwards, Prometheus wil scrape these
        metrics that will be available in the Grafana charts.

        :param endpoint_id:   The unique id of the model endpoint.
        :param drift_status:  Drift status result. Possible values can be found under DriftStatus enum class.
        :param drift_result:  A dictionary that includes the drift results for each feature.


        """
        stream_http_path = (
            mlrun.mlconf.model_endpoint_monitoring.default_http_sink.format(
                project=self.project
            )
        )

        http_session = mlrun.utils.HTTPSessionWithRetry(
            retry_on_post=True,
            verbose=True,
            max_retries=1,
        )
        try:
            # Model monitoring stream http health check
            http_session.request("GET", url=stream_http_path)

            # Update statistical metrics
            statistical_metrics = ["hellinger_mean", "tvd_mean", "kld_mean"]
            metrics = []
            for metric in statistical_metrics:
                metrics.append(
                    {
                        mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID: endpoint_id,
                        mlrun.common.schemas.model_monitoring.EventFieldType.METRIC: metric,
                        mlrun.common.schemas.model_monitoring.EventFieldType.VALUE: drift_result[
                            metric
                        ],
                    }
                )

            http_session.request(
                method="POST",
                url=stream_http_path + "/monitoring-batch-metrics",
                data=json.dumps(metrics),
            )

            # Update drift status
            drift_status_dict = {
                mlrun.common.schemas.model_monitoring.EventFieldType.ENDPOINT_ID: endpoint_id,
                mlrun.common.schemas.model_monitoring.EventFieldType.DRIFT_STATUS: drift_status.value,
            }

            http_session.request(
                method="POST",
                url=stream_http_path + "/monitoring-drift-status",
                data=json.dumps(drift_status_dict),
            )

        except requests.exceptions.ConnectionError as exc:
            logger.warning(
                "Can't push metrics to Prometheus registry."
                "Monitoring stream pod is not found, probably not deployed. "
                "To deploy, call set_tracking() on a serving function. exc: ",
                exc=exc,
            )

    def _infer_kv_schema(self):
        """
        Create KV schema file if not exist. This schema is being used by the Grafana dashboards.
        """

        schema_file = self.db.client.kv.new_cursor(
            container=self.db.container,
            table_path=self.db.path,
            filter_expression='__name==".#schema"',
        )

        if not schema_file.all():
            logger.info(
                "Generate a new V3IO KV schema file", kv_table_path=self.db.path
            )
            self.frames.execute(
                backend="kv", table=self.db.path, command="infer_schema"
            )


def handler(context: mlrun.run.MLClientCtx):
    batch_processor = BatchProcessor(
        context=context,
        project=context.project,
    )
    batch_processor.post_init()
    batch_processor.run()
    if batch_processor.exception:
        raise batch_processor.exception
