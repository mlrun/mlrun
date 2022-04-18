import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import v3io
from mlrun import get_run_db
from mlrun import store_manager
from mlrun.data_types.infer import DFDataInfer, InferOptions
from mlrun.run import MLClientCtx
from mlrun.utils import logger, config
from mlrun.utils.model_monitoring import EndpointType, parse_model_endpoint_store_prefix
from mlrun.utils.v3io_clients import get_v3io_client, get_frames_client
from sklearn.preprocessing import KBinsDiscretizer

TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f%z"


@dataclass
class TotalVarianceDistance:
    """
    Provides a symmetric drift distance between two periods t and u
    Z - vector of random variables
    Pt - Probability distribution over time span t
    """

    distrib_t: np.ndarray
    distrib_u: np.ndarray

    def compute(self) -> float:
        return np.sum(np.abs(self.distrib_t - self.distrib_u)) / 2


@dataclass
class HellingerDistance:
    """
    Hellinger distance is an f divergence measure, similar to the Kullback-Leibler (KL) divergence.
    However, unlike KL Divergence the Hellinger divergence is symmetric and bounded over a probability space.
    """

    distrib_t: np.ndarray
    distrib_u: np.ndarray

    def compute(self) -> float:
        return np.sqrt(
            0.5 * ((np.sqrt(self.distrib_u) - np.sqrt(self.distrib_t)) ** 2).sum()
        )


@dataclass
class KullbackLeiblerDivergence:
    """
    KL Divergence (or relative entropy) is a measure of how one probability distribution differs from another.
    It is an asymmetric measure (thus it's not a metric) and it doesn't satisfy the triangle inequality.
    KL Divergence of 0, indicates two identical distributions.
    """

    distrib_t: np.ndarray
    distrib_u: np.ndarray

    def compute(self, capping=None, kld_scaling=0.0001) -> float:
        t_u = np.sum(
            np.where(
                self.distrib_t != 0,
                (self.distrib_t)
                * np.log(
                    self.distrib_t
                    / np.where(self.distrib_u != 0, self.distrib_u, kld_scaling)
                ),
                0,
            )
        )
        u_t = np.sum(
            np.where(
                self.distrib_u != 0,
                (self.distrib_u)
                * np.log(
                    self.distrib_u
                    / np.where(self.distrib_t != 0, self.distrib_t, kld_scaling)
                ),
                0,
            )
        )
        result = t_u + u_t
        if capping:
            return capping if result == float("inf") else result
        return result


class VirtualDrift:
    def __init__(
        self,
        prediction_col: Optional[str] = None,
        label_col: Optional[str] = None,
        feature_weights: Optional[List[float]] = None,
        inf_capping: Optional[float] = 10,
    ):
        self.prediction_col = prediction_col
        self.label_col = label_col
        self.feature_weights = feature_weights
        self.capping = inf_capping
        self.discretizers: Dict[str, KBinsDiscretizer] = {}
        self.metrics = {
            "tvd": TotalVarianceDistance,
            "hellinger": HellingerDistance,
            "kld": KullbackLeiblerDivergence,
        }

    def dict_to_histogram(self, histogram_dict):
        histograms = {}
        for feature, stats in histogram_dict.items():
            histograms[feature] = stats["hist"][0]

        # Get features value counts
        histograms = pd.concat(
            [
                pd.DataFrame(data=hist, columns=[feature])
                for feature, hist in histograms.items()
            ],
            axis=1,
        )
        # To Distribution
        histograms = histograms / histograms.sum()
        return histograms

    def compute_metrics_over_df(self, base_histogram, latest_histogram):
        drift_measures = {}
        for metric_name, metric in self.metrics.items():
            drift_measures[metric_name] = {
                feature: metric(
                    base_histogram.loc[:, feature], latest_histogram.loc[:, feature]
                ).compute()
                for feature in base_histogram
            }
        return drift_measures

    def compute_drift_from_histograms(self, feature_stats, current_stats):
        # Process histogram dictionaries to Dataframe of the histograms
        # with Feature histogram as cols
        base_histogram = self.dict_to_histogram(feature_stats)
        latest_histogram = self.dict_to_histogram(current_stats)

        # Verify all the features exist between datasets
        base_features = set(base_histogram.columns)
        latest_features = set(latest_histogram.columns)

        features_common = list(base_features.intersection(latest_features))
        feature_difference = list(base_features ^ latest_features)

        if not features_common:
            raise ValueError(
                f"No common features found: {base_features} <> {latest_features}"
            )

        base_histogram = base_histogram.drop(
            feature_difference, axis=1, errors="ignore"
        )
        latest_histogram = latest_histogram.drop(
            feature_difference, axis=1, errors="ignore"
        )

        # Compute the drift per feature
        features_drift_measures = self.compute_metrics_over_df(
            base_histogram.loc[:, features_common],
            latest_histogram.loc[:, features_common],
        )

        # Compute total drift measures for features
        for metric_name in self.metrics.keys():
            feature_values = list(features_drift_measures[metric_name].values())
            features_drift_measures[metric_name]["total_sum"] = np.sum(feature_values)
            features_drift_measures[metric_name]["total_mean"] = np.mean(feature_values)

            # Add weighted mean by given feature weights if provided
            if self.feature_weights:
                features_drift_measures[metric_name]["total_weighted_mean"] = np.dot(
                    feature_values, self.feature_weights
                )

        drift_result = defaultdict(dict)

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

        if self.label_col:
            label_drift_measures = self.compute_metrics_over_df(
                base_histogram.loc[:, self.label_col],
                latest_histogram.loc[:, self.label_col],
            )
            for metric, values in label_drift_measures.items():
                drift_result[self.label_col][metric] = values[metric]

        if self.prediction_col:
            prediction_drift_measures = self.compute_metrics_over_df(
                base_histogram.loc[:, self.prediction_col],
                latest_histogram.loc[:, self.prediction_col],
            )
            for metric, values in prediction_drift_measures.items():
                drift_result[self.prediction_col][metric] = values[metric]

        return drift_result


class BatchProcessor:
    def __init__(
        self,
        context: MLClientCtx,
        project: str,
        model_monitoring_access_key: str,
        v3io_access_key: str,
    ):
        self.context = context
        self.project = project

        self.v3io_access_key = v3io_access_key
        self.model_monitoring_access_key = (
                model_monitoring_access_key or v3io_access_key
        )

        self.virtual_drift = VirtualDrift(inf_capping=10)

        template = config.model_endpoint_monitoring.store_prefixes.default

        kv_path = template.format(project=self.project, kind="endpoints")
        _, self.kv_container, self.kv_path = parse_model_endpoint_store_prefix(kv_path)

        tsdb_path = template.format(project=project, kind="events")
        _, self.tsdb_container, self.tsdb_path = parse_model_endpoint_store_prefix(
            tsdb_path
        )

        stream_path = template.format(project=self.project, kind="log_stream")
        _, self.stream_container, self.stream_path = parse_model_endpoint_store_prefix(
            stream_path
        )

        self.parquet_path = config.model_endpoint_monitoring.store_prefixes.user_space.format(
            project=project, kind="parquet"
        )

        logger.info(
            "Initializing BatchProcessor",
            project=project,
            model_monitoring_access_key_initalized=bool(model_monitoring_access_key),
            v3io_access_key_initialized=bool(v3io_access_key),
            parquet_path=self.parquet_path,
            kv_container=self.kv_container,
            kv_path=self.kv_path,
            tsdb_container=self.tsdb_container,
            tsdb_path=self.tsdb_path,
            stream_container=self.stream_container,
            stream_path=self.stream_path,
        )

        self.default_possible_drift_threshold = (
            config.model_endpoint_monitoring.drift_thresholds.default.possible_drift
        )
        self.default_drift_detected_threshold = (
            config.model_endpoint_monitoring.drift_thresholds.default.drift_detected
        )

        self.db = get_run_db()
        self.v3io = get_v3io_client(access_key=self.v3io_access_key)
        self.frames = get_frames_client(
            address=config.v3io_framesd,
            container=self.tsdb_container,
            token=self.v3io_access_key,
        )
        self.exception = None

    def post_init(self):
        response = self.v3io.create_stream(
            container=self.stream_container,
            path=self.stream_path,
            shard_count=1,
            raise_for_status=v3io.dataplane.RaiseForStatus.never,
            access_key=self.v3io_access_key,
        )

        if not (response.status_code == 400 and "ResourceInUse" in str(response.body)):
            response.raise_for_status([409, 204, 403])

    def run(self):

        try:
            endpoints = self.db.list_model_endpoints(self.project)
        except Exception as e:
            logger.error("Failed to list endpoints", exc=e)
            return

        active_endpoints = set()
        for endpoint in endpoints.endpoints:
            if endpoint.spec.active:
                active_endpoints.add(endpoint.metadata.uid)

        store, sub = store_manager.get_or_create_store(self.parquet_path)
        prefix = self.parquet_path.replace(sub, "")
        fs = store.get_filesystem(silent=False)

        if not fs.exists(sub):
            logger.warn(
                f"{sub} does not exist"
            )
            return

        for endpoint_dir in fs.ls(sub):
            endpoint_id = endpoint_dir["name"].split("=")[-1]
            if endpoint_id not in active_endpoints:
                continue

            try:
                last_year = self.get_last_created_dir(fs, endpoint_dir)
                last_month = self.get_last_created_dir(fs, last_year)
                last_day = self.get_last_created_dir(fs, last_month)
                last_hour = self.get_last_created_dir(fs, last_day)

                full_path = f"{prefix}{last_hour['name']}"

                logger.info(f"Now processing {full_path}")

                endpoint = self.db.get_model_endpoint(
                    project=self.project, endpoint_id=endpoint_id
                )

                if endpoint.status.endpoint_type == EndpointType.ROUTER:
                    # endpoint.status.feature_stats is None
                    logger.info(f"{endpoint_id} is router skipping")
                    continue

                df = pd.read_parquet(full_path)
                timestamp = df["timestamp"].iloc[-1]

                named_features_df = list(df["named_features"])
                named_features_df = pd.DataFrame(named_features_df)

                current_stats = DFDataInfer.get_stats(
                    df=named_features_df, options=InferOptions.Histogram
                )

                drift_result = self.virtual_drift.compute_drift_from_histograms(
                    feature_stats=endpoint.status.feature_stats,
                    current_stats=current_stats,
                )

                logger.info("Drift result", drift_result=drift_result)

                drift_status, drift_measure = self.check_for_drift(
                    drift_result=drift_result, endpoint=endpoint
                )

                logger.info(
                    "Drift status",
                    endpoint_id=endpoint_id,
                    drift_status=drift_status,
                    drift_measure=drift_measure,
                )

                if drift_status == "POSSIBLE_DRIFT" or drift_status == "DRIFT_DETECTED":
                    self.v3io.stream.put_records(
                        container=self.stream_container,
                        stream_path=self.stream_path,
                        records=[
                            {
                                "data": json.dumps(
                                    {
                                        "endpoint_id": endpoint_id,
                                        "drift_status": drift_status,
                                        "drift_measure": drift_measure,
                                        "drift_per_feature": {**drift_result},
                                    }
                                )
                            }
                        ],
                    )

                self.v3io.kv.update(
                    container=self.kv_container,
                    table_path=self.kv_path,
                    key=endpoint_id,
                    attributes={
                        "current_stats": json.dumps(current_stats),
                        "drift_measures": json.dumps(drift_result),
                        "drift_status": drift_status,
                    },
                )

                tsdb_drift_measures = {
                    "endpoint_id": endpoint_id,
                    "timestamp": pd.to_datetime(timestamp, format=TIME_FORMAT),
                    "record_type": "drift_measures",
                    "tvd_mean": drift_result["tvd_mean"],
                    "kld_mean": drift_result["kld_mean"],
                    "hellinger_mean": drift_result["hellinger_mean"],
                }

                self.frames.write(
                    backend="tsdb",
                    table=self.tsdb_path,
                    dfs=pd.DataFrame.from_dict([tsdb_drift_measures]),
                    index_cols=["timestamp", "endpoint_id", "record_type"],
                )

                logger.info(f"Done updating drift measures {full_path}")

            except Exception as e:
                logger.error(f"Exception for endpoint {endpoint_id}")
                self.exception = e

    def check_for_drift(self, drift_result, endpoint):
        tvd_mean = drift_result.get("tvd_mean")
        hellinger_mean = drift_result.get("hellinger_mean")

        drift_mean = 0.0
        if tvd_mean and hellinger_mean:
            drift_mean = (tvd_mean + hellinger_mean) / 2

        monitor_configuration = endpoint.spec.monitor_configuration or {}

        possible_drift = monitor_configuration.get(
            "possible_drift", self.default_possible_drift_threshold
        )
        drift_detected = monitor_configuration.get(
            "possible_drift", self.default_drift_detected_threshold
        )

        drift_status = "NO_DRIFT"
        if drift_mean >= drift_detected:
            drift_status = "DRIFT_DETECTED"
        elif drift_mean >= possible_drift:
            drift_status = "POSSIBLE_DRIFT"

        return drift_status, drift_mean

    @staticmethod
    def get_last_created_dir(fs, endpoint_dir):
        dirs = fs.ls(endpoint_dir["name"])
        last_dir = sorted(dirs, key=lambda k: k["name"].split("=")[-1])[-1]
        return last_dir


def handler(context: MLClientCtx):
    batch_processor = BatchProcessor(
        context=context,
        project=context.project,
        model_monitoring_access_key=os.environ.get("MODEL_MONITORING_ACCESS_KEY"),
        v3io_access_key=os.environ.get("V3IO_ACCESS_KEY"),
    )
    batch_processor.post_init()
    batch_processor.run()
    if batch_processor.exception:
        raise batch_processor.exception
