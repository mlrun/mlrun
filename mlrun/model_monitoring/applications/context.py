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
import socket
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.feature_store as fstore
import mlrun.serving
import mlrun.utils
from mlrun.artifacts import Artifact, DatasetArtifact, ModelArtifact, get_model
from mlrun.common.model_monitoring.helpers import FeatureStats, pad_features_hist
from mlrun.model_monitoring.helpers import (
    calculate_inputs_statistics,
    get_endpoint_record,
)
from mlrun.model_monitoring.model_endpoint import ModelEndpoint


class MonitoringApplicationContext:
    """
    The monitoring context holds all the relevant information for the monitoring application,
    and also it can be used for logging artifacts and results.
    The monitoring context has the following attributes:

    :param application_name:        (str) The model monitoring application name.
    :param project_name:            (str) The project name.
    :param project:                 (MlrunProject) The project object.
    :param logger:                  (nuclio.request.Logger) Nuclio logger.
    :param sample_df_stats:         (FeatureStats) The new sample distribution dictionary.
    :param feature_stats:           (FeatureStats) The train sample distribution dictionary.
    :param sample_df:               (pd.DataFrame) The new sample DataFrame.
    :param start_infer_time:        (pd.Timestamp) Start time of the monitoring schedule.
    :param end_infer_time:          (pd.Timestamp) End time of the monitoring schedule.
    :param latest_request:          (pd.Timestamp) Timestamp of the latest request on this endpoint_id.
    :param endpoint_id:             (str) ID of the monitored model endpoint
    :param output_stream_uri:       (str) URI of the output stream for results
    :param model_endpoint:          (ModelEndpoint) The model endpoint object.
    :param feature_names:           (list[str]) List of models feature names.
    :param label_names:             (list[str]) List of models label names.
    :param model:                   (tuple[str, ModelArtifact, dict]) The model file, model spec object,
                                    and list of of extra data items.
    """

    def __init__(
        self,
        *,
        graph_context: mlrun.serving.GraphContext,
        application_name: str,
        event: dict[str, Any],
        model_endpoint_dict: dict[str, ModelEndpoint],
    ) -> None:
        """
        Initialize a `MonitoringApplicationContext` object.
        Note: this object should not be instantiated manually.

        :param application_name:    The application name.
        :param event:               The instance data dictionary.
        :param model_endpoint_dict: Dictionary of model endpoints.
        """
        self.application_name = application_name

        self.project_name = cast(str, mlrun.mlconf.default_project)
        self.project = mlrun.load_project(url=self.project_name)

        # Nuclio logger - `nuclio.request.Logger`
        self.logger = graph_context.logger

        # event data
        self.start_infer_time = pd.Timestamp(
            cast(str, event.get(mm_constants.ApplicationEvent.START_INFER_TIME))
        )
        self.end_infer_time = pd.Timestamp(
            cast(str, event.get(mm_constants.ApplicationEvent.END_INFER_TIME))
        )
        self.latest_request = pd.Timestamp(
            cast(str, event.get(mm_constants.ApplicationEvent.LAST_REQUEST))
        )
        self.endpoint_id = cast(
            str, event.get(mm_constants.ApplicationEvent.ENDPOINT_ID)
        )
        self.output_stream_uri = cast(
            str, event.get(mm_constants.ApplicationEvent.OUTPUT_STREAM_URI)
        )

        self._feature_stats: Optional[FeatureStats] = json.loads(
            event.get(mm_constants.ApplicationEvent.FEATURE_STATS, "{}")
        )
        self._sample_df_stats: Optional[FeatureStats] = json.loads(
            event.get(mm_constants.ApplicationEvent.CURRENT_STATS, "{}")
        )

        # Default labels for the artifacts
        self._labels = self._get_labels()

        # Persistent data - fetched when needed
        self._sample_df: Optional[pd.DataFrame] = None
        self._model_endpoint: Optional[ModelEndpoint] = model_endpoint_dict.get(
            self.endpoint_id
        )

    def _get_labels(self) -> dict[str, str]:
        return {
            mlrun_constants.MLRunInternalLabels.runner_pod: socket.gethostname(),
            mlrun_constants.MLRunInternalLabels.producer_type: "model-monitoring-app",
            mlrun_constants.MLRunInternalLabels.app_name: self.application_name,
            mlrun_constants.MLRunInternalLabels.endpoint_id: self.endpoint_id,
        }

    @property
    def sample_df(self) -> pd.DataFrame:
        if self._sample_df is None:
            feature_set = fstore.get_feature_set(
                self.model_endpoint.status.monitoring_feature_set_uri
            )
            features = [f"{feature_set.metadata.name}.*"]
            vector = fstore.FeatureVector(
                name=f"{self.endpoint_id}_vector",
                features=features,
                with_indexes=True,
            )
            vector.metadata.tag = self.application_name
            vector.feature_set_objects = {feature_set.metadata.name: feature_set}

            offline_response = vector.get_offline_features(
                start_time=self.start_infer_time,
                end_time=self.end_infer_time,
                timestamp_for_filtering=mm_constants.FeatureSetFeatures.time_stamp(),
            )
            self._sample_df = offline_response.to_dataframe().reset_index(drop=True)
        return self._sample_df

    @property
    def model_endpoint(self) -> ModelEndpoint:
        if not self._model_endpoint:
            self._model_endpoint = ModelEndpoint.from_flat_dict(
                get_endpoint_record(self.project_name, self.endpoint_id)
            )
        return self._model_endpoint

    @property
    def feature_stats(self) -> FeatureStats:
        if not self._feature_stats:
            self._feature_stats = json.loads(self.model_endpoint.status.feature_stats)
            pad_features_hist(self._feature_stats)
        return self._feature_stats

    @property
    def sample_df_stats(self) -> FeatureStats:
        """statistics of the sample dataframe"""
        if not self._sample_df_stats:
            self._sample_df_stats = calculate_inputs_statistics(
                self.feature_stats, self.sample_df
            )
        return self._sample_df_stats

    @property
    def feature_names(self) -> list[str]:
        """The feature names of the model"""
        feature_names = self.model_endpoint.spec.feature_names
        return (
            feature_names
            if isinstance(feature_names, list)
            else json.loads(feature_names)
        )

    @property
    def label_names(self) -> list[str]:
        """The label names of the model"""
        label_names = self.model_endpoint.spec.label_names
        return label_names if isinstance(label_names, list) else json.loads(label_names)

    @property
    def model(self) -> tuple[str, ModelArtifact, dict]:
        """The model file, model spec object, and list of extra data items"""
        return get_model(self.model_endpoint.spec.model_uri)

    @staticmethod
    def dict_to_histogram(histogram_dict: FeatureStats) -> pd.DataFrame:
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

    def log_artifact(
        self,
        item,
        body=None,
        tag: str = "",
        local_path: str = "",
        artifact_path: Optional[str] = None,
        format: Optional[str] = None,
        upload: Optional[bool] = None,
        labels: Optional[dict[str, str]] = None,
        target_path: Optional[str] = None,
        **kwargs,
    ) -> Artifact:
        """
        Log an artifact.
        See :func:`~mlrun.projects.MlrunProject.log_artifact` for the documentation.
        """
        labels = (labels or {}) | self._labels
        return self.project.log_artifact(
            item,
            body=body,
            tag=tag,
            local_path=local_path,
            artifact_path=artifact_path,
            format=format,
            upload=upload,
            labels=labels,
            target_path=target_path,
            **kwargs,
        )

    def log_dataset(
        self,
        key,
        df,
        tag="",
        local_path=None,
        artifact_path=None,
        upload=None,
        labels=None,
        format="",
        preview=None,
        stats=None,
        target_path="",
        extra_data=None,
        label_column: Optional[str] = None,
        **kwargs,
    ) -> DatasetArtifact:
        """
        Log a dataset artifact.
        See :func:`~mlrun.projects.MlrunProject.log_dataset` for the documentation.
        """
        labels = (labels or {}) | self._labels
        return self.project.log_dataset(
            key,
            df,
            tag=tag,
            local_path=local_path,
            artifact_path=artifact_path,
            upload=upload,
            labels=labels,
            format=format,
            preview=preview,
            stats=stats,
            target_path=target_path,
            extra_data=extra_data,
            label_column=label_column,
            **kwargs,
        )
