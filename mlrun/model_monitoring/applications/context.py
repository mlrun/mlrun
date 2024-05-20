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
import typing

import numpy as np
import pandas as pd

import mlrun.common.helpers
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.feature_store as fstore
from mlrun.artifacts.model import ModelArtifact, get_model
from mlrun.common.model_monitoring.helpers import FeatureStats, pad_features_hist
from mlrun.execution import MLClientCtx
from mlrun.model_monitoring.helpers import (
    calculate_inputs_statistics,
    get_endpoint_record,
)
from mlrun.model_monitoring.model_endpoint import ModelEndpoint


class MonitoringApplicationContext(MLClientCtx):
    """
    The monitoring context holds all the relevant information for the monitoring application,
    and also it can be used for logging artifacts and results.
    The monitoring context has the following attributes:

    :param application_name:        (str) the app name
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
    :param model:                   (tuple[str, ModelArtifact, dict]) The model file, model spec object, and list of

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __post_init__(self):
        self.application_name: typing.Optional[str] = None
        self.start_infer_time: typing.Optional[pd.Timestamp] = None
        self.end_infer_time: typing.Optional[pd.Timestamp] = None
        self.latest_request: typing.Optional[pd.Timestamp] = None
        self.endpoint_id: typing.Optional[str] = None
        self.output_stream_uri: typing.Optional[str] = None

        self._sample_df: typing.Optional[pd.DataFrame] = None
        self._model_endpoint: typing.Optional[ModelEndpoint] = None
        self._feature_stats: typing.Optional[FeatureStats] = None
        self._sample_df_stats: typing.Optional[FeatureStats] = None

    @classmethod
    def from_dict(
        cls,
        attrs: dict,
        context=None,
        model_endpoint_dict=None,
        **kwargs,
    ) -> "MonitoringApplicationContext":
        """
        Create an instance of the MonitoringApplicationContext from a dictionary.

        :param attrs:               The instance data dictionary.
        :param context:             The current application context.
        :param model_endpoint_dict: Dictionary of model endpoints.

        """

        if not context:
            self = (
                super().from_dict(
                    attrs=attrs.get(mm_constants.ApplicationEvent.MLRUN_CONTEXT, {}),
                    **kwargs,
                ),
            )
        else:
            self = context
            self.__post_init__()

        self.start_infer_time = pd.Timestamp(
            attrs.get(mm_constants.ApplicationEvent.START_INFER_TIME)
        )
        self.end_infer_time = pd.Timestamp(
            attrs.get(mm_constants.ApplicationEvent.END_INFER_TIME)
        )
        self.latest_request = pd.Timestamp(
            attrs.get(mm_constants.ApplicationEvent.LAST_REQUEST)
        )
        self.application_name = attrs.get(
            mm_constants.ApplicationEvent.APPLICATION_NAME
        )
        self._feature_stats = json.loads(
            attrs.get(mm_constants.ApplicationEvent.FEATURE_STATS, "{}")
        )
        self._sample_df_stats = json.loads(
            attrs.get(mm_constants.ApplicationEvent.CURRENT_STATS, "{}")
        )

        self.endpoint_id = attrs.get(mm_constants.ApplicationEvent.ENDPOINT_ID)
        self._model_endpoint = model_endpoint_dict.get(self.endpoint_id)

        return self

    @property
    def sample_df(self) -> pd.DataFrame:
        if not hasattr(self, "_sample_df") or self._sample_df is None:
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
        if not hasattr(self, "_model_endpoint") or not self._model_endpoint:
            self._model_endpoint = ModelEndpoint.from_flat_dict(
                get_endpoint_record(self.project, self.endpoint_id)
            )
        return self._model_endpoint

    @property
    def feature_stats(self) -> FeatureStats:
        if not hasattr(self, "_feature_stats") or not self._feature_stats:
            self._feature_stats = json.loads(self.model_endpoint.status.feature_stats)
            pad_features_hist(self._feature_stats)
        return self._feature_stats

    @property
    def sample_df_stats(self) -> FeatureStats:
        """statistics of the sample dataframe"""
        if not hasattr(self, "_sample_df_stats") or not self._sample_df_stats:
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
        """return model file, model spec object, and list of extra data items"""
        return get_model(self.model_endpoint.spec.model_uri)

    @staticmethod
    def dict_to_histogram(
        histogram_dict: mlrun.common.model_monitoring.helpers.FeatureStats,
    ) -> pd.DataFrame:
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
