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

import socket
from typing import Any, Optional, Protocol, cast

import nuclio.request
import numpy as np
import pandas as pd

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.errors
import mlrun.feature_store as fstore
import mlrun.feature_store.feature_set as fs
import mlrun.features
import mlrun.serving
import mlrun.utils
from mlrun.artifacts import Artifact, DatasetArtifact, ModelArtifact, get_model
from mlrun.common.model_monitoring.helpers import FeatureStats
from mlrun.common.schemas import ModelEndpoint
from mlrun.model_monitoring.helpers import (
    calculate_inputs_statistics,
)


class _ArtifactsLogger(Protocol):
    """
    Classes that implement this protocol are :code:`MlrunProject` and :code:`MLClientCtx`.
    """

    def log_artifact(self, *args, **kwargs) -> Artifact: ...
    def log_dataset(self, *args, **kwargs) -> DatasetArtifact: ...


class MonitoringApplicationContext:
    _logger_name = "monitoring-application"

    def __init__(
        self,
        *,
        application_name: str,
        event: dict[str, Any],
        project: "mlrun.MlrunProject",
        artifacts_logger: _ArtifactsLogger,
        logger: mlrun.utils.Logger,
        nuclio_logger: nuclio.request.Logger,
        model_endpoint_dict: Optional[dict[str, ModelEndpoint]] = None,
        sample_df: Optional[pd.DataFrame] = None,
        feature_stats: Optional[FeatureStats] = None,
        feature_sets_dict: Optional[dict[str, fs.FeatureSet]] = None,
    ) -> None:
        """
        The :code:`MonitoringApplicationContext` object holds all the relevant information for the
        model monitoring application, and can be used for logging artifacts and messages.
        The monitoring context has the following attributes:

        :param application_name:        (str) The model monitoring application name.
        :param project:                 (:py:class:`~mlrun.projects.MlrunProject`) The current MLRun project object.
        :param project_name:            (str) The project name.
        :param logger:                  (:py:class:`~mlrun.utils.Logger`) MLRun logger.
        :param nuclio_logger:           (nuclio.request.Logger) Nuclio logger.
        :param sample_df_stats:         (FeatureStats) The new sample distribution dictionary.
        :param feature_stats:           (FeatureStats) The train sample distribution dictionary.
        :param sample_df:               (pd.DataFrame) The new sample DataFrame.
        :param start_infer_time:        (pd.Timestamp) Start time of the monitoring schedule.
        :param end_infer_time:          (pd.Timestamp) End time of the monitoring schedule.
        :param endpoint_id:             (str) ID of the monitored model endpoint
        :param feature_set:              (FeatureSet) the model endpoint feature set
        :param endpoint_name:           (str) Name of the monitored model endpoint
        :param output_stream_uri:       (str) URI of the output stream for results
        :param model_endpoint:          (ModelEndpoint) The model endpoint object.
        :param feature_names:           (list[str]) List of models feature names.
        :param label_names:             (list[str]) List of models label names.
        :param model:                   (tuple[str, ModelArtifact, dict]) The model file, model spec object,
                                        and a list of extra data items.
        """
        self.application_name = application_name

        self.project = project
        self.project_name = project.name

        self._artifacts_logger = artifacts_logger

        # MLRun Logger
        self.logger = logger
        # Nuclio logger - `nuclio.request.Logger`.
        # Note: this logger accepts keyword arguments only in its `_with` methods, e.g. `info_with`.
        self.nuclio_logger = nuclio_logger

        # event data
        self.start_infer_time = pd.Timestamp(
            cast(str, event.get(mm_constants.ApplicationEvent.START_INFER_TIME))
        )
        self.end_infer_time = pd.Timestamp(
            cast(str, event.get(mm_constants.ApplicationEvent.END_INFER_TIME))
        )
        self.endpoint_id = cast(
            str, event.get(mm_constants.ApplicationEvent.ENDPOINT_ID)
        )
        self.endpoint_name = cast(
            str, event.get(mm_constants.ApplicationEvent.ENDPOINT_NAME)
        )

        self._feature_stats: Optional[FeatureStats] = feature_stats
        self._sample_df_stats: Optional[FeatureStats] = None

        # Default labels for the artifacts
        self._default_labels = self._get_default_labels()

        # Persistent data - fetched when needed
        self._sample_df: Optional[pd.DataFrame] = sample_df
        self._model_endpoint: Optional[ModelEndpoint] = (
            model_endpoint_dict.get(self.endpoint_id) if model_endpoint_dict else None
        )
        self._feature_set: Optional[fs.FeatureSet] = (
            feature_sets_dict.get(self.endpoint_id) if feature_sets_dict else None
        )
        store, _, _ = mlrun.store_manager.get_or_create_store(
            mlrun.mlconf.artifact_path
        )
        self.storage_options = store.get_storage_options()

    @classmethod
    def _from_ml_ctx(
        cls,
        context: "mlrun.MLClientCtx",
        *,
        application_name: str,
        event: dict[str, Any],
        model_endpoint_dict: Optional[dict[str, ModelEndpoint]] = None,
        sample_df: Optional[pd.DataFrame] = None,
        feature_stats: Optional[FeatureStats] = None,
    ) -> "MonitoringApplicationContext":
        project = context.get_project_object()
        if not project:
            raise mlrun.errors.MLRunValueError("Could not load project from context")
        logger = context.logger
        artifacts_logger = context
        nuclio_logger = nuclio.request.Logger(
            level=mlrun.mlconf.log_level, name=cls._logger_name
        )
        return cls(
            application_name=application_name,
            event=event,
            model_endpoint_dict=model_endpoint_dict,
            project=project,
            logger=logger,
            nuclio_logger=nuclio_logger,
            artifacts_logger=artifacts_logger,
            sample_df=sample_df,
            feature_stats=feature_stats,
        )

    @classmethod
    def _from_graph_ctx(
        cls,
        graph_context: mlrun.serving.GraphContext,
        *,
        application_name: str,
        event: dict[str, Any],
        model_endpoint_dict: Optional[dict[str, ModelEndpoint]] = None,
        sample_df: Optional[pd.DataFrame] = None,
        feature_stats: Optional[FeatureStats] = None,
        feature_sets_dict: Optional[dict[str, fs.FeatureSet]] = None,
    ) -> "MonitoringApplicationContext":
        nuclio_logger = graph_context.logger
        artifacts_logger = graph_context.project_obj
        logger = mlrun.utils.create_logger(
            level=mlrun.mlconf.log_level,
            formatter_kind=mlrun.mlconf.log_formatter,
            name=cls._logger_name,
        )
        return cls(
            application_name=application_name,
            event=event,
            project=graph_context.project_obj,
            model_endpoint_dict=model_endpoint_dict,
            logger=logger,
            nuclio_logger=nuclio_logger,
            artifacts_logger=artifacts_logger,
            sample_df=sample_df,
            feature_stats=feature_stats,
            feature_sets_dict=feature_sets_dict,
        )

    def _get_default_labels(self) -> dict[str, str]:
        labels = {
            mlrun_constants.MLRunInternalLabels.runner_pod: socket.gethostname(),
            mlrun_constants.MLRunInternalLabels.producer_type: "model-monitoring-app",
            mlrun_constants.MLRunInternalLabels.app_name: self.application_name,
            mlrun_constants.MLRunInternalLabels.endpoint_id: self.endpoint_id,
            mlrun_constants.MLRunInternalLabels.endpoint_name: self.endpoint_name,
        }
        return {key: value for key, value in labels.items() if value is not None}

    def _add_default_labels(self, labels: Optional[dict[str, str]]) -> dict[str, str]:
        """Add the default labels to logged artifacts labels"""
        return (labels or {}) | self._default_labels

    @property
    def sample_df(self) -> pd.DataFrame:
        if self._sample_df is None:
            if (
                self.endpoint_name is None
                or self.endpoint_id is None
                or pd.isnull(self.start_infer_time)
                or pd.isnull(self.end_infer_time)
            ):
                raise mlrun.errors.MLRunValueError(
                    "You have tried to access `monitoring_context.sample_df`, but have not provided it directly "
                    "through `sample_data`, nor have you provided the model endpoint's name, ID, and the start and "
                    f"end times: `endpoint_name`={self.endpoint_name}, `endpoint_uid`={self.endpoint_id}, "
                    f"`start`={self.start_infer_time}, and `end`={self.end_infer_time}. "
                    "You can either provide the sample dataframe directly, the model endpoint's details and times, "
                    "or adapt the application's logic to not access the sample dataframe."
                )
            df = self.feature_set.to_dataframe(
                start_time=self.start_infer_time,
                end_time=self.end_infer_time,
                time_column=mm_constants.EventFieldType.TIMESTAMP,
                storage_options=self.storage_options,
            )
            self._sample_df = df.reset_index(drop=True)
        return self._sample_df

    @property
    def model_endpoint(self) -> ModelEndpoint:
        if not self._model_endpoint:
            if self.endpoint_name is None or self.endpoint_id is None:
                raise mlrun.errors.MLRunValueError(
                    "You have NOT provided the model endpoint's name and ID: "
                    f"`endpoint_name`={self.endpoint_name} and `endpoint_id`={self.endpoint_id}, "
                    "but you have tried to access `monitoring_context.model_endpoint` "
                    "directly or indirectly in your application. You can either provide them, "
                    "or adapt the application's logic to not access the model endpoint."
                )
            self._model_endpoint = mlrun.db.get_run_db().get_model_endpoint(
                name=self.endpoint_name,
                project=self.project_name,
                endpoint_id=self.endpoint_id,
                feature_analysis=True,
            )
        return self._model_endpoint

    @property
    def feature_set(self) -> fs.FeatureSet:
        if not self._feature_set and self.model_endpoint:
            self._feature_set = fstore.get_feature_set(
                self.model_endpoint.spec.monitoring_feature_set_uri
            )
        return self._feature_set

    @property
    def feature_stats(self) -> FeatureStats:
        if not self._feature_stats:
            self._feature_stats = self.model_endpoint.spec.feature_stats
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
        return self.model_endpoint.spec.feature_names

    @property
    def label_names(self) -> list[str]:
        """The label names of the model"""
        return self.model_endpoint.spec.label_names

    @property
    def model(self) -> tuple[str, ModelArtifact, dict]:
        """The model file, model spec object, and a list of extra data items"""
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
        unique_per_endpoint: bool = True,
        **kwargs,
    ) -> Artifact:
        """
        Log an artifact.

        .. caution::

            Logging artifacts in every model monitoring window may cause scale issues.
            This method should be called on special occasions only.

        See :func:`~mlrun.projects.MlrunProject.log_artifact` for the full documentation, except for one
        new argument:

        :param unique_per_endpoint: by default ``True``, we will log different artifact for each model endpoint,
                                    set to ``False`` without changing item key will cause artifact override.
        """
        labels = self._add_default_labels(labels)
        # By default, we want to log different artifact for each model endpoint
        endpoint_id = labels.get(mlrun_constants.MLRunInternalLabels.endpoint_id, "")
        if unique_per_endpoint and isinstance(item, str):
            item = f"{item}-{endpoint_id}" if endpoint_id else item
        elif unique_per_endpoint:  # isinstance(item, Artifact) is True
            item.key = f"{item.key}-{endpoint_id}" if endpoint_id else item.key

        return self._artifacts_logger.log_artifact(
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
        unique_per_endpoint: bool = True,
        **kwargs,
    ) -> DatasetArtifact:
        """
        Log a dataset artifact.

        .. caution::

            Logging datasets in every model monitoring window may cause scale issues.
            This method should be called on special occasions only.

        See :func:`~mlrun.projects.MlrunProject.log_dataset` for the full documentation, except for one
        new argument:

        :param unique_per_endpoint: by default ``True``, we will log different artifact for each model endpoint,
                                    set to ``False`` without changing item key will cause artifact override.
        """
        labels = self._add_default_labels(labels)
        # By default, we want to log different artifact for each model endpoint
        endpoint_id = labels.get(mlrun_constants.MLRunInternalLabels.endpoint_id, "")
        if unique_per_endpoint and isinstance(key, str):
            key = f"{key}-{endpoint_id}" if endpoint_id else key

        return self._artifacts_logger.log_dataset(
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
