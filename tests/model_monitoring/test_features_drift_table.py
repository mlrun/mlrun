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

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

import mlrun
import mlrun.model_monitoring.applications.context as mm_context
import mlrun.model_monitoring.applications.histogram_data_drift as histogram_data_drift
import mlrun.serving
from mlrun.common.model_monitoring.helpers import FeatureStats, pad_features_hist
from mlrun.data_types.infer import DFDataInfer, InferOptions, default_num_bins
from mlrun.model_monitoring.helpers import calculate_inputs_statistics


@contextmanager
def mocked_graph_context_project() -> Iterator[None]:
    with patch("mlrun.serving.GraphContext.project", PropertyMock) as project_mock:
        project_mock.return_value = "proj-0"
        yield


def generate_data(
    n_samples: int,
    n_features: int,
    loc_range: tuple[float, float] = (0.1, 1.1),
    scale_range: tuple[int, int] = (1, 2),
    inputs_diff_range: tuple[int, int] = (0, 1),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Generate data:
    data = {}
    for i in range(n_features):
        loc = np.random.uniform(low=loc_range[0], high=loc_range[1])
        scale = np.random.uniform(low=scale_range[0], high=scale_range[1])
        feature_data = np.random.normal(loc=loc, scale=scale, size=n_samples)
        data[f"feature_{i}"] = feature_data

    # Sample data:
    sample_data = pd.DataFrame(data)

    # Inputs data:
    min_range, max_range = inputs_diff_range
    inputs = {}
    for feature_name, feature_data in data.items():
        additions = np.random.uniform(low=min_range, high=max_range, size=n_samples)
        inputs[feature_name] = feature_data + additions
    inputs = pd.DataFrame(inputs)

    return sample_data, inputs


def plot_produce(context: mlrun.MLClientCtx):
    # Generate data:
    sample_data, inputs = generate_data(
        250000, 40, inputs_diff_range=(0, 1), scale_range=(0, 1)
    )

    # Calculate statistics:
    sample_data_statistics = FeatureStats(
        DFDataInfer.get_stats(df=sample_data, options=InferOptions.Histogram)
    )
    pad_features_hist(sample_data_statistics)
    inputs_statistics = FeatureStats(
        calculate_inputs_statistics(
            sample_set_statistics=sample_data_statistics,
            inputs=inputs,
        )
    )
    with patch(
        "mlrun.load_project",
        Mock(return_value=Mock(spec=mlrun.projects.project.MlrunProject)),
    ):
        with mocked_graph_context_project():
            monitoring_context = mm_context.MonitoringApplicationContext(
                graph_context=mlrun.serving.GraphContext(),
                application_name="histogram-data-drift",
                event={},
                model_endpoint_dict={},
            )
            monitoring_context._feature_stats = inputs_statistics
            monitoring_context._sample_df_stats = sample_data_statistics
    # Patching `log_artifact` only for this test
    monitoring_context.log_artifact = context.log_artifact
    # Initialize the app
    application = histogram_data_drift.HistogramDataDriftApplication()
    # Calculate drift
    metrics_per_feature = application._compute_metrics_per_feature(
        monitoring_context=monitoring_context,
    )
    application._log_drift_artifacts(
        monitoring_context=monitoring_context,
        metrics_per_feature=metrics_per_feature,
        log_json_artifact=False,
    )


def test_plot_produce(tmp_path: Path) -> None:
    # Run the plot production and logging:
    app_plot_run = mlrun.new_function().run(
        artifact_path=str(tmp_path), handler=plot_produce
    )

    # Validate the artifact was logged:
    assert len(app_plot_run.status.artifacts) == 1

    # Check the plot was saved properly (only the drift table plot should appear):
    artifact_directory_content = list(
        Path(app_plot_run.status.artifacts[0]["spec"]["target_path"]).parent.glob("*")
    )
    assert len(artifact_directory_content) == 1
    assert artifact_directory_content[0].name == "drift_table_plot.html"


class TestCalculateInputsStatistics:
    _HIST = "hist"
    _DEFAULT_NUM_BINS = default_num_bins
    _SHARED_FEATURE = "shared_feature"

    @classmethod
    @pytest.fixture
    def sample_set_statistics(cls) -> dict:
        return {
            cls._SHARED_FEATURE: {
                cls._HIST: [
                    [0, *list(np.random.randint(10, size=cls._DEFAULT_NUM_BINS)), 0],
                    [
                        -10e20,
                        *list(np.linspace(0, 1, cls._DEFAULT_NUM_BINS + 1)),
                        10e20,
                    ],
                ]
            }
        }

    @classmethod
    @pytest.fixture
    def inputs_df(cls) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[cls._SHARED_FEATURE, "feature_1"],
            data=np.random.randint(-15, 20, size=(9, 2)),
        )

    @classmethod
    def test_histograms_features(
        cls, sample_set_statistics: dict, inputs_df: pd.DataFrame
    ) -> None:
        current_stats = calculate_inputs_statistics(
            sample_set_statistics=sample_set_statistics, inputs=inputs_df
        )
        assert (
            current_stats.keys() == sample_set_statistics.keys()
        ), "Inputs statistics and the current statistics should have the same features"
