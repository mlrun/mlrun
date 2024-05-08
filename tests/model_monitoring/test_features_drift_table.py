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
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mlrun
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.model_monitoring.applications.context as mm_context
import mlrun.model_monitoring.applications.histogram_data_drift as histogram_data_drift
import mlrun.utils
from mlrun.common.model_monitoring.helpers import FeatureStats, pad_features_hist
from mlrun.data_types.infer import DFDataInfer, default_num_bins
from mlrun.model_monitoring.helpers import calculate_inputs_statistics


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
        DFDataInfer.get_stats(
            df=sample_data,
            options=mlrun.data_types.infer.InferOptions.Histogram,
        )
    )
    pad_features_hist(sample_data_statistics)
    inputs_statistics = FeatureStats(
        calculate_inputs_statistics(
            sample_set_statistics=sample_data_statistics,
            inputs=inputs,
        )
    )
    context.__class__ = mm_context.MonitoringApplicationContext
    monitoring_context = mm_context.MonitoringApplicationContext().from_dict(
        {
            mm_constants.ApplicationEvent.FEATURE_STATS: json.dumps(inputs_statistics),
            mm_constants.ApplicationEvent.CURRENT_STATS: json.dumps(
                sample_data_statistics
            ),
        },
        context=context,
        model_endpoint_dict={},
    )
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
        artifact_path=str(tmp_path),
        handler=plot_produce,
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

    @staticmethod
    @pytest.fixture
    def shared_feat() -> str:
        return "orig_feat0"

    @staticmethod
    @pytest.fixture
    def new_feat() -> str:
        return "new_feat0"

    @classmethod
    @pytest.fixture
    def sample_set_statistics(cls, shared_feat: str) -> dict:
        return {
            shared_feat: {
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

    @staticmethod
    @pytest.fixture
    def inputs_df(shared_feat: str, new_feat: str) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[shared_feat, new_feat],
            data=np.random.randint(-15, 20, size=(9, 2)),
        )

    @staticmethod
    @pytest.fixture
    def input_statistics(sample_set_statistics: dict, inputs_df: pd.DataFrame) -> dict:
        return calculate_inputs_statistics(
            sample_set_statistics=sample_set_statistics, inputs=inputs_df
        )

    @classmethod
    def test_histograms_length(
        cls, shared_feat: str, new_feat: str, input_statistics: dict
    ) -> None:
        assert len(input_statistics[shared_feat][cls._HIST][0]) == len(
            input_statistics[new_feat][cls._HIST][0]
        ), "The lengths of the histograms do not match"
