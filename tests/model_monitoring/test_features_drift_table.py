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
import json
import os
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

import mlrun
from mlrun.artifacts import Artifact
from mlrun.data_types.infer import DFDataInfer, default_num_bins
from mlrun.model_monitoring.features_drift_table import FeaturesDriftTablePlot
from mlrun.model_monitoring.model_monitoring_batch import (
    VirtualDrift,
    calculate_inputs_statistics,
)


def generate_data(
    n_samples: int,
    n_features: int,
    loc_range: Tuple[int, int] = (0.1, 1.1),
    scale_range: Tuple[int, int] = (1, 2),
    inputs_diff_range: Tuple[int, int] = (0, 1),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    sample_data_statistics = DFDataInfer.get_stats(
        df=sample_data,
        options=mlrun.data_types.infer.InferOptions.Histogram,
    )
    inputs_statistics = calculate_inputs_statistics(
        sample_set_statistics=sample_data_statistics,
        inputs=inputs,
    )

    # Calculate drift:
    virtual_drift = VirtualDrift(inf_capping=10)
    metrics = virtual_drift.compute_drift_from_histograms(
        feature_stats=sample_data_statistics,
        current_stats=inputs_statistics,
    )
    drift_results = virtual_drift.check_for_drift_per_feature(
        metrics_results_dictionary=metrics
    )

    # Plot:
    html_plot = FeaturesDriftTablePlot().produce(
        features=list(sample_data.columns),
        sample_set_statistics=sample_data_statistics,
        inputs_statistics=inputs_statistics,
        metrics=metrics,
        drift_results=drift_results,
    )

    # Log:
    context.log_artifact(
        Artifact(body=html_plot, format="html", key="drift_table_plot")
    )


def test_plot_produce(rundb_mock):
    # Create a temp directory:
    output_path = tempfile.TemporaryDirectory()

    # Run the plot production and logging:
    train_run = mlrun.new_function().run(
        artifact_path=output_path.name,
        handler=plot_produce,
    )

    # Print the outputs for manual validation:
    print(json.dumps(train_run.outputs, indent=4))

    # Validate the artifact was logged:
    assert len(train_run.status.artifacts) == 1

    # Check the plot was saved properly (only the drift table plot should appear):
    artifact_directory_content = os.listdir(
        os.path.dirname(train_run.status.artifacts[0]["spec"]["target_path"])
    )
    assert len(artifact_directory_content) == 1
    assert artifact_directory_content[0] == "drift_table_plot.html"

    # Clean up the temporary directory:
    output_path.cleanup()


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
