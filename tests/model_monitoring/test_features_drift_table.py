import os
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd

import mlrun
from mlrun.data_types.infer import DFDataInfer
from mlrun.model_monitoring.features_drift_table import FeaturesDriftTablePlot
from mlrun.model_monitoring.model_monitoring_batch import VirtualDrift


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


def test_plot_produce():
    # Create a temp directory:
    output_path = tempfile.TemporaryDirectory()

    # Generate data:
    sample_data, inputs = generate_data(
        250000, 40, inputs_diff_range=(0, 150000), scale_range=(0, 1500)
    )

    # Calculate statistics:
    sample_data_statistics = DFDataInfer.get_stats(
        df=sample_data,
        options=mlrun.data_types.infer.InferOptions.Histogram,
    )
    inputs_statistics = DFDataInfer.get_stats(
        df=inputs,
        options=mlrun.data_types.infer.InferOptions.Histogram,
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
    FeaturesDriftTablePlot().produce(
        features=list(sample_data.columns),
        sample_set_statistics=sample_data_statistics,
        inputs_statistics=inputs_statistics,
        metrics=metrics,
        drift_results=drift_results,
        output_path=output_path.name,
    )

    # Check the plot was saved:
    assert len(os.listdir(output_path.name)) == 1

    # Clean up the temporary directory:
    output_path.cleanup()
