import json
import os
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd

import mlrun
from mlrun.artifacts import Artifact
from mlrun.data_types.infer import DFDataInfer
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


def test_plot_produce():
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
        os.path.dirname(train_run.outputs["drift_table_plot"])
    )
    assert len(artifact_directory_content) == 1
    assert artifact_directory_content[0] == "drift_table_plot.html"

    # Clean up the temporary directory:
    output_path.cleanup()
