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

from unittest import mock

import pytest

from mlrun.data_types import InferOptions
from mlrun.datastore.targets import ParquetTarget
from mlrun.feature_store import Entity
from mlrun.feature_store.common import RunConfig
from mlrun.feature_store.feature_set import FeatureSet
from mlrun.model import DataSource, DataTargetBase


@mock.patch("mlrun.feature_store.api._ingest")
def test_ingest_method(mock_ingest):
    # Create an instance of FeatureSet
    fset = FeatureSet()

    # Define your test inputs
    test_source = "test_source"
    test_targets = ["target1", "target2"]
    test_namespace = "test_namespace"
    test_return_df = True
    test_infer_options = InferOptions.default()
    test_run_config = RunConfig()
    test_mlrun_context = "test_mlrun_context"
    test_spark_context = "test_spark_context"
    test_overwrite = True

    # Call the ingest method
    fset.ingest(
        source=test_source,
        targets=test_targets,
        namespace=test_namespace,
        return_df=test_return_df,
        infer_options=test_infer_options,
        run_config=test_run_config,
        mlrun_context=test_mlrun_context,
        spark_context=test_spark_context,
        overwrite=test_overwrite,
    )

    # Assert that mlrun.feature_store.api.ingest was called with the correct parameters
    mock_ingest.assert_called_once_with(
        fset,
        test_source,
        test_targets,
        test_namespace,
        test_return_df,
        test_infer_options,
        test_run_config,
        test_mlrun_context,
        test_spark_context,
        test_overwrite,
    )


@mock.patch("mlrun.feature_store.api._preview")
def test_preview_method(mock_preview):
    # Create an instance of FeatureSet
    fset = FeatureSet()

    # Define your test inputs
    test_source = "test_source"
    test_entity_columns = ["col1", "col2"]
    test_namespace = "test_namespace"
    test_options = InferOptions.default()  # Assuming InferOptions is available
    test_verbose = True
    test_sample_size = 100

    # Call the preview method
    fset.preview(
        source=test_source,
        entity_columns=test_entity_columns,
        namespace=test_namespace,
        options=test_options,
        verbose=test_verbose,
        sample_size=test_sample_size,
    )

    # Assert that mlrun.feature_store.api.preview was called with the correct parameters
    mock_preview.assert_called_once_with(
        fset,
        test_source,
        test_entity_columns,
        test_namespace,
        test_options,
        test_verbose,
        test_sample_size,
    )


@mock.patch("mlrun.feature_store.api._deploy_ingestion_service_v2")
def test_deploy_ingestion_service(mock_deploy):
    # Create an instance of FeatureSet
    fset = FeatureSet()

    # Define your test inputs
    test_source = DataSource()  # Assuming DataSource is a valid class
    test_targets = [
        DataTargetBase(),
        DataTargetBase(),
    ]  # Replace with valid DataTargetBase instances
    test_name = "test_service"
    test_run_config = RunConfig()  # Assuming RunConfig is a valid class
    test_verbose = True

    # Call the deploy_ingestion_service method
    fset.deploy_ingestion_service(
        source=test_source,
        targets=test_targets,
        name=test_name,
        run_config=test_run_config,
        verbose=test_verbose,
    )

    # Assert that deploy_ingestion_service was called with the correct parameters
    mock_deploy.assert_called_once_with(
        fset, test_source, test_targets, test_name, test_run_config, test_verbose
    )


@pytest.mark.parametrize(
    "num_targets",
    [
        1,
        2,
    ],
)
@pytest.mark.parametrize(
    "after_step_value",
    [
        ["step2"],  # Single final step (list with 1 item)
        ["step2", "step3"],  # Multiple final steps (list with 2 items)
    ],
)
def test_feature_set_plot_with_targets(num_targets, after_step_value):
    include_step_3 = "step3" in after_step_value

    fset = FeatureSet("test", entities=[Entity("id")])
    fset.graph.add_step(name="step1", class_name="storey.Map", _fn="(event)")
    fset.graph.add_step(
        name="step2", class_name="storey.Map", _fn="(event)", after="step1"
    )

    if include_step_3:
        fset.graph.add_step(
            name="step3", class_name="storey.Map", _fn="(event)", after="step1"
        )

    if num_targets == 1:
        fset.set_targets(
            targets=[ParquetTarget(name="test-target", after_step=after_step_value)],
            with_defaults=False,
        )
    else:
        # Multiple targets (use defaults and set after_step)
        fset.set_targets()
        for target in fset.spec.targets:
            target.after_step = after_step_value
    graph = fset.plot(rankdir="LR", with_targets=True)

    assert graph is not None
    graph_source = graph.source

    assert "step2" in graph_source
    if include_step_3:
        assert "step3" in graph_source
    else:
        assert "step3" not in graph_source

    target_count = 0
    if "parquet" in graph_source.lower():
        target_count += 1
    if "nosql" in graph_source.lower():
        target_count += 1

    assert target_count == num_targets
