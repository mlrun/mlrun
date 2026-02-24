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
    "targets,expected_target_count",
    [
        (None, 2),  # Multiple default targets (parquet + nosql)
        ([ParquetTarget()], 1),  # Single target
    ],
)
@pytest.mark.parametrize(
    "aggregations,description",
    [
        # (
        #     [("amount", "amount_agg1", ["sum"], ["1h"])],
        #     "single aggregation",
        # ),
        (
            [
                ("amount", "amount_agg1", ["sum"], ["1h"]),
                ("amount", "amount_agg2", ["avg"], ["2h"]),
            ],
            "multiple aggregations",
        ),
    ],
)
def test_feature_set_plot_with_targets(
    targets, expected_target_count, aggregations, description
):
    """Test plot with targets - regression test for list handling bug.

    This test creates a scenario where target.after contains multiple steps,
    ensuring our fix properly handles iterating over the list.
    """
    fset = FeatureSet("test", entities=[Entity("id")])

    # Add aggregations based on parametrized input
    for column, agg_name, operations, windows in aggregations:
        fset.add_aggregation(
            name=agg_name,
            column=column,
            operations=operations,
            windows=windows,
            period="1h",
        )

    # Set targets based on parametrized input
    if targets is None:
        fset.set_targets()
    else:
        fset.set_targets(targets, with_defaults=False)


    # Should not crash with AttributeError
    graph = fset.plot(rankdir="LR", with_targets=True)
    assert graph is not None
    assert hasattr(graph, "source")
    graph_source = graph.source
    assert graph_source is not None

    # Verify edges were created
    assert "->" in graph_source

    # Verify expected number of targets in graph
    target_count = 0
    if "parquet" in graph_source.lower():
        target_count += 1
    if "nosql" in graph_source.lower():
        target_count += 1

    assert target_count == expected_target_count


def test_feature_set_plot_with_multiple_after_steps_manual():
    from mlrun.serving.states import BaseStep, RootFlowStep

    # Create a root flow with branching paths
    flow = RootFlowStep()

    # Add steps: step1 -> (step2, step3) both branch from step1
    flow.add_step(name="step1", class_name="storey.Map", _fn="(event)")
    flow.add_step(name="step2", class_name="storey.Map", _fn="(event)", after="step1")
    flow.add_step(name="step3", class_name="storey.Map", _fn="(event)", after="step1")

    # Create target that comes after BOTH step2 and step3
    # target.after will be ['step2', 'step3'] (list with 2 items)
    target = BaseStep(
        "parquet/test-target",
        after=["step2", "step3"],
        shape="cylinder",
    )

    # Should not crash - fix creates edge from step2->target AND step3->target
    graph = flow.plot(targets=[target])

    assert graph is not None
    graph_source = graph.source

    # Verify both steps and target exist in graph
    assert "step2" in graph_source
    assert "step3" in graph_source
    assert "test-target" in graph_source

