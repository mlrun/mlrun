# Copyright 2026 Iguazio
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

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

import pandas as pd
import pytest
from datastore.remote_model.remote_model_utils import BATCH_INPUT_DATA

import mlrun
import mlrun.common.schemas.model_monitoring.constants as mm_constants
from mlrun.datastore.model_provider.model_provider import UsageResponseKeys
from mlrun.runtimes.nuclio.function import AsyncSpec
from tests.datastore.remote_model.remote_model_utils import (
    setup_remote_model_test,
)
from tests.datastore.remote_model.test_remote_model import BaseMockModelProviderTest
from tests.system.model_monitoring import TestMLRunSystemModelMonitoring


class TestMockModelProviderTracking(
    BaseMockModelProviderTest, TestMLRunSystemModelMonitoring
):
    """Test MockModelProvider with tracking using real function deployment"""

    project_name = "mock-model-tracking-test"
    image = "mlrun/mlrun"

    def _verify_direct_parquet_contents(self, v3io_df, endpoint_name, batch_len):
        """Verify parquet contents by splitting by request_id and validating each group"""
        grouped = v3io_df.groupby("request_id")

        # Should have 2 request groups: 1 single invocation + 1 batch invocation
        assert len(grouped) == 2, f"Expected 2 request groups, got {len(grouped)}"
        single_group = None
        batch_group = None

        for request_id, group in grouped:
            if len(group) == 1:
                single_group = group
            elif len(group) == batch_len:
                batch_group = group
            else:
                raise AssertionError(
                    f"Unexpected group size: {len(group)} for request_id {request_id}"
                )

        assert not single_group.empty
        assert not batch_group.empty

        self._verify_single_parquet_row(
            single_group.iloc[0], endpoint_name, BATCH_INPUT_DATA[0]
        )

        # Verify batch invocation group
        self._verify_direct_batch_parquet_rows(
            batch_group, endpoint_name, BATCH_INPUT_DATA
        )

    def _verify_batch_row_common(
        self, row, endpoint_name, batch_size, expected_input, expected_counter=None
    ):
        """Verify common batch row structure and content"""
        assert row["endpoint_name"] == endpoint_name
        assert row["model_class"] == "LLModel"
        assert row["effective_sample_count"] == batch_size
        assert row["estimated_prediction_count"] == batch_size

        expected_feature_names = list(expected_input.keys())

        assert list(row["feature_names"]) == expected_feature_names

        assert list(row["label_names"]) == ["answer", "usage"]

        for key in expected_input:
            assert (
                row[key] == expected_input[key]
            ), f"Field {key} mismatch: {row[key]} != {expected_input[key]}"

        assert "mock model provider" in row["answer"].lower()

        # Only check item counter if expected_counter is provided (for batch invocations)
        if expected_counter is not None:
            assert (
                f"(Item {expected_counter})" in row["answer"]
            ), f"Expected '(Item {expected_counter})' in answer"

        assert isinstance(row["usage"], dict)
        assert row["usage"]["prompt_tokens"] == 0
        assert row["usage"]["completion_tokens"] == 0
        assert row["usage"]["total_tokens"] == 0

    def _verify_single_parquet_row(self, row, endpoint_name, expected_input):
        """Verify a single parquet row matches expected input and output structure"""
        # Single invocation has batch_size=1 and no item counter
        self._verify_batch_row_common(
            row,
            endpoint_name,
            batch_size=1,
            expected_input=expected_input,
            expected_counter=None,
        )

    def _verify_batch_group_common(
        self, batch_group, endpoint_name, batch_size, batch_id=""
    ):
        """Verify common batch group properties: timestamp, latency, and per-row structure"""
        # All rows in same batch must have same timestamp and latency
        for field in ["timestamp", "latency"]:
            values = batch_group[field].unique()
            assert (
                len(values) == 1
            ), f"Batch {batch_id}: expected same {field} for all rows, got {len(values)} different values"

    def _verify_direct_batch_parquet_rows(
        self, batch_group, endpoint_name, expected_inputs
    ):
        """Verify batch parquet rows match expected inputs and output structure"""
        batch_size = len(expected_inputs)
        self._verify_batch_group_common(batch_group, endpoint_name, batch_size)

        # Order rows by original BATCH_INPUT_DATA position for straightforward index-based comparison
        batch_group["original_index"] = batch_group["question"].map(
            {inp["question"]: i for i, inp in enumerate(expected_inputs)}
        )
        batch_sorted = batch_group.sort_values("original_index").reset_index(drop=True)

        for i, row in batch_sorted.iterrows():
            self._verify_batch_row_common(
                row, endpoint_name, batch_size, expected_inputs[i], expected_counter=i
            )

    def _verify_batch_step_parquet_contents(self, v3io_df, endpoint_name):
        """Verify parquet contents for batch step test (3 batches: 2+2+1)"""
        grouped = v3io_df.groupby("request_id")

        # Should have 3 request groups (3 batches)
        assert len(grouped) == 3, f"Expected 3 request groups, got {len(grouped)}"

        # Group sizes should be 2, 2, 1
        group_sizes = [len(group) for _, group in grouped]
        assert group_sizes == [2, 2, 1]

        # Verify each batch
        for request_id, group in grouped:
            batch_size = len(group)
            self._verify_batch_group_common(
                group, endpoint_name, batch_size, batch_id=request_id
            )

            # Order rows by original BATCH_INPUT_DATA position
            group["original_index"] = group["question"].map(
                {inp["question"]: i for i, inp in enumerate(BATCH_INPUT_DATA)}
            )
            batch_sorted = group.sort_values("original_index").reset_index(drop=True)

            # Verify each row in the batch
            for idx, row in batch_sorted.iterrows():
                expected_input = BATCH_INPUT_DATA[int(group["original_index"][idx])]

                # Item counter matches batch position (0 or 1 for batch of 2, 0 for batch of 1)
                expected_counter = idx % 2 if batch_size == 2 else 0
                self._verify_batch_row_common(
                    row, endpoint_name, batch_size, expected_input, expected_counter
                )

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_tracking(self, execution_mechanism):
        """Test single and batch invocations with MockModelProvider with model monitoring"""
        mlrun_model_name = "mock_model"
        endpoint_name = "my_endpoint"
        model_url = "mock://my-mock-model"
        batch_len = len(BATCH_INPUT_DATA)
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            self.project,
            model_url,
            mlrun_model_name=mlrun_model_name,
            image=self.image,
            execution_mechanism=execution_mechanism,
        )

        self.set_mm_credentials()
        function.set_tracking()
        self.project.enable_model_monitoring(
            deploy_histogram_data_drift_app=False,
            image=self.image,
        )

        function.deploy()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    self._check_single_invocation,
                    function.invoke,
                    mlrun_model_name,
                ),
                executor.submit(
                    self._check_batch_invocation,
                    function.invoke,
                    mlrun_model_name,
                ),
                executor.submit(
                    self._check_single_invocation_with_error,
                    function.invoke,
                    mlrun_model_name,
                ),
                executor.submit(
                    self._check_batch_invocation_with_error,
                    function.invoke,
                    mlrun_model_name,
                ),
            ]

        for future in as_completed(futures):
            future.result()

        sleep(5)

        endpoint = (
            mlrun.get_run_db()
            .list_model_endpoints(
                self.project_name, metric_list=["error_count"], tsdb_metrics=True
            )
            .endpoints[0]
        )
        assert endpoint.metadata.name == endpoint_name

        sleep(180)

        function_name = function.metadata.name
        mep = mlrun.db.get_run_db().get_model_endpoint(
            name=endpoint_name,
            project=self.project.name,
            function_name=function_name,
            function_tag="latest",
            feature_analysis=True,
            tsdb_metrics=True,
        )
        assert mep is not None

        tsdb_client = mlrun.model_monitoring.get_tsdb_connector(
            project=self.project.name, profile=self.mm_tsdb_profile
        )
        predictions = tsdb_client._get_records(
            table=mm_constants.V3IOTSDBTables.PREDICTIONS, start="now-50m", end="now"
        )

        assert len(predictions) == 2
        single_predication = (
            predictions[predictions["estimated_prediction_count"] == 1]
            .iloc[0]
            .to_dict()
        )
        batch_prediction = (
            predictions[predictions["estimated_prediction_count"] == batch_len]
            .iloc[0]
            .to_dict()
        )
        assert single_predication
        assert batch_prediction
        assert single_predication["effective_sample_count"] == 1
        assert batch_prediction["effective_sample_count"] == batch_len

        v3io_df = pd.read_parquet(
            f"v3io:///projects/{self.project.name}/artifacts/model-endpoints/parquet/key={mep.metadata.uid}"
        )
        assert len(v3io_df) == batch_len + 1

        self._verify_direct_parquet_contents(v3io_df, endpoint_name, batch_len)

        error_df = tsdb_client.get_error_count(endpoint_ids=mep.metadata.uid)
        assert len(error_df) == 1
        error_dict = error_df.head(1).to_dict(orient="records")[0]
        assert error_dict["error_count"] == 2

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_batch_step_with_graph(self, execution_mechanism):
        mlrun_model_name = "mock_model"
        model_url = "mock://my-mock-model"

        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            self.project,
            model_url,
            mlrun_model_name=mlrun_model_name,
            image=self.image,
            execution_mechanism=execution_mechanism,
            batch_step=True,
        )
        function.with_http(workers=None, async_spec=AsyncSpec())
        self.set_mm_credentials()
        function.set_tracking()
        self.project.enable_model_monitoring(
            deploy_histogram_data_drift_app=False,
            image=self.image,
        )

        function.deploy()

        def send_event(event, delay):
            sleep(delay)
            return function.invoke(f"v2/models/{mlrun_model_name}/infer", event)

        with ThreadPoolExecutor(max_workers=len(BATCH_INPUT_DATA)) as executor:
            # MockProvider requires a larger delay (0.3s) because batching output depends on the order of requests,
            # which can introduce race conditions, unlike real providers where batching output depends on input.
            futures = [
                executor.submit(send_event, input_event, i * 0.3)
                for i, input_event in enumerate(BATCH_INPUT_DATA)
            ]
            responses = [future.result() for future in futures]

        self._verify_batch_response(responses)
        for i, response in enumerate(responses):
            output = response["output"]
            # in order to check batches of 2:
            expected_counter = i % 2
            assert f"(Item {expected_counter})" in output[UsageResponseKeys.ANSWER]

        # Verify tracking data - model monitoring verification
        sleep(180)

        endpoint_name = "my_endpoint"
        function_name = function.metadata.name

        mep = mlrun.db.get_run_db().get_model_endpoint(
            name=endpoint_name,
            project=self.project.name,
            function_name=function_name,
            function_tag="latest",
            feature_analysis=True,
            tsdb_metrics=True,
        )
        assert mep is not None

        # Verify TSDB predictions - should still have 3 successful batches (2+2+1)
        # Error batch is not counted as a prediction
        tsdb_client = mlrun.model_monitoring.get_tsdb_connector(
            project=self.project.name, profile=self.mm_tsdb_profile
        )
        predictions = tsdb_client._get_records(
            table=mm_constants.V3IOTSDBTables.PREDICTIONS, start="now-50m", end="now"
        )

        # Verify batch sizes (2+2+1)
        assert len(predictions) == 3
        batch_sizes = predictions["estimated_prediction_count"].tolist()
        assert batch_sizes == [2, 2, 1]

        v3io_df = pd.read_parquet(
            f"v3io:///projects/{self.project.name}/artifacts/model-endpoints/parquet/key={mep.metadata.uid}"
        )
        assert len(v3io_df) == len(BATCH_INPUT_DATA)

        # Verify batch step structure - still 3 request groups (2+2+1, error batch not included)
        self._verify_batch_step_parquet_contents(v3io_df, endpoint_name)
