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

    def _verify_parquet_contents(self, v3io_df, endpoint_name, batch_len):
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
        self._verify_batch_parquet_rows(batch_group, endpoint_name, BATCH_INPUT_DATA)

    def _verify_single_parquet_row(self, row, endpoint_name, expected_input):
        """Verify a single parquet row matches expected input and output structure"""
        assert row["endpoint_name"] == endpoint_name
        assert row["model_class"] == "LLModel"
        assert row["effective_sample_count"] == row["estimated_prediction_count"] == 1
        expected_feature_names = list(expected_input.keys())

        assert list(row["feature_names"]) == expected_feature_names

        assert list(row["label_names"]) == ["answer", "usage"]

        for key in expected_input:
            assert (
                row[key] == expected_input[key]
            ), f"Field {key} mismatch: {row[key]} != {expected_input[key]}"

        assert "mock model provider" in row["answer"].lower()

        #  TODO : extract usage data to different columns
        assert isinstance(row["usage"], dict)
        assert row["usage"]["prompt_tokens"] == 0
        assert row["usage"]["completion_tokens"] == 0
        assert row["usage"]["total_tokens"] == 0

    def _verify_batch_parquet_rows(self, batch_group, endpoint_name, expected_inputs):
        """Verify batch parquet rows match expected inputs and output structure"""
        timestamps = batch_group["timestamp"].unique()
        assert (
            len(timestamps) == 1
        ), f"Expected same timestamp for all batch rows, got {len(timestamps)} different values"

        latencies = batch_group["latency"].unique()
        assert (
            len(latencies) == 1
        ), f"Expected same latency for all batch rows, got {len(latencies)} different values"

        # Order rows by original BATCH_INPUT_DATA position for straightforward index-based comparison
        batch_group["original_index"] = batch_group["question"].map(
            {inp["question"]: i for i, inp in enumerate(expected_inputs)}
        )
        batch_sorted = batch_group.sort_values("original_index").reset_index(drop=True)

        for i, row in batch_sorted.iterrows():
            expected_input = expected_inputs[i]

            assert row["endpoint_name"] == endpoint_name
            assert row["model_class"] == "LLModel"
            assert (
                row["effective_sample_count"]
                == row["estimated_prediction_count"]
                == len(expected_inputs)
            )

            expected_feature_names = list(expected_input.keys())
            assert list(row["feature_names"]) == expected_feature_names
            assert list(row["label_names"]) == ["answer", "usage"]

            for key in expected_input:
                assert (
                    row[key] == expected_input[key]
                ), f"Row {i}, field {key} mismatch: {row[key]} != {expected_input[key]}"

            assert "mock model provider" in row["answer"].lower()
            assert f"(Item {i})" in row["answer"], f"Expected '(Item {i})' in answer"

            assert isinstance(row["usage"], dict)
            assert row["usage"]["prompt_tokens"] == 0
            assert row["usage"]["completion_tokens"] == 0
            assert row["usage"]["total_tokens"] == 0

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

        self._verify_parquet_contents(v3io_df, endpoint_name, batch_len)

        error_df = tsdb_client.get_error_count(endpoint_ids=mep.metadata.uid)
        assert len(error_df) == 1
        error_dict = error_df.head(1).to_dict(orient="records")[0]
        assert error_dict["error_count"] == 2
