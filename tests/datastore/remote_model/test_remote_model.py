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
import unittest

import pytest

import mlrun
from mlrun.datastore.model_provider.model_provider import UsageResponseKeys
from tests.datastore.remote_model.remote_model_utils import (
    INPUT_DATA,
    create_mocked_get_store_artifact,
    setup_remote_model_test,
)


class TestMockModelProvider:
    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_batch(self, execution_mechanism, rundb_mock):
        """Test batch processing of multiple events with MockModelProvider"""
        project = mlrun.new_project("test-mock-model-batch", save=False)
        model_url = "mock://my-mock-model"
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            project,
            model_url,
            execution_mechanism=execution_mechanism,
        )
        function.set_tracking("dummy://", enable_tracking=True)

        mocked_get_store_artifact = create_mocked_get_store_artifact(
            {
                model_artifact.uri: model_artifact,
                llm_prompt_artifact.uri: llm_prompt_artifact,
            }
        )
        with (
            unittest.mock.patch(
                "mlrun.artifacts.llm_prompt.mlrun.datastore.store_manager.get_store_artifact",
                side_effect=lambda *args, **kwargs: mocked_get_store_artifact(
                    *args, **kwargs
                ),
            ),
        ):
            server = function.to_mock_server()
        try:
            # Send all INPUT_DATA events as batch
            response = server.test(body=INPUT_DATA)

            # Assert we got list of 5 responses
            assert isinstance(response, list)
            assert len(response) == len(INPUT_DATA)

            # Verify each response has correct structure
            for i, full_result in enumerate(response):
                result = full_result["output"]
                assert len(result) == 2  # answer + usage

                # Get answer and usage
                answer = result[UsageResponseKeys.ANSWER]
                stats = result[UsageResponseKeys.USAGE]

                # Verify mock message includes item index
                assert f"(Item {i})" in answer
                assert "mock model provider" in answer.lower()

                # Verify mock usage stats (should be 0)
                assert stats["prompt_tokens"] == 0
                assert stats["completion_tokens"] == 0
                assert stats["total_tokens"] == 0
        finally:
            server.wait_for_completion()
        dummy_stream = server.context.stream.output_stream
        event = dummy_stream.event_list[0]
        assert event["effective_sample_count"] == len(INPUT_DATA)
        assert event["request"]["inputs"] == INPUT_DATA
        assert event["labels"] == {}
        assert event["model"] == "my_endpoint"
        assert event["metrics"] is None

    def test_llmodel_batch_with_errors(self, rundb_mock):
        """Test that batch processing fails fast when MockModelProvider raises error"""
        project = mlrun.new_project("test-mock-model-batch-errors", save=False)
        model_url = "mock://my-mock-model"

        # Append error input to INPUT_DATA - the ERROR keyword will trigger mock error
        inputs = INPUT_DATA + [
            {
                "question": "ERROR - this should fail",
                "depth_level": "basic",
                "persona": "teacher",
                "tone": "formal",
            }
        ]

        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            project,
            model_url,
            execution_mechanism="naive",
        )
        function.set_tracking("dummy://", enable_tracking=True)

        mocked_get_store_artifact = create_mocked_get_store_artifact(
            {
                model_artifact.uri: model_artifact,
                llm_prompt_artifact.uri: llm_prompt_artifact,
            }
        )
        with unittest.mock.patch(
            "mlrun.artifacts.llm_prompt.mlrun.datastore.store_manager.get_store_artifact",
            side_effect=lambda *args, **kwargs: mocked_get_store_artifact(
                *args, **kwargs
            ),
        ):
            server = function.to_mock_server()

        try:
            # Should raise RuntimeError with "Mock error triggered" message
            with pytest.raises(
                RuntimeError, match=".*Mock error triggered by ERROR keyword.*"
            ):
                server.test(body=inputs)
        finally:
            server.wait_for_completion()
        dummy_stream = server.context.stream.output_stream
        event = dummy_stream.event_list[0]
        assert event["effective_sample_count"] == len(inputs)
        assert event["request"]["inputs"] == inputs
        assert "Mock error triggered by ERROR keyword" in event["error"]
