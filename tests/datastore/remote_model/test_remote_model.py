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
    def test_llmodel_single_invocation(self, execution_mechanism, rundb_mock):
        """Test single invocation with MockModelProvider"""
        project = mlrun.new_project("test-mock-model-single", save=False)
        model_url = "mock://my-mock-model"

        # Single input
        input_data = INPUT_DATA[0]

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
        with unittest.mock.patch(
            "mlrun.artifacts.llm_prompt.mlrun.datastore.store_manager.get_store_artifact",
            side_effect=lambda *args, **kwargs: mocked_get_store_artifact(
                *args, **kwargs
            ),
        ):
            server = function.to_mock_server()

        try:
            # Send single input
            response = server.test(body=input_data)

            # Assert we got single response (not a list)
            assert isinstance(response, dict)
            result = response["output"]
            assert len(result) == 2  # answer + usage

            # Get answer and usage
            answer = result[UsageResponseKeys.ANSWER]
            stats = result[UsageResponseKeys.USAGE]

            # Verify mock message (no counter for single invocation)
            assert "mock model provider" in answer.lower()
            assert "(Item" not in answer  # No counter for single invocation

            # Verify mock usage stats (should be 0)
            assert stats["prompt_tokens"] == 0
            assert stats["completion_tokens"] == 0
            assert stats["total_tokens"] == 0
        finally:
            server.wait_for_completion()

        # Verify tracking data
        dummy_stream = server.context.stream.output_stream
        event = dummy_stream.event_list[0]
        assert event["effective_sample_count"] == 1
        assert event["request"]["input_schema"] == list(input_data.keys())
        assert event["request"]["inputs"] == [list(input_data.values())]
        assert event["resp"]["output_schema"] == UsageResponseKeys.fields()
        assert len(event["resp"]["outputs"]) == 1
        output = event["resp"]["outputs"][0]
        assert output[0] == answer
        assert output[1] == stats
        assert event["labels"] == {}
        assert event["model"] == "my_endpoint"
        assert event["error"] is None
        assert event["metrics"] is None

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_single_invocation_with_error(
        self, execution_mechanism, rundb_mock
    ):
        """Test single invocation with error using MockModelProvider"""
        project = mlrun.new_project("test-mock-model-single-error", save=False)
        model_url = "mock://my-mock-model"

        # Single input with ERROR keyword
        input_data = {
            "question": "ERROR - this should fail",
            "depth_level": "basic",
            "persona": "teacher",
            "tone": "formal",
        }

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
                server.test(body=input_data)
        finally:
            server.wait_for_completion()

        # Verify error was tracked
        dummy_stream = server.context.stream.output_stream
        event = dummy_stream.event_list[0]
        assert event["effective_sample_count"] == 1
        assert event["request"]["input_schema"] == list(input_data.keys())
        assert event["request"]["inputs"] == [list(input_data.values())]
        assert event["resp"]["output_schema"] is None
        assert event["resp"]["outputs"] == [None]
        assert "Mock error triggered by ERROR keyword" in event["error"]
        assert event["model"] == "my_endpoint"
        assert event["labels"] == {}
        assert event["metrics"] is None

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
        assert event["request"]["input_schema"] == list(INPUT_DATA[0].keys())
        for i, input_as_list in enumerate(event["request"]["inputs"]):
            assert input_as_list == list(INPUT_DATA[i].values())
        assert event["resp"]["output_schema"] == UsageResponseKeys.fields()
        for i, resp in enumerate(event["resp"]["outputs"]):
            answer = resp[0]
            usage = resp[1]
            assert f"(Item {i})" in answer
            assert "mock model provider" in answer.lower()
            assert usage["prompt_tokens"] == 0
            assert usage["completion_tokens"] == 0
            assert usage["total_tokens"] == 0
        assert event["labels"] == {}
        assert event["model"] == "my_endpoint"
        assert event["metrics"] is None

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_batch_with_errors(self, execution_mechanism, rundb_mock):
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
            execution_mechanism=execution_mechanism,
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
        assert event["request"]["input_schema"] == list(inputs[0].keys())
        for i, input_as_list in enumerate(event["request"]["inputs"]):
            assert input_as_list == list(inputs[i].values())
        assert event["resp"]["output_schema"] is None
        assert event["resp"]["outputs"] == [None]
        assert "Mock error triggered by ERROR keyword" in event["error"]
        assert event["model"] == "my_endpoint"
        assert event["labels"] == {}
        assert event["metrics"] is None
