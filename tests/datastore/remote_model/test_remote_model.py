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
import json
import unittest
from typing import Optional

import pytest

import mlrun
from mlrun.datastore.model_provider.model_provider import UsageResponseKeys
from tests.datastore.remote_model.remote_model_utils import (
    BATCH_INPUT_DATA,
    create_mocked_get_store_artifact,
    setup_remote_model_test,
)


class BaseMockModelProviderTest:
    """Base class with common helper methods for MockModelProvider tests"""

    def _verify_single_response(self, response, expect_counter=False):
        """Verify structure and content of single invocation response"""
        assert len(response) == 2  # answer + usage
        answer = response[UsageResponseKeys.ANSWER]
        stats = response[UsageResponseKeys.USAGE]

        # Verify mock message (no counter for single invocation)
        assert "mock model provider" in answer.lower()
        if expect_counter:
            assert "(Item" in answer
        else:
            assert "(Item" not in answer
        # Verify mock usage stats (should be 0)
        assert stats["prompt_tokens"] == 0
        assert stats["completion_tokens"] == 0
        assert stats["total_tokens"] == 0

    def _verify_batch_response(self, batch_response):
        """Verify structure and content of batch invocation responses"""
        # Assert we got list of responses
        assert isinstance(batch_response, list)
        assert len(batch_response) == len(BATCH_INPUT_DATA)

        # Verify each response has correct structure
        for i, full_result in enumerate(batch_response):
            result = full_result["output"]
            # Use single response verification
            self._verify_single_response(result, expect_counter=True)
            # Additionally verify batch-specific: item index in answer

    def _verify_single_tracking_output(self, output):
        """Verify structure and content of a single tracking output"""
        assert "mock model provider" in output[0]  # answer
        assert output[1]["prompt_tokens"] == 0
        assert output[1]["completion_tokens"] == 0
        assert output[1]["total_tokens"] == 0

    def _verify_single_tracking(self, event, input_data):
        """Verify tracking data for single invocation"""
        assert event["effective_sample_count"] == 1
        assert event["request"]["input_schema"] == list(input_data.keys())
        assert event["request"]["inputs"] == [list(input_data.values())]
        assert event["resp"]["output_schema"] == UsageResponseKeys.fields()
        assert len(event["resp"]["outputs"]) == 1
        self._verify_single_tracking_output(event["resp"]["outputs"][0])
        assert event["labels"] == {}
        assert event["model"] == "my_endpoint"
        assert event["error"] is None
        assert event["metrics"] is None

    def _verify_batch_tracking(self, event, inputs=None):
        """Verify tracking data for batch invocation"""
        if inputs is None:
            inputs = BATCH_INPUT_DATA

        assert event["effective_sample_count"] == len(inputs)
        assert event["request"]["input_schema"] == list(inputs[0].keys())
        for i, input_as_list in enumerate(event["request"]["inputs"]):
            assert input_as_list == list(inputs[i].values())
        assert event["resp"]["output_schema"] == UsageResponseKeys.fields()
        for i, resp in enumerate(event["resp"]["outputs"]):
            # Verify item index in answer (batch-specific)
            assert f"(Item {i})" in resp[0]
            # Use single tracking output verification for common checks
            self._verify_single_tracking_output(resp)
        assert event["labels"] == {}
        assert event["model"] == "my_endpoint"
        assert event["metrics"] is None

    def _verify_error_tracking(self, event, input_data):
        """Verify tracking data for error invocation"""
        assert event["request"]["input_schema"] == list(input_data.keys())
        assert event["resp"]["output_schema"] is None
        assert event["resp"]["outputs"] == [None]
        assert "Mock error triggered by ERROR keyword" in event["error"]
        assert event["model"] == "my_endpoint"
        assert event["labels"] == {}
        assert event["metrics"] is None

    def _verify_single_error_tracking(self, event, input_data):
        """Verify tracking data for single invocation with error"""
        assert event["effective_sample_count"] == 1
        assert event["request"]["inputs"] == [list(input_data.values())]
        self._verify_error_tracking(event, input_data)

    def _verify_batch_error_tracking(self, event, inputs):
        """Verify tracking data for batch invocation with error"""
        assert event["effective_sample_count"] == len(inputs)
        for i, input_as_list in enumerate(event["request"]["inputs"]):
            assert input_as_list == list(inputs[i].values())
        self._verify_error_tracking(event, inputs[0])

    def _check_single_invocation(
        self, invoke_func, mlrun_model_name: Optional[str] = None
    ):
        """Helper to test single invocation and verify response"""
        if mlrun_model_name:
            # System test - use function.invoke()
            response = invoke_func(
                f"v2/models/{mlrun_model_name}/infer",
                json.dumps(BATCH_INPUT_DATA[0]),
            )["output"]
        else:
            # Unit test - use server.test()
            response = invoke_func(body=BATCH_INPUT_DATA[0])
            assert isinstance(response, dict)
            response = response["output"]

        self._verify_single_response(response)

    def _check_batch_invocation(
        self, invoke_func, mlrun_model_name: Optional[str] = None
    ):
        """Helper to test batch invocation and verify responses"""
        if mlrun_model_name:
            # System test - use function.invoke()
            batch_response = invoke_func(
                f"v2/models/{mlrun_model_name}/infer",
                json.dumps(BATCH_INPUT_DATA),
            )
        else:
            # Unit test - use server.test()
            batch_response = invoke_func(body=BATCH_INPUT_DATA)

        self._verify_batch_response(batch_response)

    def _check_single_invocation_with_error(
        self, invoke_func, mlrun_model_name: Optional[str] = None
    ):
        """Helper to test single invocation with error and verify error is raised"""
        # Single input with ERROR keyword to trigger mock error
        error_input = {
            "question": "ERROR - this should fail",
            "depth_level": "basic",
            "persona": "teacher",
            "tone": "formal",
        }

        # Should raise RuntimeError with "Mock error triggered" message
        with pytest.raises(
            RuntimeError, match=".*Mock error triggered by ERROR keyword.*"
        ):
            if mlrun_model_name:
                # System test - use function.invoke()
                invoke_func(
                    f"v2/models/{mlrun_model_name}/infer",
                    json.dumps(error_input),
                )
            else:
                # Unit test - use server.test()
                invoke_func(body=error_input)

    def _check_batch_invocation_with_error(
        self, invoke_func, mlrun_model_name: Optional[str] = None
    ):
        """Helper to test batch invocation with error and verify error is raised"""
        # Append error input to BATCH_INPUT_DATA - the ERROR keyword will trigger mock error
        inputs_with_error = BATCH_INPUT_DATA + [
            {
                "question": "ERROR - this should fail",
                "depth_level": "basic",
                "persona": "teacher",
                "tone": "formal",
            }
        ]

        # Should raise RuntimeError with "Mock error triggered" message
        with pytest.raises(
            RuntimeError, match=".*Mock error triggered by ERROR keyword.*"
        ):
            if mlrun_model_name:
                # System test - use function.invoke()
                invoke_func(
                    f"v2/models/{mlrun_model_name}/infer",
                    json.dumps(inputs_with_error),
                )
            else:
                # Unit test - use server.test()
                invoke_func(body=inputs_with_error)


class TestMockModelProvider(BaseMockModelProviderTest):
    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_single_invocation(self, execution_mechanism, rundb_mock):
        """Test single invocation with MockModelProvider"""
        project = mlrun.new_project("test-mock-model-single", save=False)
        model_url = "mock://my-mock-model"

        # Single input
        input_data = BATCH_INPUT_DATA[0]

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
            # Test single invocation and get answer/stats
            self._check_single_invocation(server.test)
        finally:
            server.wait_for_completion()

        # Verify tracking data
        dummy_stream = server.context.stream.output_stream
        event = dummy_stream.event_list[0]
        self._verify_single_tracking(event, input_data)

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
        error_input = {
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
            # Test single invocation with error
            self._check_single_invocation_with_error(server.test)
        finally:
            server.wait_for_completion()

        # Verify error was tracked
        dummy_stream = server.context.stream.output_stream
        event = dummy_stream.event_list[0]
        self._verify_single_error_tracking(event, error_input)

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
            # Test batch invocation
            self._check_batch_invocation(server.test)
        finally:
            server.wait_for_completion()

        # Verify tracking data
        dummy_stream = server.context.stream.output_stream
        event = dummy_stream.event_list[0]
        self._verify_batch_tracking(event)

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_batch_with_errors(self, execution_mechanism, rundb_mock):
        """Test that batch processing fails fast when MockModelProvider raises error"""
        project = mlrun.new_project("test-mock-model-batch-errors", save=False)
        model_url = "mock://my-mock-model"

        # Append error input to BATCH_INPUT_DATA - the ERROR keyword will trigger mock error
        inputs = BATCH_INPUT_DATA + [
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
            # Test batch invocation with error
            self._check_batch_invocation_with_error(server.test)
        finally:
            server.wait_for_completion()

        # Verify error was tracked
        dummy_stream = server.context.stream.output_stream
        event = dummy_stream.event_list[0]
        self._verify_batch_error_tracking(event, inputs)
