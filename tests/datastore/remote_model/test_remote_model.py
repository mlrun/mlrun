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
import inspect
import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pytest
from storey.dtypes import StreamingError

import mlrun
from mlrun.datastore.model_provider.model_provider import UsageResponseKeys
from mlrun.serving.states import ModelRunnerStep
from tests.datastore.remote_model.remote_model_utils import (
    BATCH_INPUT_DATA,
    PROMPT_LEGEND,
    PROMPT_TEMPLATE,
    create_mocked_get_store_artifact,
    setup_remote_model_test,
)

UNIT_TEST_FLUSH_AFTER_SECONDS = 0.7  # Use faster flush for unit tests
UNIT_REQUEST_DELAY_SECONDS = 0.2  # Delay between


class BaseMockModelProviderTest:
    """Base class with common helper methods for MockModelProvider tests"""

    # Error input to trigger MockModelProvider error
    ERROR_INPUT = {
        "question": "ERROR - this should fail",
        "depth_level": "basic",
        "persona": "teacher",
        "tone": "formal",
    }

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

    def _verify_batch_tracking(self, event, inputs=None, model_name="my_endpoint"):
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
        assert event["model"] == model_name
        assert event["metrics"] is None
        assert event["error"] is None

    def _verify_error_tracking(self, event, input_data, model="my_endpoint"):
        """Verify tracking data for error invocation"""
        assert event["request"]["input_schema"] == list(input_data.keys())
        assert event["resp"]["output_schema"] is None
        assert event["resp"]["outputs"] is None
        assert "Mock error triggered by ERROR keyword" in event["error"]
        assert event["model"] == model
        assert event["labels"] == {}
        assert event["metrics"] is None

    def _verify_streaming_response(self, response):
        """Verify that a streaming response is a generator with the expected mock text."""
        assert inspect.isgenerator(
            response
        ), f"Expected generator, got {type(response)}"
        text = "".join(response)
        assert "mock model provider" in text.lower()

    def _verify_single_error_tracking(self, event, input_data):
        """Verify tracking data for single invocation with error"""
        assert event["effective_sample_count"] == 1
        assert event["request"]["inputs"] == [list(input_data.values())]
        self._verify_error_tracking(event, input_data)

    def _verify_batch_error_tracking(self, event, inputs, model="my_endpoint"):
        """Verify tracking data for batch invocation with error"""
        assert event["effective_sample_count"] == len(inputs)
        for i, input_as_list in enumerate(event["request"]["inputs"]):
            assert input_as_list == list(inputs[i].values())
        for invocation_input in inputs:
            self._verify_error_tracking(event, invocation_input, model)

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
        # Should raise RuntimeError with "Mock error triggered" message
        with pytest.raises(
            RuntimeError, match=".*Mock error triggered by ERROR keyword.*"
        ):
            if mlrun_model_name:
                # System test - use function.invoke()
                invoke_func(
                    f"v2/models/{mlrun_model_name}/infer",
                    json.dumps(self.ERROR_INPUT),
                )
            else:
                # Unit test - use server.test()
                invoke_func(body=self.ERROR_INPUT)

    def _check_batch_invocation_with_error(
        self, invoke_func, mlrun_model_name: Optional[str] = None
    ):
        """Helper to test batch invocation with error and verify error is raised"""
        # Append error input to BATCH_INPUT_DATA - the ERROR keyword will trigger mock error
        inputs_with_error = BATCH_INPUT_DATA + [self.ERROR_INPUT]

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

    def _setup_multiple_models_server(
        self,
        project,
        model_url,
        execution_mechanism,
        function_name="test-llm-function",
        batch_step=False,
    ):
        """Helper to set up a server with multiple models for testing

        :param batch_step: If True, adds storey.Batch and FlatMap steps for batch step testing
        """
        # Create model artifact
        model_artifact = project.log_model(
            "model_key",
            model_url=model_url,
        )

        llm_prompt_artifact = project.log_llm_prompt(
            "llm_artifact",
            prompt_template=PROMPT_TEMPLATE,
            description="test llm prompt",
            prompt_legend=PROMPT_LEGEND,
            model_artifact=model_artifact,
        )

        # Create serving function
        function = project.set_function(
            name=function_name,
            kind="serving",
        )
        function.set_tracking("dummy://", enable_tracking=True)

        graph = function.set_topology("flow", engine="async")

        if batch_step:
            # Add batch step for batch step tests
            graph = graph.to(
                "storey.Batch",
                "my_batching",
                max_events=2,
                flush_after_seconds=UNIT_TEST_FLUSH_AFTER_SECONDS,
                full_event=True,
            )

        model_runner_step = ModelRunnerStep(name="my_model_runner")

        # Add first model
        model_runner_step.add_model(
            endpoint_name="my_endpoint",
            model_artifact=llm_prompt_artifact,
            execution_mechanism=execution_mechanism,
            model_class="mlrun.serving.states.LLModel",
            result_path="output",
        )

        # Add second model
        model_runner_step.add_model(
            endpoint_name="my_endpoint_2",
            model_artifact=llm_prompt_artifact,
            execution_mechanism=execution_mechanism,
            model_class="mlrun.serving.states.LLModel",
            result_path="output",
        )

        step = graph.to(model_runner_step)

        if batch_step:
            # FlatMap unpacks batch results back to individual events
            step = step.to("storey.FlatMap", _fn="(event.body)", full_event=True)

        step.respond()

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

        return server


class TestMockModelProviderSingleInvoke(BaseMockModelProviderTest):
    """Tests for single invocation with MockModelProvider"""

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
        self._verify_single_error_tracking(event, self.ERROR_INPUT)


class TestMockModelProviderDirectBatch(BaseMockModelProviderTest):
    """Tests for direct batch invocation (without batch step) with MockModelProvider"""

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_direct_batch(self, execution_mechanism, rundb_mock):
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
    def test_llmodel_direct_batch_with_errors(self, execution_mechanism, rundb_mock):
        """Test that batch processing fails fast when MockModelProvider raises error"""
        project = mlrun.new_project("test-mock-model-batch-errors", save=False)
        model_url = "mock://my-mock-model"

        # Append error input to BATCH_INPUT_DATA - the ERROR keyword will trigger mock error
        inputs = BATCH_INPUT_DATA + [self.ERROR_INPUT]

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

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_direct_batch_multiple_models(
        self, execution_mechanism, rundb_mock
    ):
        """Test batch processing with multiple models using MockModelProvider"""
        project = mlrun.new_project("test-mock-batch-multi", save=False)
        model_url = "mock://my-mock-model"

        server = self._setup_multiple_models_server(
            project, model_url, execution_mechanism, function_name="test-llm-function"
        )

        try:
            # Test batch invocation
            batch_response = server.test(body=BATCH_INPUT_DATA)

            # Response should be dict with model names as keys
            assert isinstance(batch_response, dict)
            assert "my_endpoint" in batch_response
            assert "my_endpoint_2" in batch_response

            # Verify each model's batch response
            for model_name in ["my_endpoint", "my_endpoint_2"]:
                model_batch = batch_response[model_name]
                assert isinstance(model_batch, list)
                assert len(model_batch) == len(BATCH_INPUT_DATA)

                for i, full_result in enumerate(model_batch):
                    result = full_result["output"]
                    self._verify_single_response(result, expect_counter=True)
                    assert f"(Item {i})" in result[UsageResponseKeys.ANSWER]

        finally:
            server.wait_for_completion()

        # Verify tracking data - should have 2 events (one per model)
        dummy_stream = server.context.stream.output_stream
        assert len(dummy_stream.event_list) == 2

        # Verify both model events
        model_events = {event["model"]: event for event in dummy_stream.event_list}
        assert "my_endpoint" in model_events
        assert "my_endpoint_2" in model_events

        for model_name, event in model_events.items():
            self._verify_batch_tracking(
                event, inputs=BATCH_INPUT_DATA, model_name=model_name
            )

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_direct_batch_multiple_models_with_errors(
        self, execution_mechanism, rundb_mock
    ):
        """Test that batch processing with multiple models fails fast when MockModelProvider raises error"""
        project = mlrun.new_project("test-mock-batch-multi-errors", save=False)
        model_url = "mock://my-mock-model"

        # Append error input to BATCH_INPUT_DATA - the ERROR keyword will trigger mock error
        inputs = BATCH_INPUT_DATA + [self.ERROR_INPUT]

        server = self._setup_multiple_models_server(
            project,
            model_url,
            execution_mechanism,
            function_name="test-llm-function-multi-errors",
        )

        try:
            # Test batch invocation with error - should raise RuntimeError
            with pytest.raises(
                RuntimeError, match=".*Mock error triggered by ERROR keyword.*"
            ):
                server.test(body=inputs)
        finally:
            server.wait_for_completion()

        # Verify error was tracked - should have 2 events (one per model)
        dummy_stream = server.context.stream.output_stream
        assert len(dummy_stream.event_list) == 2

        # Verify both model events have error tracking
        model_events = {event["model"]: event for event in dummy_stream.event_list}
        assert "my_endpoint" in model_events
        assert "my_endpoint_2" in model_events

        for model_name, event in model_events.items():
            self._verify_batch_error_tracking(event, inputs, model_name)


class TestMockModelProviderBatchStep(BaseMockModelProviderTest):
    """Tests for batch step (with storey.Batch) with MockModelProvider"""

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_batch_step(self, execution_mechanism, rundb_mock):
        """Test batch processing using storey.Batch step with LLModel and MockModelProvider"""

        project = mlrun.new_project("test-mock-batch-graph", save=False)
        model_url = "mock://my-mock-model"

        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            project,
            model_url,
            execution_mechanism=execution_mechanism,
            batch_step=True,
            flush_after_seconds=UNIT_TEST_FLUSH_AFTER_SECONDS,
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
            # Send events concurrently with staggered timing
            def send_event(event, delay):
                time.sleep(delay)
                return server.test(body=event)

            with ThreadPoolExecutor(max_workers=len(BATCH_INPUT_DATA)) as executor:
                # MockProvider requires a larger delay because batching output depends on the order of requests,
                # which can introduce race conditions, unlike real providers where batching output depends on input.
                futures = [
                    executor.submit(send_event, event, i * UNIT_REQUEST_DELAY_SECONDS)
                    for i, event in enumerate(BATCH_INPUT_DATA)
                ]
                responses = [future.result() for future in futures]
        finally:
            server.wait_for_completion()

        # Verify we got all responses
        assert len(responses) == len(BATCH_INPUT_DATA)

        # Verify each response has correct structure
        for i, response in enumerate(responses):
            assert "output" in response
            output = response["output"]
            self._verify_single_response(output, expect_counter=True)
            # in order to check batches of 2:
            expected_counter = i % 2
            assert f"(Item {expected_counter})" in output[UsageResponseKeys.ANSWER]

        # Verify tracking events - should have 3 batches (2+2+1 = 5 events)
        dummy_stream = server.context.stream.output_stream
        assert len(dummy_stream.event_list) == 3

        # Verify each batch separately using _verify_batch_tracking
        expected_batch_sizes = [2, 2, 1]  # 2+2+1 = 5 events
        start_idx = 0

        for batch_idx in range(3):
            event = dummy_stream.event_list[batch_idx]
            expected_size = expected_batch_sizes[batch_idx]
            end_idx = start_idx + expected_size
            batch_inputs = BATCH_INPUT_DATA[start_idx:end_idx]
            # Use _verify_batch_tracking for each batch
            self._verify_batch_tracking(event, inputs=batch_inputs)
            start_idx = end_idx

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_batch_step_with_errors(self, execution_mechanism, rundb_mock):
        """Test batch step error handling using storey.Batch step with LLModel and MockModelProvider"""

        project = mlrun.new_project("test-mock-batch-graph-errors", save=False)
        model_url = "mock://my-mock-model"

        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            project,
            model_url,
            execution_mechanism=execution_mechanism,
            batch_step=True,
            flush_after_seconds=UNIT_TEST_FLUSH_AFTER_SECONDS,
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
            # Send events concurrently with staggered timing
            def send_event(event, delay):
                time.sleep(delay)
                return server.test(body=event)

            # Send 2 events with one error - both should fail
            good_input = BATCH_INPUT_DATA[0]

            # Send both events in parallel - one good, one bad
            # Both should fail because the batch will fail when processing the error
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(send_event, good_input, 0),
                    executor.submit(
                        send_event, self.ERROR_INPUT, UNIT_REQUEST_DELAY_SECONDS
                    ),
                ]
                # Both should fail when the batch encounters the error
                for future in futures:
                    with pytest.raises(
                        RuntimeError, match="Mock error triggered by ERROR keyword"
                    ):
                        future.result()
        finally:
            server.wait_for_completion()

        # Verify error was tracked
        dummy_stream = server.context.stream.output_stream
        assert len(dummy_stream.event_list) == 1

        error_event = dummy_stream.event_list[0]
        error_inputs = [good_input, self.ERROR_INPUT]
        self._verify_batch_error_tracking(error_event, error_inputs)

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_batch_step_multiple_models(self, execution_mechanism, rundb_mock):
        """Test batch processing using storey.Batch step with multiple LLModels and MockModelProvider"""

        project = mlrun.new_project("test-mock-batch-graph-multiple", save=False)
        model_url = "mock://my-mock-model"

        server = self._setup_multiple_models_server(
            project,
            model_url,
            execution_mechanism,
            function_name="test-llm-function-batch-multi",
            batch_step=True,
        )

        try:
            # Send events concurrently with staggered timing
            def send_event(event, delay):
                time.sleep(delay)
                return server.test(body=event)

            with ThreadPoolExecutor(max_workers=len(BATCH_INPUT_DATA)) as executor:
                # MockProvider requires a larger delay because batching output depends on the order of requests,
                # which can introduce race conditions, unlike real providers where batching output depends on input.
                futures = [
                    executor.submit(send_event, event, i * UNIT_REQUEST_DELAY_SECONDS)
                    for i, event in enumerate(BATCH_INPUT_DATA)
                ]
                responses = [future.result() for future in futures]

        finally:
            server.wait_for_completion()

        # Verify we got all responses
        assert len(responses) == len(BATCH_INPUT_DATA)

        # Verify each response has both models' outputs organized by model name
        for i, response in enumerate(responses):
            # Response should be dict with model names as keys
            assert isinstance(response, dict)
            assert "my_endpoint" in response
            assert "my_endpoint_2" in response

            # Verify first model output
            output = response["my_endpoint"]["output"]
            self._verify_single_response(output, expect_counter=True)
            expected_counter = i % 2
            assert f"(Item {expected_counter})" in output[UsageResponseKeys.ANSWER]

            # Verify second model output
            output_2 = response["my_endpoint_2"]["output"]
            self._verify_single_response(output_2, expect_counter=True)
            assert f"(Item {expected_counter})" in output_2[UsageResponseKeys.ANSWER]

        # Verify tracking events - should have 6 events (2 models × 3 batches)
        dummy_stream = server.context.stream.output_stream
        assert len(dummy_stream.event_list) == 6

        # Separate events by model
        model_events = {"my_endpoint": [], "my_endpoint_2": []}
        for event in dummy_stream.event_list:
            model_name = event["model"]
            model_events[model_name].append(event)

        # Verify each model has 3 batches
        assert len(model_events["my_endpoint"]) == 3
        assert len(model_events["my_endpoint_2"]) == 3

        # Verify first 3 batches for each model
        expected_batch_sizes = [2, 2, 1]  # 2+2+1 = 5 events
        for model_name in ["my_endpoint", "my_endpoint_2"]:
            start_idx = 0
            for batch_idx in range(3):
                event = model_events[model_name][batch_idx]
                expected_size = expected_batch_sizes[batch_idx]
                end_idx = start_idx + expected_size
                batch_inputs = BATCH_INPUT_DATA[start_idx:end_idx]
                # Use _verify_batch_tracking for each batch
                self._verify_batch_tracking(
                    event, inputs=batch_inputs, model_name=model_name
                )
                start_idx = end_idx

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "asyncio", "thread_pool"],
    )
    def test_llmodel_batch_step_multiple_models_with_errors(
        self, execution_mechanism, rundb_mock
    ):
        """Test batch step error handling with multiple models using storey.Batch step"""

        project = mlrun.new_project("test-mock-batch-graph-multiple-errors", save=False)
        model_url = "mock://my-mock-model"

        server = self._setup_multiple_models_server(
            project,
            model_url,
            execution_mechanism,
            function_name="test-llm-function-batch-multi-errors",
            batch_step=True,
        )

        try:
            # Send events concurrently with staggered timing
            def send_event(event, delay):
                time.sleep(delay)
                return server.test(body=event)

            # Send 2 events with one error in a batch
            good_input = BATCH_INPUT_DATA[0]

            # Send both events in parallel - one good, one bad
            # Both should fail because the batch will fail when processing the error
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(send_event, good_input, 0),
                    executor.submit(
                        send_event, self.ERROR_INPUT, UNIT_REQUEST_DELAY_SECONDS
                    ),
                ]
                # Both should fail when the batch encounters the error
                for future in futures:
                    with pytest.raises(
                        RuntimeError, match="Mock error triggered by ERROR keyword"
                    ):
                        future.result()

        finally:
            server.wait_for_completion()

        # Verify tracking events - should have 2 events (1 error per model)
        dummy_stream = server.context.stream.output_stream
        assert len(dummy_stream.event_list) == 2

        # Separate events by model
        model_events = {event["model"]: event for event in dummy_stream.event_list}
        assert "my_endpoint" in model_events
        assert "my_endpoint_2" in model_events

        # Verify the error batch for both models
        error_inputs = [good_input, self.ERROR_INPUT]
        for model_name in ["my_endpoint", "my_endpoint_2"]:
            error_event = model_events[model_name]
            self._verify_batch_error_tracking(error_event, error_inputs, model_name)

    @pytest.mark.parametrize("multiple_models", (True, False))
    def test_llmodel_batch_step_multiple_models_error_in_dict(
        self, multiple_models, rundb_mock
    ):
        """
        Test batch step with single/multiple models where errors are returned in response dict (raise_exception=False).
        Only testing with naive execution mechanism (no need to test all mechanisms again).
        """
        project = mlrun.new_project("test-mock-batch-graph-error-dict", save=False)
        model_url = "mock://my-mock-model"

        # Create model artifact
        model_artifact = project.log_model(
            "model_key",
            model_url=model_url,
        )

        llm_prompt_artifact = project.log_llm_prompt(
            "llm_artifact",
            prompt_template=PROMPT_TEMPLATE,
            description="test llm prompt",
            prompt_legend=PROMPT_LEGEND,
            model_artifact=model_artifact,
        )

        # Create serving function
        function = project.set_function(
            name="test-llm-function-batch-error-dict",
            kind="serving",
        )
        function.set_tracking("dummy://", enable_tracking=True)

        graph = function.set_topology("flow", engine="async")

        # Add batch step
        graph = graph.to(
            "storey.Batch",
            "my_batching",
            max_events=2,
            flush_after_seconds=UNIT_TEST_FLUSH_AFTER_SECONDS,
            full_event=True,
        )

        # ModelRunnerStep with raise_exception=False to return errors in dict
        model_runner_step = ModelRunnerStep(
            name="my_model_runner", raise_exception=False
        )

        # Add first model
        model_runner_step.add_model(
            endpoint_name="my_endpoint",
            model_artifact=llm_prompt_artifact,
            execution_mechanism="naive",  # Only testing naive
            model_class="mlrun.serving.states.LLModel",
            result_path="output",
        )

        # Add second model if testing multiple models
        if multiple_models:
            model_runner_step.add_model(
                endpoint_name="my_endpoint_2",
                model_artifact=llm_prompt_artifact,
                execution_mechanism="naive",  # Only testing naive
                model_class="mlrun.serving.states.LLModel",
                result_path="output",
            )

        step = graph.to(model_runner_step)
        step = step.to("storey.FlatMap", _fn="(event.body)", full_event=True)
        step.respond()

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
            # Send events concurrently with staggered timing
            def send_event(event, delay):
                time.sleep(delay)
                return server.test(body=event)

            # Send 2 events - one good, one with error
            good_input = BATCH_INPUT_DATA[0]

            # Send both events in parallel - one good, one bad
            # Error should be returned in dict, not raised
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(send_event, good_input, 0),
                    executor.submit(
                        send_event, self.ERROR_INPUT, UNIT_REQUEST_DELAY_SECONDS
                    ),
                ]
                responses = [future.result() for future in futures]

        finally:
            server.wait_for_completion()

        # Verify we got both responses (not exceptions)
        assert len(responses) == 2

        # All responses should contain error field
        if multiple_models:
            model_names = ["my_endpoint", "my_endpoint_2"]
            for response in responses:
                assert isinstance(response, dict)
                assert all(
                    "error" in response.get(model, {}) for model in model_names
                ), f"Expected error field for each model in response, got {response}"

                # Verify error message for each model
                for model_name in model_names:
                    assert (
                        "Mock error triggered by ERROR keyword"
                        in response[model_name]["error"]
                    )
        else:
            # Single model - error should be in response body directly
            for response in responses:
                assert isinstance(response, dict)
                assert (
                    "error" in response
                ), f"Expected error field in response, got {response}"
                assert "Mock error triggered by ERROR keyword" in response["error"]

        # Verify tracking events
        dummy_stream = server.context.stream.output_stream
        num_models = 2 if multiple_models else 1
        assert len(dummy_stream.event_list) == num_models

        # Verify the error batch for each model
        error_inputs = [good_input, self.ERROR_INPUT]
        if multiple_models:
            # Separate events by model
            model_events = {event["model"]: event for event in dummy_stream.event_list}
            assert "my_endpoint" in model_events
            assert "my_endpoint_2" in model_events

            for model_name in ["my_endpoint", "my_endpoint_2"]:
                error_event = model_events[model_name]
                self._verify_batch_error_tracking(
                    error_event, error_inputs, model=model_name
                )
        else:
            # Single model
            error_event = dummy_stream.event_list[0]
            self._verify_batch_error_tracking(error_event, error_inputs)


class TestMockModelProviderStreaming(BaseMockModelProviderTest):
    """Tests for streaming invocation with MockModelProvider through MRS."""

    def test_llmodel_streaming(self, rundb_mock):
        """Test streaming invocation through MRS with MockModelProvider."""
        project = mlrun.new_project("test-mock-streaming", save=False)
        model_url = "mock://my-mock-model"

        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            project,
            model_url,
            execution_mechanism="asyncio",
            streaming=True,
        )

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
            response = server.test(body=BATCH_INPUT_DATA[0])
            self._verify_streaming_response(response)
        finally:
            server.wait_for_completion()

    def test_llmodel_streaming_error(self, rundb_mock):
        """Test streaming error tracking through MRS with MockModelProvider."""
        project = mlrun.new_project("test-mock-streaming-error", save=False)
        model_url = "mock://my-mock-model"

        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            project,
            model_url,
            execution_mechanism="asyncio",
            streaming=True,
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
            response = server.test(body=self.ERROR_INPUT)
            with pytest.raises(
                StreamingError, match="Mock error triggered by ERROR keyword"
            ):
                list(response)
        finally:
            server.wait_for_completion()

        dummy_stream = server.context.stream.output_stream
        assert len(dummy_stream.event_list) == 1
        event = dummy_stream.event_list[0]
        self._verify_single_error_tracking(event, self.ERROR_INPUT)
