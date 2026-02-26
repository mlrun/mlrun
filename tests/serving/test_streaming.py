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

"""Tests for streaming support in serving graphs."""

import inspect
import unittest.mock

import pytest
from storey.dtypes import StreamingError

import mlrun
import mlrun.errors
from mlrun.datastore.model_provider.mock_model_provider import MockModelProvider
from mlrun.datastore.model_provider.model_provider import ModelProvider
from mlrun.runtimes.nuclio.serving import ServingSpec
from mlrun.serving import Model
from mlrun.serving.server import (
    v2_serving_handler,
    v2_serving_streaming_handler,
)
from mlrun.serving.states import LLModel


class TestServingSpecStreaming:
    """Tests for streaming attribute in ServingSpec."""

    def test_streaming_default_none(self):
        """Test that streaming is None by default."""
        function = mlrun.new_function("test", kind="serving")
        assert function.spec.streaming is None

    def test_streaming_in_dict_fields(self):
        """Test that streaming is included in _dict_fields for serialization."""
        assert "streaming" in ServingSpec._dict_fields

    def test_streaming_serialization(self):
        """Test that streaming setting is properly serialized/deserialized."""
        function = mlrun.new_function("test", kind="serving")
        function.set_topology("flow", engine="async")
        function.set_streaming(enabled=True)

        # Serialize and deserialize
        func_dict = function.to_dict()
        restored = mlrun.new_function(runtime=func_dict)

        assert restored.spec.streaming is True


class TestSetStreaming:
    """Tests for ServingRuntime.set_streaming() method."""

    def test_set_streaming_enabled(self):
        """Test enabling streaming mode."""
        function = mlrun.new_function("test", kind="serving")
        function.set_topology("flow", engine="async")
        function.set_streaming(enabled=True)

        assert function.spec.streaming is True

    def test_set_streaming_disabled(self):
        """Test disabling streaming mode."""
        function = mlrun.new_function("test", kind="serving")
        function.set_topology("flow", engine="async")
        function.set_streaming(enabled=True)
        function.set_streaming(enabled=False)

        assert function.spec.streaming is False

    def test_set_streaming_validates_existing_triggers(self):
        """Test that set_streaming validates existing non-HTTP triggers."""
        function = mlrun.new_function("test", kind="serving")
        function.set_topology("flow", engine="async")

        # Add a non-HTTP trigger
        function.spec.config["spec.triggers.my_kafka"] = {
            "kind": "kafka",
            "url": "kafka://localhost:9092",
        }

        # Should raise error when trying to enable streaming
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Streaming is only supported with HTTP triggers",
        ):
            function.set_streaming(enabled=True)

    def test_set_streaming_allows_http_triggers(self):
        """Test that set_streaming allows HTTP triggers."""
        function = mlrun.new_function("test", kind="serving")
        function.set_topology("flow", engine="async")

        # Add an HTTP trigger
        function.spec.config["spec.triggers.my_http"] = {
            "kind": "http",
            "port": 8080,
        }

        # Should not raise error
        function.set_streaming(enabled=True)
        assert function.spec.streaming is True


class TestAddTriggerWithStreaming:
    """Tests for trigger validation when streaming is enabled."""

    def test_add_non_http_trigger_when_streaming_enabled(self):
        """Test that adding non-HTTP trigger fails when streaming is enabled."""
        function = mlrun.new_function("test", kind="serving")
        function.set_topology("flow", engine="async")
        function.set_streaming(enabled=True)

        # Try to add a Kafka trigger
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Cannot add non-HTTP trigger",
        ):
            function.add_trigger(
                "my_kafka", {"kind": "kafka", "url": "kafka://localhost"}
            )

    def test_add_http_trigger_when_streaming_enabled(self):
        """Test that adding HTTP trigger works when streaming is enabled."""
        function = mlrun.new_function("test", kind="serving")
        function.set_topology("flow", engine="async")
        function.set_streaming(enabled=True)

        # Should not raise error
        function.add_trigger("my_http", {"kind": "http", "port": 8080})

    def test_add_non_http_trigger_when_streaming_disabled(self):
        """Test that adding non-HTTP trigger works when streaming is disabled."""
        function = mlrun.new_function("test", kind="serving")
        function.set_topology("flow", engine="async")
        # streaming is None/False by default

        # Should not raise error
        function.add_trigger("my_kafka", {"kind": "kafka", "url": "kafka://localhost"})


class TestStreamingToJob:
    """Tests for to_job() validation with streaming."""

    def test_to_job_fails_when_streaming_enabled(self):
        """Test that to_job() fails when streaming is enabled."""
        function = mlrun.new_function("test", kind="serving")
        function.set_topology("flow", engine="async")
        function.set_streaming(enabled=True)

        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="streaming is enabled",
        ):
            function.to_job()

    def test_to_job_works_when_streaming_disabled(self):
        """Test that to_job() works when streaming is disabled."""
        function = mlrun.new_function("test", kind="serving")
        function.set_topology("flow", engine="async")
        # streaming is None/False by default

        # Should not raise error
        job = function.to_job()
        assert job


class TestStreamingHandler:
    """Tests for v2_serving_streaming_handler."""

    def test_streaming_handler_is_async_generator(self):
        """Test that v2_serving_streaming_handler is an async generator function."""
        assert inspect.isasyncgenfunction(v2_serving_streaming_handler)

    def test_regular_handler_is_not_generator(self):
        """Test that v2_serving_handler is not a generator function."""
        assert not inspect.isgeneratorfunction(v2_serving_handler)


class StreamingStep:
    """A step that yields streaming chunks."""

    def __init__(self, context=None, name=None, num_chunks=3):
        self.context = context
        self.name = name
        self.num_chunks = num_chunks

    def do(self, x):
        """Yield multiple chunks for a single input."""
        for i in range(self.num_chunks):
            yield f"{x}_chunk_{i}"


class NonStreamingStep:
    """A regular non-streaming step."""

    def __init__(self, context=None, name=None):
        self.context = context
        self.name = name

    def do(self, x):
        """Return a single result."""
        return f"{x}_processed"


class DoubleStreamer:
    """A streaming step for testing streaming-on-streaming errors."""

    def __init__(self, context=None, name=None):
        self.context = context
        self.name = name

    def do(self, x):
        yield f"{x}_a"
        yield f"{x}_b"


class ReStreamer:
    """A streaming step that re-streams collected chunks."""

    def __init__(self, context=None, name=None):
        self.context = context
        self.name = name

    def do(self, x):
        # x is now a list from collector
        for item in x:
            yield f"re_{item}"


class ErrorStreamingStep:
    """A streaming step that raises an error mid-stream."""

    def __init__(self, context=None, name=None):
        self.context = context
        self.name = name

    def do(self, x):
        yield f"{x}_chunk_0"
        raise ValueError("Generator error mid-stream")


class FailingIntermediateStep:
    """A non-streaming step that fails on specific input."""

    def __init__(self, context=None, name=None, fail_on_chunk=1):
        self.context = context
        self.name = name
        self.fail_on_chunk = fail_on_chunk
        self._count = 0

    def do(self, x):
        if self._count == self.fail_on_chunk:
            raise RuntimeError(f"Failed on chunk {self._count}")
        self._count += 1
        return f"{x}_processed"


class TestStreamingEndToEnd:
    """End-to-end tests for streaming in serving graphs."""

    def test_streaming_step_produces_multiple_results(self):
        """Test that a streaming step produces multiple results collected by Reduce."""
        function = mlrun.new_function("test", kind="serving")
        graph = function.set_topology("flow", engine="async")

        # Add a streaming step followed by a collector
        graph.to(
            name="streamer", class_name="tests.serving.test_streaming.StreamingStep"
        )
        graph.add_step(
            name="collector",
            class_name="storey.Collector",
            after="streamer",
        ).respond()

        server = function.to_mock_server()
        try:
            # Test with mock server
            result = server.test("/", body="test")

            # The collector should aggregate all chunks into a list
            assert isinstance(result, list)
            assert len(result) == 3
            assert result == ["test_chunk_0", "test_chunk_1", "test_chunk_2"]
        finally:
            server.wait_for_completion()

    def test_non_streaming_step_passthrough(self):
        """Test that non-streaming steps work normally."""
        function = mlrun.new_function("test", kind="serving")
        graph = function.set_topology("flow", engine="async")

        graph.to(
            name="processor", class_name="tests.serving.test_streaming.NonStreamingStep"
        ).respond()

        server = function.to_mock_server()
        try:
            result = server.test("/", body="test")
            assert result == "test_processed"
        finally:
            server.wait_for_completion()

    def test_streaming_through_intermediate_steps(self):
        """Test that streaming chunks flow through intermediate non-streaming steps."""
        function = mlrun.new_function("test", kind="serving")
        graph = function.set_topology("flow", engine="async")

        # streaming step -> non-streaming step -> collector
        graph.to(
            name="streamer", class_name="tests.serving.test_streaming.StreamingStep"
        )
        graph.add_step(
            name="processor",
            class_name="tests.serving.test_streaming.NonStreamingStep",
            after="streamer",
        )
        graph.add_step(
            name="collector",
            class_name="storey.Collector",
            after="processor",
        ).respond()

        server = function.to_mock_server()
        try:
            result = server.test("/", body="test")

            # Each chunk should be processed by the non-streaming step
            assert isinstance(result, list)
            assert len(result) == 3
            # Each chunk was processed: "test_chunk_X" -> "test_chunk_X_processed"
            assert result == [
                "test_chunk_0_processed",
                "test_chunk_1_processed",
                "test_chunk_2_processed",
            ]
        finally:
            server.wait_for_completion()


class TestStreamingErrors:
    """Tests for streaming error conditions."""

    def test_streaming_on_streaming_raises_error(self):
        """Test that streaming on top of streaming raises an error."""
        function = mlrun.new_function("test", kind="serving")
        graph = function.set_topology("flow", engine="async")

        # Two streaming steps in sequence without collector
        graph.to(
            name="streamer1", class_name="tests.serving.test_streaming.StreamingStep"
        )
        graph.add_step(
            name="streamer2",
            class_name="tests.serving.test_streaming.DoubleStreamer",
            after="streamer1",
        ).respond()

        server = function.to_mock_server()
        try:
            # The mock server catches StreamingError and re-raises as RuntimeError
            with pytest.raises(RuntimeError, match="Streaming on top of streaming"):
                server.test("/", body="test")
        finally:
            server.wait_for_completion()

    def test_streaming_after_collector_allowed(self):
        """Test that streaming after a Collector is allowed."""
        function = mlrun.new_function("test", kind="serving")
        graph = function.set_topology("flow", engine="async")

        # streaming -> collector -> streaming -> collector
        graph.to(
            name="streamer1",
            class_name="tests.serving.test_streaming.StreamingStep",
            num_chunks=2,
        )
        graph.add_step(
            name="collector1",
            class_name="storey.Collector",
            after="streamer1",
        )
        graph.add_step(
            name="restreamer",
            class_name="ReStreamer",
            after="collector1",
        )
        graph.add_step(
            name="collector2",
            class_name="storey.Collector",
            after="restreamer",
        ).respond()

        server = function.to_mock_server()
        try:
            result = server.test("/", body="test")

            # First stream: ["test_chunk_0", "test_chunk_1"]
            # Re-streamed: ["re_test_chunk_0", "re_test_chunk_1"]
            assert result == ["re_test_chunk_0", "re_test_chunk_1"]
        finally:
            server.wait_for_completion()

    def test_streaming_generator_raises_error(self):
        """Test that error in generator mid-stream propagates without hanging."""
        function = mlrun.new_function("test", kind="serving")
        graph = function.set_topology("flow", engine="async")

        # Streaming step that raises an error after yielding one chunk
        graph.to(
            name="error_streamer",
            class_name="ErrorStreamingStep",
        ).respond()

        server = function.to_mock_server()
        try:
            result = server.test("/", body="test")

            # Result should be a generator
            assert inspect.isgenerator(result), "Expected generator result"

            # Collect chunks until error - consumer gets StreamingError wrapping the message
            chunks = []
            with pytest.raises(StreamingError, match="Generator error mid-stream"):
                for chunk in result:
                    chunks.append(chunk)

            # Verify first chunk was received before error
            assert chunks == ["test_chunk_0"]
        finally:
            server.wait_for_completion()

    def test_streaming_error_in_intermediate_step(self):
        """Test that error in non-streaming step processing chunks propagates correctly."""
        function = mlrun.new_function("test", kind="serving")
        graph = function.set_topology("flow", engine="async")

        # streaming step -> failing intermediate step -> collector
        graph.to(name="streamer", class_name="StreamingStep")
        graph.add_step(
            name="failing_step",
            class_name="FailingIntermediateStep",
            after="streamer",
        )
        graph.add_step(
            name="collector",
            class_name="storey.Collector",
            after="failing_step",
        ).respond()

        server = function.to_mock_server()
        try:
            # The error should propagate
            with pytest.raises(RuntimeError, match="Failed on chunk 1"):
                server.test("/", body="test")
        finally:
            server.wait_for_completion()

    def test_streaming_error_produces_error_event_through_collector(self):
        """Test that a streaming generator error produces an error dict event via Collector.

        When a streaming step errors mid-stream, the Collector should emit
        an event with body={"error": "..."} matching non-streaming error format.
        """
        function = mlrun.new_function("test", kind="serving")
        graph = function.set_topology("flow", engine="async")

        graph.to(
            name="error_streamer",
            class_name="ErrorStreamingStep",
        )
        graph.add_step(
            name="collector",
            class_name="storey.Collector",
            after="error_streamer",
        ).respond()

        server = function.to_mock_server()
        try:
            resp = server.test("/", body="test")
            assert isinstance(resp, dict), f"Expected error dict, got {type(resp)}"
            assert "error" in resp, f"Expected 'error' key in response: {resp}"
            assert "ValueError" in resp["error"]
            assert "Generator error mid-stream" in resp["error"]
        finally:
            server.wait_for_completion()


class TestStreamingGenerator:
    """Tests for test() returning a generator when streaming is enabled."""

    def test_streaming_yields_chunks_incrementally(self):
        """Test that test() yields chunks as they arrive (without collector)."""
        function = mlrun.new_function("test", kind="serving")
        graph = function.set_topology("flow", engine="async")

        # Streaming step without collector - should yield individual chunks
        graph.to(name="streamer", class_name="StreamingStep").respond()

        server = function.to_mock_server()
        try:
            result = server.test("/", body="test")
            assert inspect.isgenerator(result), (
                "test() should return a generator for streaming"
            )

            chunks = list(result)
            assert chunks == ["test_chunk_0", "test_chunk_1", "test_chunk_2"]
        finally:
            server.wait_for_completion()


class TestModelIsStreaming:
    """Tests for Model.is_streaming() method."""

    def test_is_streaming_detects_generator_function(self):
        """Test that is_streaming() returns True for generator predict()."""

        class GeneratorModel(Model):
            def predict(self, body, **kwargs):
                yield "chunk1"
                yield "chunk2"

        model = GeneratorModel(name="test")
        assert model.is_streaming() is True

    def test_is_streaming_detects_async_generator_function(self):
        """Test that is_streaming() returns True for async generator predict_async()."""

        class AsyncGeneratorModel(Model):
            def predict(self, body, **kwargs):
                return body  # Not a generator

            async def predict_async(self, body, **kwargs):
                yield "chunk1"
                yield "chunk2"

        model = AsyncGeneratorModel(name="test")
        assert model.is_streaming() is True

    def test_is_streaming_returns_false_for_regular_predict(self):
        """Test that is_streaming() returns False for non-generator predict()."""

        class RegularModel(Model):
            def predict(self, body, **kwargs):
                return {"result": body}

        model = RegularModel(name="test")
        assert model.is_streaming() is False

    def test_is_streaming_override_for_returned_generator(self):
        """Test that is_streaming() can be overridden when predict() returns a generator.

        This tests the documented use case where predict() returns a generator
        from an external source without being a generator function itself.
        """

        def external_streaming_api(body):
            """Simulates an external API that returns a generator."""
            for i in range(3):
                yield f"{body}_chunk_{i}"

        class WrappedStreamingModel(Model):
            def predict(self, body, **kwargs):
                # Returns a generator, but this method is NOT a generator function
                return external_streaming_api(body)

            def is_streaming(self) -> bool:
                # Override required since predict() is not a generator function
                return True

        model = WrappedStreamingModel(name="test")

        # Verify that without override, is_streaming would return False
        # (since predict is not a generator function)
        assert not inspect.isgeneratorfunction(model.predict)

        # But with the override, it returns True
        assert model.is_streaming() is True

        # And predict() actually returns a generator
        result = model.predict("test")
        assert inspect.isgenerator(result)
        assert list(result) == ["test_chunk_0", "test_chunk_1", "test_chunk_2"]


def _make_mock_streaming_provider():
    """Create a MockModelProvider instance without a real parent."""
    provider = MockModelProvider.__new__(MockModelProvider)
    provider.default_invoke_kwargs = {}
    return provider


class _MockNonStreamingProvider(ModelProvider):
    """A mock model provider that does NOT support streaming."""

    supports_streaming = False

    def __init__(self):
        self.default_invoke_kwargs = {}


class TestLLModelStreaming:
    """Tests for LLModel streaming integration with model providers."""

    _sentinel = object()

    @classmethod
    def _make_model(cls, streaming_server=True, provider=_sentinel):
        model = LLModel(name="test")
        model._streaming_enabled = streaming_server
        model.model_provider = (
            _make_mock_streaming_provider() if provider is cls._sentinel else provider
        )
        return model

    def test_is_streaming_true(self):
        """is_streaming returns True when server.streaming and provider both support it."""
        assert self._make_model().is_streaming() is True

    @pytest.mark.parametrize(
        "streaming_server, provider",
        [
            (False, _make_mock_streaming_provider()),
            (True, _MockNonStreamingProvider()),
            (True, None),
        ],
        ids=["server-off", "provider-unsupported", "no-provider"],
    )
    def test_is_streaming_false(self, streaming_server, provider):
        """is_streaming returns False when any precondition is missing."""
        assert self._make_model(streaming_server, provider).is_streaming() is False

    def test_predict_returns_generator_when_streaming(self):
        """predict() returns a generator of tokens when streaming is active."""
        model = self._make_model()
        result = model.predict(
            body={"input": "hello"},
            messages=[{"role": "user", "content": "hello"}],
            llm_prompt_artifact=unittest.mock.MagicMock(
                spec=mlrun.artifacts.LLMPromptArtifact
            ),
        )
        assert inspect.isgenerator(result)
        tokens = list(result)
        assert len(tokens) > 0
        full_text = "".join(tokens)
        assert "mock model provider" in full_text.lower()

    def test_run_async_returns_async_generator_when_streaming(self):
        """run_async() returns an async generator directly when streaming is active."""
        model = self._make_model()
        model.invocation_artifact = unittest.mock.MagicMock(
            spec=mlrun.artifacts.LLMPromptArtifact
        )
        model._artifact_were_loaded = True
        messages = [{"role": "user", "content": "hello"}]
        model.enrich_prompt = lambda body, origin, llm_prompt_artifact: (
            messages,
            {},
        )
        result = model.run_async(body={"input": "hello"}, path="/")
        assert inspect.isasyncgen(result)

    def test_init_raises_when_provider_does_not_support_streaming(self):
        """init() raises when streaming is enabled but provider doesn't support it."""
        model = self._make_model(provider=_MockNonStreamingProvider())
        model._execution_mechanism = "asyncio"
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="does not support streaming",
        ):
            model.init()
