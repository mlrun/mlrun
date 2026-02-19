# Copyright 2025 Iguazio
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

import asyncio
import inspect
import math
import time
import unittest.mock

import pytest

import mlrun
import mlrun.errors


class TestOpenAIBatch:
    """Test batch invocation for both sync (invoke) and async (async_invoke) methods."""

    @pytest.fixture
    def mock_async_single_invoke(self):
        state = {
            "current_running": 0,
            "max_concurrent_observed": 0,
            "lock": asyncio.Lock(),
            "call_count": 0,
        }

        async def _mock(self, messages, invoke_response_format, **kwargs):
            async with state["lock"]:
                state["current_running"] += 1
                state["call_count"] += 1
                state["max_concurrent_observed"] = max(
                    state["max_concurrent_observed"], state["current_running"]
                )

            # Simulate API latency for a single OpenAI call
            await asyncio.sleep(0.1)

            async with state["lock"]:
                state["current_running"] -= 1

            return {"mock": "response", "answer": "mocked"}

        _mock.state = state
        return _mock

    @pytest.fixture
    def mock_async_single_invoke_with_failure(self):
        """Async mock that fails on a specific message index for testing error handling."""
        state = {
            "lock": asyncio.Lock(),
            "call_count": 0,
            "fail_on_index": None,  # Set this in the test
        }

        async def _mock(self, messages, invoke_response_format, **kwargs):
            async with state["lock"]:
                current_index = state["call_count"]
                state["call_count"] += 1

            # Check if this call should fail BEFORE sleep
            if current_index == state["fail_on_index"]:
                # Fail quickly to test fast-fail behavior
                await asyncio.sleep(0.05)
                raise RuntimeError(f"Simulated API error on message {current_index}")

            # Normal flow: simulate API latency
            await asyncio.sleep(0.5)

            return {"mock": "response", "answer": "mocked"}

        _mock.state = state
        return _mock

    def test_sync_batch_concurrency_limit(self, mock_async_single_invoke):
        """Ensure sync batch invocation caps concurrent tasks to openai_batch_max_concurrent."""
        latency = 0.1
        per_batch_limit = mlrun.mlconf.model_providers.openai_batch_max_concurrent
        total_messages = per_batch_limit * 2

        with unittest.mock.patch(
            "mlrun.datastore.model_provider.openai_provider.OpenAIProvider._async_single_invoke",
            mock_async_single_invoke,
        ):
            provider = mlrun.get_model_provider(
                url="openai://gpt-4o-mini",
                secrets={"OPENAI_API_KEY": "test-key"},
            )

            messages_list = [
                [{"role": "user", "content": f"message {i}"}]
                for i in range(total_messages)
            ]

            start = time.perf_counter()
            results = provider.invoke(messages=messages_list)
            duration = time.perf_counter() - start

        state = mock_async_single_invoke.state
        assert len(results) == total_messages
        assert state["call_count"] == total_messages
        assert state["max_concurrent_observed"] <= per_batch_limit

        expected_duration = (total_messages / per_batch_limit) * latency
        upper_bound = expected_duration + 0.1
        assert expected_duration <= duration <= upper_bound

    def test_sync_batch_error_handling_fast_fail(
        self, mock_async_single_invoke_with_failure
    ):
        """Verify sync batch invocation fails fast when one invocation raises an exception."""
        per_batch_limit = mlrun.mlconf.model_providers.openai_batch_max_concurrent
        fail_on_index = math.ceil(per_batch_limit / 2)
        total_messages = per_batch_limit * 2

        mock_async_single_invoke_with_failure.state["fail_on_index"] = fail_on_index

        with unittest.mock.patch(
            "mlrun.datastore.model_provider.openai_provider.OpenAIProvider._async_single_invoke",
            mock_async_single_invoke_with_failure,
        ):
            provider = mlrun.get_model_provider(
                url="openai://gpt-4o-mini",
                secrets={"OPENAI_API_KEY": "test-key"},
            )

            messages_list = [
                [{"role": "user", "content": f"message {i}"}]
                for i in range(total_messages)
            ]

            start = time.perf_counter()

            with pytest.raises(
                RuntimeError, match=f"Simulated API error on message {fail_on_index}"
            ):
                provider.invoke(messages=messages_list)

            duration = time.perf_counter() - start

        state = mock_async_single_invoke_with_failure.state

        assert duration < 0.7, "Should fail fast, not wait for all tasks"
        assert state["call_count"] == per_batch_limit + 1
        assert state["call_count"] < total_messages, (
            f"Fast-fail should prevent remaining tasks from executing: "
            f"expected < {total_messages}, got {state['call_count']}"
        )

    @pytest.mark.parametrize(
        "invalid_messages, error_match",
        [
            (
                [
                    [{"role": "user", "content": "message 1"}],  # list
                    {"role": "user", "content": "message 2"},  # dict - INVALID
                ],
                "cannot mix list and dict items",
            ),
            (
                ["message 1", "message 2", "message 3"],  # list of strings - INVALID
                "list of strings is not supported",
            ),
            (
                [],  # empty list - INVALID
                "Messages must be a non-empty list of dictionaries or list of lists of dictionaries.",
            ),
            (
                None,  # not a list - INVALID
                "Messages must be a non-empty list of dictionaries or list of lists of dictionaries.",
            ),
            (
                "single message string",  # string instead of list - INVALID
                "Messages must be a non-empty list of dictionaries or list of lists of dictionaries.",
            ),
        ],
    )
    def test_sync_invalid_messages_raises_error(self, invalid_messages, error_match):
        """Verify that invalid message formats raise appropriate errors in sync invocation."""
        provider = mlrun.get_model_provider(
            url="openai://gpt-4o-mini",
            secrets={"OPENAI_API_KEY": "test-key"},
        )

        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match=error_match,
        ):
            provider.invoke(messages=invalid_messages)

    @pytest.mark.asyncio
    async def test_sync_batch_invoke_from_event_loop(self, mock_async_single_invoke):
        """Verify that sync batch invoke works when called from within an event loop."""
        with unittest.mock.patch(
            "mlrun.datastore.model_provider.openai_provider.OpenAIProvider._async_single_invoke",
            mock_async_single_invoke,
        ):
            provider = mlrun.get_model_provider(
                url="openai://gpt-4o-mini",
                secrets={"OPENAI_API_KEY": "test-key"},
            )

            messages_list = [
                [{"role": "user", "content": "message 1"}],
                [{"role": "user", "content": "message 2"}],
            ]

            # Should work even when called from within an existing event loop
            # (Currently will fail with RuntimeError until we fix the asyncio.run() issue)
            results = provider.invoke(messages=messages_list)

            assert len(results) == 2
            assert all(result["mock"] == "response" for result in results)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_async", [True, False])
    async def test_batch_concurrency_limit_from_event_loop(
        self, mock_async_single_invoke, use_async
    ):
        """Ensure batch invocation caps concurrent tasks to openai_batch_max_concurrent."""
        latency = 0.1
        per_batch_limit = mlrun.mlconf.model_providers.openai_batch_max_concurrent
        total_messages = per_batch_limit * 2

        with unittest.mock.patch(
            "mlrun.datastore.model_provider.openai_provider.OpenAIProvider._async_single_invoke",
            mock_async_single_invoke,
        ):
            provider = mlrun.get_model_provider(
                url="openai://gpt-4o-mini",
                secrets={"OPENAI_API_KEY": "test-key"},
            )

            messages_list = [
                [{"role": "user", "content": f"message {i}"}]
                for i in range(total_messages)
            ]

            start = time.perf_counter()
            if use_async:
                results = await provider.async_invoke(messages=messages_list)
            else:
                results = provider.invoke(messages=messages_list)
            duration = time.perf_counter() - start

        state = mock_async_single_invoke.state
        assert len(results) == total_messages
        assert state["call_count"] == total_messages
        assert state["max_concurrent_observed"] <= per_batch_limit

        expected_duration = (total_messages / per_batch_limit) * latency
        upper_bound = expected_duration + 0.1
        assert expected_duration <= duration <= upper_bound

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_async", [True, False])
    async def test_batch_error_handling_fast_fail_from_event_loop(
        self, mock_async_single_invoke_with_failure, use_async
    ):
        """Verify batch invocation fails fast when one invocation raises an exception."""
        per_batch_limit = mlrun.mlconf.model_providers.openai_batch_max_concurrent
        fail_on_index = math.ceil(per_batch_limit / 2)
        total_messages = per_batch_limit * 2

        mock_async_single_invoke_with_failure.state["fail_on_index"] = fail_on_index

        with unittest.mock.patch(
            "mlrun.datastore.model_provider.openai_provider.OpenAIProvider._async_single_invoke",
            mock_async_single_invoke_with_failure,
        ):
            provider = mlrun.get_model_provider(
                url="openai://gpt-4o-mini",
                secrets={"OPENAI_API_KEY": "test-key"},
            )

            messages_list = [
                [{"role": "user", "content": f"message {i}"}]
                for i in range(total_messages)
            ]

            start = time.perf_counter()

            with pytest.raises(
                RuntimeError,
                match=f"Simulated API error on message {fail_on_index}",
            ):
                if use_async:
                    await provider.async_invoke(messages=messages_list)
                else:
                    provider.invoke(messages=messages_list)

            duration = time.perf_counter() - start

        state = mock_async_single_invoke_with_failure.state

        assert duration < 0.7, "Should fail fast, not wait for all tasks"
        assert state["call_count"] == per_batch_limit + 1
        assert state["call_count"] < total_messages, (
            f"Fast-fail should prevent remaining tasks from executing: "
            f"expected < {total_messages}, got {state['call_count']}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "invalid_messages, error_match",
        [
            (
                [
                    [{"role": "user", "content": "message 1"}],  # list
                    {"role": "user", "content": "message 2"},  # dict - INVALID
                ],
                "cannot mix list and dict items",
            ),
            (
                ["message 1", "message 2", "message 3"],  # list of strings - INVALID
                "list of strings is not supported",
            ),
            (
                [],  # empty list - INVALID
                "Messages must be a non-empty list of dictionaries or list of lists of dictionaries.",
            ),
            (
                None,  # not a list - INVALID
                "Messages must be a non-empty list of dictionaries or list of lists of dictionaries.",
            ),
            (
                "single message string",  # string instead of list - INVALID
                "Messages must be a non-empty list of dictionaries or list of lists of dictionaries.",
            ),
        ],
    )
    async def test_async_invalid_messages_raises_error(
        self, invalid_messages, error_match
    ):
        """Verify that invalid message formats raise appropriate errors in async invocation."""
        provider = mlrun.get_model_provider(
            url="openai://gpt-4o-mini",
            secrets={"OPENAI_API_KEY": "test-key"},
        )

        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match=error_match,
        ):
            await provider.async_invoke(messages=invalid_messages)


class _MockChunkDelta:
    """Simulates openai ChatCompletionChunk delta."""

    def __init__(self, content):
        self.content = content


class _MockChunkChoice:
    """Simulates openai ChatCompletionChunk choice."""

    def __init__(self, content):
        self.delta = _MockChunkDelta(content)


class _MockChunk:
    """Simulates openai ChatCompletionChunk."""

    def __init__(self, content):
        self.choices = [_MockChunkChoice(content)] if content is not None else []


class TestOpenAIStreaming:
    """Tests for OpenAI streaming invoke methods."""

    def test_supports_streaming_flag(self):
        """Verify that OpenAIProvider declares streaming support."""
        provider = mlrun.get_model_provider(
            url="openai://gpt-4o-mini",
            secrets={"OPENAI_API_KEY": "test-key"},
        )
        assert provider.supports_streaming is True

    def test_invoke_stream_yields_tokens(self):
        """invoke_stream yields content tokens from the OpenAI streaming API."""
        chunks = [_MockChunk("Hello"), _MockChunk(None), _MockChunk(" world")]

        provider = mlrun.get_model_provider(
            url="openai://gpt-4o-mini",
            secrets={"OPENAI_API_KEY": "test-key"},
        )
        with unittest.mock.patch.object(
            provider.client.chat.completions, "create", return_value=iter(chunks)
        ):
            tokens = list(
                provider.invoke_stream(
                    messages=[{"role": "user", "content": "hi"}],
                )
            )
        assert tokens == ["Hello", " world"]

    def test_invoke_stream_rejects_batch(self):
        """invoke_stream raises on batch invocation (list of lists)."""
        provider = mlrun.get_model_provider(
            url="openai://gpt-4o-mini",
            secrets={"OPENAI_API_KEY": "test-key"},
        )
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Batch invocation is not supported in streaming mode",
        ):
            list(
                provider.invoke_stream(
                    messages=[
                        [{"role": "user", "content": "msg1"}],
                        [{"role": "user", "content": "msg2"}],
                    ],
                )
            )

    @pytest.mark.asyncio
    async def test_async_invoke_stream_yields_tokens(self):
        """async_invoke_stream yields content tokens from the OpenAI async streaming API."""
        chunks = [_MockChunk("Hello"), _MockChunk(None), _MockChunk(" world")]

        class _MockAsyncStream:
            """Simulates an OpenAI AsyncStream object (awaitable + async iterable)."""

            def __init__(self, items):
                self._items = items

            def __aiter__(self):
                return self._async_iter()

            async def _async_iter(self):
                for item in self._items:
                    yield item

        async def _mock_create(**kwargs):
            return _MockAsyncStream(chunks)

        provider = mlrun.get_model_provider(
            url="openai://gpt-4o-mini",
            secrets={"OPENAI_API_KEY": "test-key"},
        )
        with unittest.mock.patch.object(
            provider.async_client.chat.completions,
            "create",
            side_effect=_mock_create,
        ):
            tokens = [
                token
                async for token in provider.async_invoke_stream(
                    messages=[{"role": "user", "content": "hi"}],
                )
            ]
        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_async_invoke_stream_rejects_batch(self):
        """async_invoke_stream raises on batch invocation."""
        provider = mlrun.get_model_provider(
            url="openai://gpt-4o-mini",
            secrets={"OPENAI_API_KEY": "test-key"},
        )
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Batch invocation is not supported in streaming mode",
        ):
            async for _ in provider.async_invoke_stream(
                messages=[
                    [{"role": "user", "content": "msg1"}],
                    [{"role": "user", "content": "msg2"}],
                ],
            ):
                pass

    def test_llmodel_streaming(self, monkeypatch):
        """Streaming through MRS yields concatenated token string via mocked OpenAI."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234")

        from tests.datastore.remote_model.remote_model_utils import (
            BATCH_INPUT_DATA,
            create_mocked_get_store_artifact,
            setup_remote_model_test,
        )

        project = mlrun.new_project("test-openai-stream-mrs", save=False)
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            project,
            "openai://gpt-4o-mini",
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

            async def mock_async_stream(self_provider, messages, **kwargs):
                for token in ["Hello", " world"]:
                    yield token

            with unittest.mock.patch(
                "mlrun.datastore.model_provider.openai_provider.OpenAIProvider.async_invoke_stream",
                mock_async_stream,
            ):
                response = server.test(body=BATCH_INPUT_DATA[0])
            assert inspect.isgenerator(
                response
            ), f"Expected generator, got {type(response)}"
            response = "".join(response)
            assert "Hello world" in response
        finally:
            server.wait_for_completion()

    def test_invoke_stream_passes_invoke_kwargs(self):
        """invoke_stream forwards invoke_kwargs and default_invoke_kwargs to the API call."""
        provider = mlrun.get_model_provider(
            url="openai://gpt-4o-mini",
            secrets={"OPENAI_API_KEY": "test-key"},
        )
        mock_create = unittest.mock.MagicMock(return_value=iter([]))
        with unittest.mock.patch.object(
            provider.client.chat.completions, "create", mock_create
        ):
            list(
                provider.invoke_stream(
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=0.5,
                )
            )
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs["stream"] is True
        assert call_kwargs.kwargs["temperature"] == 0.5
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"
