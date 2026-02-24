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
#
#
# Note: Downloading HuggingFace models requires stable network connectivity and may fail or get stuck
# on unreliable connections. Ensure adequate network bandwidth when running tests that download models.

import unittest.mock

import pytest

import mlrun
from mlrun.datastore.model_provider.huggingface_provider import HuggingFaceProvider


@pytest.mark.parametrize(
    "response, expected_str_response",
    [
        (
            [{"generated_text": "The capital of Germany is Berlin."}],
            "The capital of Germany is Berlin.",
        ),
    ],
)
def test_response_to_str(response, expected_str_response):
    extracted_string = HuggingFaceProvider._extract_string_output(response=response)
    assert extracted_string == expected_str_response


def test_response_to_str_error():
    # This response can be reproduced with Hugging Face Provider by invoking with num_return_sequences=2:
    response = [
        {"generated_text": "The capital of Germany is Berlin."},
        {"generated_text": "The capital of Japan is Tokyo"},
    ]
    with pytest.raises(
        mlrun.errors.MLRunInvalidArgumentError,
        match="HuggingFaceProvider: extracting string from response is only"
        " supported for single-response outputs",
    ):
        HuggingFaceProvider._extract_string_output(response=response)


def _make_provider():
    """Create a HuggingFaceProvider without downloading a model or loading a pipeline."""
    mock_parent = unittest.mock.MagicMock()
    mock_parent.secret.return_value = None
    with unittest.mock.patch.object(HuggingFaceProvider, "_download_model"):
        provider = HuggingFaceProvider(
            parent=mock_parent,
            schema="huggingface",
            name="test",
            endpoint="test-model",
            secrets={"HF_TOKEN": "fake"},
        )

    mock_tokenizer = unittest.mock.MagicMock()
    mock_pipeline = unittest.mock.MagicMock()
    mock_pipeline.task = "text-generation"
    mock_pipeline.tokenizer = mock_tokenizer
    provider._client = mock_pipeline
    return provider


class _FakeStreamer:
    """Simulates a TextIteratorStreamer that yields pre-set tokens.

    Iteration blocks until :meth:`push` is called (from the generation thread).
    """

    def __init__(self):
        import threading

        self._tokens = []
        self._done = threading.Event()

    def push(self, tokens):
        self._tokens = tokens
        self._done.set()

    def __iter__(self):
        self._done.wait()
        return iter(self._tokens)


class TestHuggingFaceStreaming:
    """Tests for HuggingFace streaming invoke methods."""

    def test_supports_streaming_flag(self):
        assert HuggingFaceProvider.supports_streaming is True

    def test_invoke_stream_yields_tokens(self):
        """invoke_stream yields tokens produced by the streamer."""
        provider = _make_provider()
        tokens = ["Hello", "", " world"]
        fake_streamer = _FakeStreamer()

        def _mock_prepare_stream(messages, invoke_kwargs):
            return fake_streamer, invoke_kwargs

        original_custom_invoke = provider.custom_invoke

        def _mock_custom_invoke(**kwargs):
            fake_streamer.push(tokens)
            return original_custom_invoke(**kwargs)

        provider._prepare_stream = _mock_prepare_stream
        provider.custom_invoke = _mock_custom_invoke

        result = list(
            provider.invoke_stream(
                messages=[{"role": "user", "content": "hi"}],
            )
        )
        assert result == ["Hello", " world"]

    def test_invoke_stream_rejects_batch(self):
        """invoke_stream raises on batch invocation."""
        provider = _make_provider()
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

    def test_invoke_stream_rejects_non_text_generation(self):
        """invoke_stream raises when pipeline task is not text-generation."""
        provider = _make_provider()
        provider._client.task = "image-classification"
        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="streaming supports text-generation task only",
        ):
            list(
                provider.invoke_stream(
                    messages=[{"role": "user", "content": "hi"}],
                )
            )
