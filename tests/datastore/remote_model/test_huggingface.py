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
from typing import cast

import pytest

import mlrun
from mlrun.datastore.datastore_profile import (
    HuggingFaceProfile,
    register_temporary_client_datastore_profile,
)
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


@pytest.mark.parametrize("cred_mode", ["profile", "secrets"])
def test_snapshot_download_called_with_all_params(cred_mode):
    """
    Verifies that all expected parameters are forwarded correctly to snapshot_download,
    and that endpoint is NOT present in client_options (pipeline kwargs).
    Runs for both profile-based and direct secrets-dict credential modes.
    """
    fake_token = "fake-token"
    fake_endpoint = "https://my-custom-hub.example.com"
    model_name = "fake-org/fake-model"
    profile_name = "test-hf-profile"
    fake_max_workers = 4

    if cred_mode == "profile":
        profile = HuggingFaceProfile(
            name=profile_name,
            token=fake_token,
            endpoint=fake_endpoint,
            task="text-generation",
            device="cpu",
            device_map="auto",
            trust_remote_code=True,
            model_kwargs={"torch_dtype": "float16"},
            max_workers=fake_max_workers,
        )
        register_temporary_client_datastore_profile(profile)
        url = f"ds://{profile_name}/{model_name}"
        secrets = {}
    else:
        url = f"huggingface://{model_name}"
        secrets = {
            "HF_TOKEN": fake_token,
            "HF_ENDPOINT": fake_endpoint,
            "HF_TASK": "text-generation",
            "HF_DEVICE": "cpu",
            "HF_DEVICE_MAP": "auto",
            "HF_TRUST_REMOTE_CODE": True,
            "HF_MODEL_KWARGS": {"torch_dtype": "float16"},
            "HF_MAX_WORKERS": fake_max_workers,
        }

    with unittest.mock.patch("huggingface_hub.snapshot_download") as mock_snapshot:
        provider = mlrun.get_model_provider(url=url, secrets=secrets)

        mock_snapshot.assert_called_once_with(
            repo_id=model_name,
            local_dir_use_symlinks=False,
            token=fake_token,
            endpoint=fake_endpoint,
            max_workers=fake_max_workers,
        )

    provider = cast(HuggingFaceProvider, provider)

    # endpoint and max_workers must not bleed into pipeline() kwargs
    assert "endpoint" not in provider.options
    assert "max_workers" not in provider.options

    # verify all other client options are correctly populated
    assert provider.options["task"] == "text-generation"
    assert provider.options["token"] == fake_token
    assert provider.options["device"] == "cpu"
    assert provider.options["device_map"] == "auto"
    assert provider.options["trust_remote_code"] is True
    assert provider.options["model_kwargs"] == {"torch_dtype": "float16"}


@pytest.mark.parametrize("cred_mode", ["profile", "secrets"])
def test_client_options_defaults(cred_mode):
    """
    Verifies that when no optional params are provided,
    client_options fall back to their expected defaults.
    """
    model_name = "fake-org/fake-model"
    profile_name = "test-hf-defaults-profile"

    if cred_mode == "profile":
        profile = HuggingFaceProfile(name=profile_name)
        register_temporary_client_datastore_profile(profile)
        url = f"ds://{profile_name}/{model_name}"
        secrets = {}
    else:
        url = f"huggingface://{model_name}"
        secrets = {}

    with unittest.mock.patch("huggingface_hub.snapshot_download") as mock_snapshot:
        provider = mlrun.get_model_provider(url=url, secrets=secrets)

        mock_snapshot.assert_called_once_with(
            repo_id=model_name,
            local_dir_use_symlinks=False,
            token=None,
            endpoint=None,
            max_workers=None,
        )

    provider = cast(HuggingFaceProvider, provider)

    assert provider.options["task"] == "text-generation"
    assert "token" not in provider.options
    assert "device" not in provider.options
    assert "device_map" not in provider.options
    assert "trust_remote_code" not in provider.options
    assert "model_kwargs" not in provider.options
    assert "endpoint" not in provider.options
    assert "max_workers" not in provider.options


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
