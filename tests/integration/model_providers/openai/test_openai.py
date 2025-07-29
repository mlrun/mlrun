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
import os
import time
import unittest.mock
from typing import cast

import openai.types.chat
import pytest
import tiktoken
import yaml
from openai import AsyncOpenAI, OpenAI
from openai.types import CreateEmbeddingResponse

import mlrun
import mlrun.artifacts
import mlrun.serving.states
from mlrun.datastore import store_manager
from mlrun.datastore.datastore_profile import (
    OpenAIProfile,
    register_temporary_client_datastore_profile,
)
from mlrun.datastore.model_provider.openai_provider import OpenAIProvider
from tests.datastore.remote_model.remote_model_utils import (
    EXPECTED_RESULTS,
    INPUT_DATA,
    assert_async_invocations,
    formatted_messages,
    setup_remote_model_test,
)

here = os.path.dirname(__file__)
config = {}
config_file_path = os.path.join(here, "test-openai.yml")
if os.path.exists(config_file_path):
    with open(config_file_path) as yaml_file:
        config = yaml.safe_load(yaml_file).get("env", {})


def create_mocked_get_store_artifact(uri_to_artifact: dict):
    def mocked_get_store_artifact(uri, **kwargs):
        artifact = uri_to_artifact.get(uri)
        if not artifact:
            raise mlrun.errors.MLRunInvalidArgumentError("Artifact uri not found")
        return artifact, None

    return mocked_get_store_artifact


def openai_configured():
    if (
        not config
        or not config.get("OPENAI_API_KEY")
        or not config.get("OPENAI_BASE_URL")
    ):
        return False
    return True


@pytest.mark.skipif(
    not openai_configured(),
    reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL to be set under test-openai.yml",
)
class TestBasicOpenAIProvider:
    profile_name = "openai_profile"
    env_secrets = config

    @classmethod
    def setup_class(cls):
        cls.basic_llm_model = "gpt-4o-mini"

    @classmethod
    def reset_env(cls):
        for key, env_param in cls.env_secrets.items():
            if env_param:
                os.environ.pop(key, None)

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self):
        for key, env_param in self.env_secrets.items():
            if env_param:
                os.environ[key] = env_param
        store_manager.reset_secrets()
        # noinspection PyAttributeOutsideInit
        self.url_prefix = "openai://"

    def setup_datastore_profile(self):
        # noinspection PyAttributeOutsideInit
        self.profile = OpenAIProfile(
            name=self.profile_name,
            api_key=self.env_secrets.get("OPENAI_API_KEY"),
            organization=self.env_secrets.get("OPENAI_ORG_ID"),
            project=self.env_secrets.get("OPENAI_PROJECT_ID"),
            base_url=self.env_secrets.get("OPENAI_BASE_URL"),
            timeout=self.env_secrets.get("OPENAI_TIMEOUT"),
            max_retries=self.env_secrets.get("OPENAI_MAX_RETRIES"),
        )
        register_temporary_client_datastore_profile(self.profile)
        # noinspection PyAttributeOutsideInit
        self.url_prefix = f"ds://{self.profile_name}/"
        self.reset_env()


class TestOpenAIProvider(TestBasicOpenAIProvider):
    @classmethod
    def check_basic_invoke(cls, model_url: str, secrets: dict, model_name: str):
        messages = [formatted_messages[0]]
        model_provider = mlrun.get_model_provider(
            url=model_url, secrets=secrets, default_invoke_kwargs={"max_tokens": 100}
        )
        model_provider = cast(OpenAIProvider, model_provider)
        assert model_provider.model == model_name
        result = model_provider.invoke(messages=messages, as_str=True)
        assert isinstance(result, str)
        assert EXPECTED_RESULTS[0] in result.lower()

        encoding = tiktoken.encoding_for_model(model_name)
        token_count = len(encoding.encode(result))
        assert token_count == 100
        # checking as_str = False
        response = model_provider.invoke(
            messages=messages,
            max_tokens=50,
        )
        token_count = len(encoding.encode(response.choices[0].message.content))
        assert isinstance(response, openai.types.chat.ChatCompletion)
        assert token_count == 50

    @pytest.mark.parametrize("cred_mode", ["profile", "env", "secrets"])
    def test_basic_invoke(self, cred_mode):
        secrets = {}
        if cred_mode == "profile":
            self.setup_datastore_profile()
        elif cred_mode == "secrets":
            self.reset_env()
            secrets = self.env_secrets

        model_url = self.url_prefix + self.basic_llm_model
        self.check_basic_invoke(
            model_url=model_url, secrets=secrets, model_name=self.basic_llm_model
        )

    def test_configurable_model(self):
        configurable_model = mlrun.mlconf.model_providers.openai_default_model
        if not configurable_model:
            pytest.skip(
                "model_providers.openai_default_model is not configured in conf, cannot perform the test"
            )

        #  checking default model usage:
        model_url = self.url_prefix
        #  env check
        self.check_basic_invoke(
            model_url=model_url, secrets={}, model_name=configurable_model
        )
        # secrets check
        self.reset_env()
        self.check_basic_invoke(
            model_url=model_url, secrets=self.env_secrets, model_name=configurable_model
        )

    def test_system_prompt(self):
        model_url = self.url_prefix + self.basic_llm_model
        system_prompt = "You are a special LLM model that always answers user questions with one word only."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is your opinion on climate change?"},
        ]
        model_provider = mlrun.get_model_provider(
            url=model_url, default_invoke_kwargs={"max_tokens": 200}
        )
        result = model_provider.invoke(messages=messages, as_str=True)
        assert isinstance(result, str)
        result = result.strip()
        assert result
        assert " " not in result.strip()  # checking one-word answer

    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_datastore_profile", [True, False])
    async def test_async_invoke(self, use_datastore_profile):
        if use_datastore_profile:
            self.setup_datastore_profile()
        model_url = self.url_prefix + self.basic_llm_model
        model_provider = mlrun.get_model_provider(
            url=model_url, default_invoke_kwargs={"max_tokens": 100}
        )
        model_provider = cast(OpenAIProvider, model_provider)
        assert model_provider.model == self.basic_llm_model
        coroutine1 = model_provider.async_invoke(
            messages=[formatted_messages[0]], as_str=True
        )
        coroutine2 = model_provider.async_invoke(messages=[formatted_messages[1]])
        result1, result2 = await asyncio.gather(coroutine1, coroutine2)
        result2 = result2.choices[0].message.content
        assert EXPECTED_RESULTS[0] in result1.lower()
        assert EXPECTED_RESULTS[1] in result2.lower()

        encoding = tiktoken.encoding_for_model(self.basic_llm_model)
        assert len(encoding.encode(result1)) == 100
        assert len(encoding.encode(result2)) == 100

    @pytest.mark.asyncio
    @pytest.mark.parametrize("run_async", [True, False])
    async def test_custom_invoke(self, run_async):
        model_name = "text-embedding-3-small"
        model_url = self.url_prefix + model_name
        model_provider = mlrun.get_model_provider(url=model_url)
        prompt = "OpenAI is amazing"
        client: OpenAI = model_provider.client
        async_client: AsyncOpenAI = model_provider.async_client
        if run_async:
            embeddings = await model_provider.async_custom_invoke(
                operation=async_client.embeddings.create, input=prompt
            )
            with pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError,
                match="OpenAI async_custom_invoke operation"
                " must be a coroutine function",
            ):
                _ = await model_provider.async_custom_invoke(
                    operation=client.embeddings.create, input=prompt
                )
        else:
            embeddings = model_provider.custom_invoke(
                operation=client.embeddings.create, input=prompt
            )
            with pytest.raises(
                mlrun.errors.MLRunInvalidArgumentError,
                match="OpenAI custom_invoke " "operation must be a callable",
            ):
                _ = await model_provider.custom_invoke(operation="test", input=prompt)
        encoding = tiktoken.encoding_for_model(model_name)
        token_count = len(encoding.encode(prompt))
        assert embeddings.data[0].embedding is not None
        assert len(embeddings.data[0].embedding) > 0
        assert embeddings.usage.total_tokens == token_count
        assert isinstance(embeddings, CreateEmbeddingResponse)


class TestOpenAIModel(TestBasicOpenAIProvider):
    @pytest.mark.parametrize("execution_mechanism", ["naive", "asyncio"])
    def test_model_runner_with_openai(self, execution_mechanism):
        project = mlrun.new_project("test-openai-model", save=False)
        model_url = self.url_prefix + self.basic_llm_model
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            project,
            model_url,
            execution_mechanism=execution_mechanism,
            default_config={"max_tokens": 100},
        )
        # # Mock needed since no artifact is saved in this test, so retrieval by URI isn't possible.
        # # Mocked function used to verify artifact URI is passed correctly.
        #
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
            result = server.test(body=INPUT_DATA[0])["result"]
            assert EXPECTED_RESULTS[0] in result.lower()
            encoding = tiktoken.encoding_for_model(self.basic_llm_model)
            assert len(encoding.encode(result)) == 100
        finally:
            server.wait_for_completion()

    def test_open_ai_async_parallel_events(self):
        # test that we have the ability to run multiple events asynchronously, by custom model setup
        project = mlrun.new_project("test-openai-model", save=False)
        model_url = self.url_prefix + self.basic_llm_model
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            project,
            model_url,
            execution_mechanism="asyncio",
            model_class="MyOpenAIAsyncEvents",
            default_config={"max_tokens": 100},
        )
        # # Mock needed since no artifact is saved in this test, so retrieval by URI isn't possible.
        # # Mocked function used to verify artifact URI is passed correctly.
        #
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
            start = time.perf_counter()
            results_with_times = server.test(body={"input": INPUT_DATA})
            total_duration = time.perf_counter() - start

            assert_async_invocations(
                results_with_times=results_with_times,
                model_name=self.basic_llm_model,
                total_duration=total_duration,
            )
        finally:
            server.wait_for_completion()
