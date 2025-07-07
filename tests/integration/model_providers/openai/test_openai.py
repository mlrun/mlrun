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

import pytest
import tiktoken
import yaml
from openai import OpenAI
from openai.types import CreateEmbeddingResponse

import mlrun
import mlrun.artifacts
import mlrun.serving.states
from mlrun.datastore import store_manager
from mlrun.datastore.datastore_profile import (
    OpenAIProfile,
    register_temporary_client_datastore_profile,
)
from mlrun.datastore.model_provider.model_provider import ModelProvider
from mlrun.datastore.model_provider.openai_provider import OpenAIProvider
from mlrun.serving import ModelRunnerStep

here = os.path.dirname(__file__)
config = {}
config_file_path = os.path.join(here, "test-openai.yml")
if os.path.exists(config_file_path):
    with open(config_file_path) as yaml_file:
        config = yaml.safe_load(yaml_file).get("env", {})


async def timed(coro):
    start = time.perf_counter()
    result = await coro
    duration = time.perf_counter() - start
    return result, duration


class MyOpenAILLM(mlrun.serving.states.Model):
    def predict(self, body):
        if isinstance(
            self.invocation_artifact, mlrun.artifacts.LLMPromptArtifact
        ) and isinstance(self.model_provider, ModelProvider):
            prompt = self.enrich_prompt(body)
            body["result"] = self.model_provider.invoke(
                prompt=prompt,
                **(self.invocation_artifact.spec.model_configuration or {}),
            )
        return body

    async def predict_async(self, body):
        if isinstance(
            self.invocation_artifact, mlrun.artifacts.LLMPromptArtifact
        ) and isinstance(self.model_provider, ModelProvider):
            prompt_parameters: list = body["input"]
            prompts = [
                self.enrich_prompt(single_prompt_parameters)
                for single_prompt_parameters in prompt_parameters
            ]

            tasks = [
                timed(
                    self.model_provider.async_invoke(
                        prompt,
                        **(self.invocation_artifact.spec.model_configuration or {}),
                    )
                )
                for prompt in prompts
            ]
            results_with_times = await asyncio.gather(*tasks)
            results = [r for r, _ in results_with_times]
            invoke_times = [t for _, t in results_with_times]
            body["results"] = results
            body["invoke_times"] = invoke_times
        return body

    def enrich_prompt(self, body) -> str:
        # TODO: Update this once ML-8172 is completed
        if isinstance(self.invocation_artifact, mlrun.artifacts.LLMPromptArtifact):
            prompt_template = self.invocation_artifact.spec.prompt_string
            needed_params = ["question", "depth_level", "persona", "tone"]
            sub_dict = {k: body[k] for k in needed_params if k in body}
            return prompt_template.format(**sub_dict)
        return ""


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
    @staticmethod
    def check_basic_invoke(model_url: str, secrets: dict, model_name: str):
        prompt = "What is the capital of France? Provide a detailed and thorough history of the city"
        model_provider = mlrun.get_model_provider(
            url=model_url, secrets=secrets, default_invoke_kwargs={"max_tokens": 100}
        )
        model_provider = cast(OpenAIProvider, model_provider)
        assert model_provider.model == model_name
        result = model_provider.invoke(prompt=prompt)
        assert "paris" in result.lower()

        encoding = tiktoken.encoding_for_model(model_name)
        token_count = len(encoding.encode(result))
        assert token_count == 100

        result = model_provider.invoke(
            prompt=prompt,
            max_tokens=50,
        )
        token_count = len(encoding.encode(result))
        assert token_count == 50

    @pytest.mark.parametrize("use_datastore_profile", [True, False])
    def test_basic_invoke(self, use_datastore_profile):
        if use_datastore_profile:
            self.setup_datastore_profile()
        model_url = self.url_prefix + self.basic_llm_model
        #  env check
        self.check_basic_invoke(
            model_url=model_url, secrets={}, model_name=self.basic_llm_model
        )
        # secrets check
        self.reset_env()
        self.check_basic_invoke(
            model_url=model_url,
            secrets=self.env_secrets,
            model_name=self.basic_llm_model,
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

    def test_customized_invoke(self):
        model_name = "text-embedding-3-small"
        model_url = self.url_prefix + model_name
        model_provider = mlrun.get_model_provider(url=model_url)
        prompt = "OpenAI is amazing"
        client: OpenAI = model_provider.client
        embeddings = model_provider.customized_invoke(
            operation=client.embeddings.create, input=prompt
        )
        encoding = tiktoken.encoding_for_model(model_name)
        token_count = len(encoding.encode(prompt))
        assert embeddings.data[0].embedding is not None
        assert len(embeddings.data[0].embedding) > 0
        assert embeddings.usage.total_tokens == token_count
        assert isinstance(embeddings, CreateEmbeddingResponse)

    @pytest.mark.asyncio
    async def test_async_invoke(self):
        model_url = self.url_prefix + self.basic_llm_model
        prompt = "What is the capital of France? Provide a detailed and thorough history of the city"
        model_provider = mlrun.get_model_provider(
            url=model_url, default_invoke_kwargs={"max_tokens": 100}
        )
        model_provider = cast(OpenAIProvider, model_provider)
        assert model_provider.model == self.basic_llm_model
        result = await model_provider.async_invoke(prompt=prompt)
        assert "paris" in result.lower()

        encoding = tiktoken.encoding_for_model(self.basic_llm_model)
        token_count = len(encoding.encode(result))
        assert token_count == 100


class TestOpenAIModel(TestBasicOpenAIProvider):
    @pytest.fixture
    def prompt_data(self):
        return {
            "input": [
                {
                    "question": "What is the capital of France, and give a brief historical overview.",
                    "depth_level": "detailed",
                    "persona": "teacher",
                    "tone": "casual",
                },
                {
                    "question": "What is 2 + 2? Answer shortly and then explain with details.",
                    "depth_level": "basic",
                    "persona": "math teacher",
                    "tone": "simple",
                },
                {
                    "question": "Who wrote Hamlet? Answer shortly and then explain with details.",
                    "depth_level": "basic",
                    "persona": "literature professor",
                    "tone": "formal",
                },
                {
                    "question": "What color is the sky on a clear day? Answer shortly and then explain with details.",
                    "depth_level": "basic",
                    "persona": "child",
                    "tone": "fun",
                },
                {
                    "question": "What planet do we live on? Answer shortly and then explain with details.",
                    "depth_level": "basic",
                    "persona": "astronaut",
                    "tone": "educational",
                },
            ],
        }

    @pytest.fixture
    def prompt_expected_results(self):
        return ["paris", "4", "shakespeare", "blue", "earth"]

    def _get_test_attributes(self):
        project = mlrun.new_project("test-openai-model", save=False)
        model_url = self.url_prefix + self.basic_llm_model
        model_artifact = project.log_model(
            "my_model",
            model_url=model_url,
            default_config={"max_tokens": 100},
        )
        prompt_template = (
            "{question}. Explain {depth_level} as a {persona} in {tone} style."
        )
        llm_prompt_artifact = project.log_llm_prompt(
            "my_llm_prompt",
            prompt_string=prompt_template,
            model_artifact=model_artifact.uri,
        )
        function = mlrun.new_function("tests", kind="serving")
        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(name="my_model_runner")
        return model_artifact, llm_prompt_artifact, function, graph, model_runner_step

    def test_model_runner_with_openai(self, prompt_data):
        model_artifact, llm_prompt_artifact, function, graph, model_runner_step = (
            self._get_test_attributes()
        )
        model_runner_step.add_model(
            model_class="MyOpenAILLM",
            endpoint_name="my_endpoint",
            execution_mechanism="naive",
            model_artifact=llm_prompt_artifact,
        )
        graph.to(model_runner_step).respond()
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
            result = server.test(body=prompt_data["input"][0])["result"]
            assert "paris" in result.lower()
            encoding = tiktoken.encoding_for_model(self.basic_llm_model)
            assert len(encoding.encode(result)) == 100
        finally:
            server.wait_for_completion()

    def test_model_runner_with_openai_async(self, prompt_data, prompt_expected_results):
        model_artifact, llm_prompt_artifact, function, graph, model_runner_step = (
            self._get_test_attributes()
        )
        model_runner_step.add_model(
            model_class="MyOpenAILLM",
            endpoint_name="my_endpoint",
            execution_mechanism="asyncio",
            model_artifact=llm_prompt_artifact,
        )
        graph.to(model_runner_step).respond()
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
            results_with_times = server.test(body=prompt_data)
            total_duration = time.perf_counter() - start

            results = results_with_times["results"]
            invoke_times = results_with_times["invoke_times"]
            encoding = tiktoken.encoding_for_model(self.basic_llm_model)
            for i in range(len(prompt_expected_results)):
                assert prompt_expected_results[i] in results[i].lower()
                assert len(encoding.encode(results[i])) == 100
            assert total_duration < sum(invoke_times)
        finally:
            server.wait_for_completion()
