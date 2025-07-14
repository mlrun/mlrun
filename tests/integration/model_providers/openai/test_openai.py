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

import os
import unittest.mock
from typing import cast

import openai.types.chat
import pytest
import tiktoken
import yaml

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


class MyOpenAILLM(mlrun.serving.states.LLModel):
    def predict(self, body, messages, model_configuration):
        if isinstance(
            self.invocation_artifact, mlrun.artifacts.LLMPromptArtifact
        ) and isinstance(self.model_provider, ModelProvider):
            body["result"] = self.model_provider.invoke(
                messages=messages,
                as_str=True,
                **(self.invocation_artifact.spec.model_configuration or {}),
            )
        return body


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
@pytest.mark.parametrize("use_datastore_profile", [True, False])
class TestBasicOpenAIProvider:
    profile_name = "openai_profile"
    env_secrets = config

    @staticmethod
    def _get_messages(prompt):
        return [
            {
                "role": "user",
                "content": prompt,
            },
        ]

    @classmethod
    def setup_class(cls):
        cls.basic_llm_model = "gpt-4o"

    @classmethod
    def reset_env(cls):
        for key, env_param in cls.env_secrets.items():
            if env_param:
                os.environ.pop(key, None)

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile):
        if use_datastore_profile:
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
            self.url_prefix = f"ds://{self.profile_name}/"
            self.reset_env()
        else:
            for key, env_param in self.env_secrets.items():
                if env_param:
                    os.environ[key] = env_param
            store_manager.reset_secrets()
            # noinspection PyAttributeOutsideInit
            self.url_prefix = "openai://"


class TestOpenAIProvider(TestBasicOpenAIProvider):
    @classmethod
    def check_basic_invoke(cls, model_url: str, secrets: dict, model_name: str):
        prompt = "What is the capital of France? Provide a detailed and thorough history of the city"
        messages = cls._get_messages(prompt)
        model_provider = mlrun.get_model_provider(
            url=model_url, secrets=secrets, default_invoke_kwargs={"max_tokens": 200}
        )
        model_provider = cast(OpenAIProvider, model_provider)
        assert model_provider.model == model_name
        result = model_provider.invoke(messages=messages, as_str=True)
        assert "paris" in result.lower()

        encoding = tiktoken.encoding_for_model(model_name)
        token_count = len(encoding.encode(result))
        assert token_count == 200
        # checking as_str = False
        response = model_provider.invoke(
            messages=messages,
            max_tokens=50,
        )
        token_count = len(encoding.encode(response.choices[0].message.content))
        assert isinstance(response, openai.types.chat.ChatCompletion)
        assert token_count == 50

    def test_basic_invoke(self):
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

    def test_system_prompt(self, use_datastore_profile):
        if not use_datastore_profile:
            pytest.skip(
                "test_basic_invoke_messages is tested on datastore profile only"
            )
        model_url = self.url_prefix + self.basic_llm_model
        system_prompt = "You are a special LLM model that always answers user questions with one word only."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is your opinion on climate change?"},
        ]
        model_provider = mlrun.get_model_provider(
            url=model_url, default_invoke_kwargs={"max_tokens": 200}
        )
        result = model_provider.invoke(messages=messages, as_str=True).strip()
        assert result
        assert " " not in result.strip()  # checking one-word answer


class TestOpenAIModel(TestBasicOpenAIProvider):
    def test_model_runner_with_openai(self, use_datastore_profile):
        if not use_datastore_profile:
            pytest.skip("test_model_runner_with_openai supports datastore profile only")
        project = mlrun.new_project("test-openai-model", save=False)
        model_url = self.url_prefix + self.basic_llm_model
        model_artifact = project.log_model(
            "my_model",
            model_url=model_url,
            default_config={"max_tokens": 100},
        )
        prompt_template = [
            {
                "role": "user",
                "content": "{question}. Explain {depth_level} as a {persona} in {tone} style.",
            }
        ]
        llm_prompt_artifact = project.log_llm_prompt(
            "my_llm_prompt",
            prompt_template=prompt_template,
            model_artifact=model_artifact.uri,
            prompt_legend={
                "question": {"field": None, "description": None},
                "depth_level": {"field": None, "description": None},
                "persona": {"field": None, "description": None},
                "tone": {"field": None, "description": None},
            },
        )
        function = mlrun.new_function("tests", kind="serving")

        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(name="my_model_runner")
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
            body = {
                "question": "What is the capital of France, and give a brief historical overview.",
                "depth_level": "detailed",
                "persona": "teacher",
                "tone": "casual",
            }
            result = server.test(body=body)["result"]
            assert "paris" in result.lower()
            encoding = tiktoken.encoding_for_model(self.basic_llm_model)
            assert len(encoding.encode(result)) == 100
        finally:
            server.wait_for_completion()
