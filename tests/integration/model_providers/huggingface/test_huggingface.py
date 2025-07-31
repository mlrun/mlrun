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
from typing import Optional, cast

import pytest
import yaml
from PIL import Image
from transformers import AutoTokenizer

import mlrun
import mlrun.artifacts
import mlrun.serving.states
from mlrun.datastore import store_manager
from mlrun.datastore.datastore_profile import (
    HuggingFaceProfile,
    register_temporary_client_datastore_profile,
)
from mlrun.datastore.model_provider.huggingface_provider import HuggingFaceProvider
from tests.datastore.remote_model.remote_model_utils import (
    EXPECTED_RESULTS,
    INPUT_DATA,
    formatted_messages,
    setup_remote_model_test,
)
from tests.integration.model_providers.model_providers_utils import (
    create_mocked_get_store_artifact,
)

here = os.path.dirname(__file__)
config = {}
config_file_path = os.path.join(here, "test-huggingface.yml")
if os.path.exists(config_file_path):
    with open(config_file_path) as yaml_file:
        config = yaml.safe_load(yaml_file).get("env", {})


@pytest.mark.skipif(
    not config.get("HF_TOKEN"),
    reason="test_configurable_model Requires HF_TOKEN",
)
class TestBasicHuggingFaceProvider:
    profile_name = "huggingface_profile"
    env_secrets = config

    @classmethod
    def setup_class(cls):
        cls.basic_llm_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        cls.system_prompt_llm_model = "microsoft/Phi-3-mini-4k-instruct"
        cls.image_path = os.path.join(os.path.dirname(__file__), "cat.jpg")

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
        self.url_prefix = "huggingface://"

    def setup_datastore_profile(self, task=None, model_kwargs=None):
        # noinspection PyAttributeOutsideInit
        self.profile = HuggingFaceProfile(
            name=self.profile_name,
            task=task or "text-generation",
            token=self.env_secrets.get("HF_TOKEN"),
            device=self.env_secrets.get("HF_DEVICE"),
            device_map=self.env_secrets.get("HF_DEVICE_MAP"),
            trust_remote_code=self.env_secrets.get("HF_TRUST_REMOTE_CODE"),
            model_kwargs=model_kwargs,
        )
        register_temporary_client_datastore_profile(self.profile)
        # noinspection PyAttributeOutsideInit
        self.url_prefix = f"ds://{self.profile_name}/"
        self.reset_env()


class TestHuggingFaceProvider(TestBasicHuggingFaceProvider):
    @classmethod
    def check_basic_invoke(
        cls,
        model_url: str,
        secrets: dict,
        model_name: str,
        expected_torch_dtype: Optional[str] = None,
    ):
        messages = [formatted_messages[0]]
        model_provider = mlrun.get_model_provider(
            url=model_url,
            secrets=secrets,
            default_invoke_kwargs={"max_new_tokens": 100},
        )
        model_provider = cast(HuggingFaceProvider, model_provider)
        assert model_provider.model == model_name
        result = model_provider.invoke(messages=messages, as_str=True)
        assert isinstance(result, str)
        assert EXPECTED_RESULTS[0] in result.lower()
        if expected_torch_dtype:
            assert model_provider.client.model.dtype == expected_torch_dtype

        token_count = len(model_provider.client.tokenizer.encode(result))
        # Extra token is due to the EOS token, which signals end of generation.
        assert token_count in (100, 101)
        # checking as_str = False
        response = model_provider.invoke(
            messages=messages,
            max_new_tokens=50,
        )
        assert isinstance(response, list)
        assert response[0]["generated_text"][0] == formatted_messages[0]

        assistant_response = response[0]["generated_text"][1]
        result = assistant_response["content"]
        token_count = len(model_provider.client.tokenizer.encode(result))
        assert assistant_response["role"] == "assistant"
        assert token_count in (50, 51)

    @pytest.mark.parametrize("cred_mode", ["profile", "env", "secrets"])
    def test_basic_invoke(self, cred_mode):
        # torch cannot be included in the dev image
        from torch import float16  # noqa

        secrets = {}
        if cred_mode == "profile":
            self.setup_datastore_profile(model_kwargs={"torch_dtype": float16})
            expected_torch_dtype = float16
        elif cred_mode == "secrets":
            self.reset_env()
            secrets = self.env_secrets.copy()
            secrets["HF_MODEL_KWARGS"] = {"torch_dtype": float16}
            expected_torch_dtype = float16
        else:
            expected_torch_dtype = None

        model_url = self.url_prefix + self.basic_llm_model
        self.check_basic_invoke(
            model_url=model_url,
            secrets=secrets,
            model_name=self.basic_llm_model,
            expected_torch_dtype=expected_torch_dtype,
        )

    def test_configurable_model(self):
        configurable_model = mlrun.mlconf.model_providers.huggingface_default_model
        if not configurable_model:
            pytest.skip(
                "model_providers.huggingface_default_model is not configured in conf, cannot perform the test"
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
        #  Tinyllama does not function well with system prompts, but is free to use without hf_key.
        model_url = self.url_prefix + self.system_prompt_llm_model
        system_prompt = "You are a special LLM model that always answers user questions with one word only."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is your opinion on climate change?"},
        ]
        model_provider = mlrun.get_model_provider(
            url=model_url, default_invoke_kwargs={"max_new_tokens": 200}
        )
        result = model_provider.invoke(messages=messages, as_str=True)
        assert isinstance(result, str)
        result = result.strip()
        assert result
        assert " " not in result.strip()  # checking one-word answer

    @pytest.mark.parametrize("use_datastore_profile", [True, False])
    def test_custom_invoke(self, use_datastore_profile):
        model_name = "microsoft/resnet-50"
        task = "image-classification"
        secrets = None
        top_k = 2

        if use_datastore_profile:
            self.setup_datastore_profile(task=task)
        else:
            secrets = {"HF_TASK": task}
        model_url = self.url_prefix + model_name
        model_provider = mlrun.get_model_provider(
            url=model_url, secrets=secrets, default_invoke_kwargs={"top_k": top_k}
        )
        image = Image.open(self.image_path)
        classification_results = model_provider.custom_invoke(inputs=image)
        assert len(classification_results) == top_k
        assert "cat" in classification_results[0]["label"]

        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="HuggingFaceProvider.invoke supports text-generation task only",
        ):
            model_provider.invoke(messages=[formatted_messages[0]])

        with pytest.raises(
            mlrun.errors.MLRunInvalidArgumentError,
            match="Huggingface operation must inherit from 'Pipeline' object",
        ):
            model_provider.custom_invoke(
                operation=lambda *args, **kwargs: None, messages=[formatted_messages[0]]
            )


class TestHuggingFaceAIModel(TestBasicHuggingFaceProvider):
    def test_hf_model_runner(self):
        project = mlrun.new_project("test-hf-model", save=False)
        model_url = self.url_prefix + self.basic_llm_model
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            project, model_url, default_config={"max_new_tokens": 100}
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
            tokenizer = AutoTokenizer.from_pretrained(self.basic_llm_model)
            token_count = len(tokenizer.encode(result))
            # Extra token is due to the EOS token, which signals end of generation.
            assert token_count in (100, 101)
        finally:
            server.wait_for_completion()
