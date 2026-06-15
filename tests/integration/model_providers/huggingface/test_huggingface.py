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

import inspect
import os
import time
import unittest.mock
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest
import yaml
from PIL import Image

import mlrun
import mlrun.artifacts
import mlrun.serving.states
from mlrun.datastore import store_manager
from mlrun.datastore.datastore_profile import (
    HuggingFaceProfile,
    register_temporary_client_datastore_profile,
)
from mlrun.datastore.model_provider.huggingface_provider import HuggingFaceProvider
from mlrun.datastore.model_provider.model_provider import (
    InvokeResponseFormat,
    UsageResponseKeys,
)
from tests.datastore.remote_model.remote_model_utils import (
    BATCH_INPUT_DATA,
    EXPECTED_RESULTS,
    PROMPT_LEGEND,
    PROMPT_TEMPLATE,
    LLMContentMismatchError,
    create_mocked_get_store_artifact,
    formatted_messages,
    retry_on_content_mismatch,
    setup_remote_model_test,
    validate_llm_batch_response_system,
    validate_llm_single_response,
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
    # Max retry attempts for LLM content mismatches (non-deterministic failures)
    max_retries = 2

    @classmethod
    def setup_class(cls):
        cls.basic_llm_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        cls.image_classification_model = "microsoft/resnet-50"
        cls.system_prompt_llm_model = "microsoft/Phi-3-mini-4k-instruct"

        # cat.jpg – free for personal & commercial use (Unsplash license):
        # https://unsplash.com/photos/brown-tabby-cat-on-white-stairs-mJaD10XeD7w
        # https://unsplash.com/license
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

    @staticmethod
    def _check_string_response(result: str, expected_result: str, tokenizer) -> None:
        assert isinstance(result, str)
        token_count = len(tokenizer.encode(result))
        assert 95 <= token_count <= 101

        if expected_result not in result.lower():
            raise LLMContentMismatchError(
                f"Expected '{expected_result}' not found in LLM answer: '{result[:100]}...'"
            )

    @staticmethod
    def _check_full_response(
        result: list,
        expected_input_message: dict,
        expected_result: str,
        tokenizer,
        min_tokens: int = 95,
        max_tokens: int = 101,
    ) -> None:
        assert isinstance(result, list)
        assert result[0]["generated_text"][0] == expected_input_message
        assistant_response = result[0]["generated_text"][1]
        assert assistant_response["role"] == "assistant"
        token_count = len(tokenizer.encode(assistant_response["content"]))
        assert min_tokens <= token_count <= max_tokens

        content = assistant_response["content"].lower()
        if expected_result not in content:
            raise LLMContentMismatchError(
                f"Expected '{expected_result}' not found in LLM answer: '{content[:100]}...'"
            )

    @staticmethod
    def _check_usage_response(
        result: dict,
        expected_result: str,
        messages: list[dict] | None = None,
        tokenizer=None,
        min_tokens: int = 95,
        max_tokens: int = 101,
    ) -> None:
        assert isinstance(result, dict)
        assert UsageResponseKeys.ANSWER in result
        assert UsageResponseKeys.USAGE in result
        assert (
            min_tokens
            <= result[UsageResponseKeys.USAGE]["completion_tokens"]
            <= max_tokens
        )
        assert (
            result[UsageResponseKeys.USAGE]["total_tokens"]
            == result[UsageResponseKeys.USAGE]["prompt_tokens"]
            + result[UsageResponseKeys.USAGE]["completion_tokens"]
        )

        if messages is not None and tokenizer is not None:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = len(tokenizer.encode(prompt))
            assert result[UsageResponseKeys.USAGE]["prompt_tokens"] == prompt_tokens

        answer = result[UsageResponseKeys.ANSWER].lower()
        if expected_result not in answer:
            raise LLMContentMismatchError(
                f"Expected '{expected_result}' not found in LLM answer: '{answer[:100]}...'"
            )

    def setup_datastore_profile(self, task=None, model_kwargs=None):
        # noinspection PyAttributeOutsideInit
        raw_max_workers = self.env_secrets.get("HF_MAX_WORKERS")
        self.profile = HuggingFaceProfile(
            name=self.profile_name,
            task=task or "text-generation",
            token=self.env_secrets.get("HF_TOKEN"),
            endpoint=self.env_secrets.get("HF_ENDPOINT"),
            device=self.env_secrets.get("HF_DEVICE") or "cpu",
            device_map=self.env_secrets.get("HF_DEVICE_MAP"),
            trust_remote_code=self.env_secrets.get("HF_TRUST_REMOTE_CODE"),
            model_kwargs=model_kwargs,
            max_workers=int(raw_max_workers) if raw_max_workers else None,
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
        expected_torch_dtype: str | None = None,
    ):
        messages = [formatted_messages[0]]
        model_provider = mlrun.get_model_provider(
            url=model_url,
            secrets=secrets,
            default_invoke_kwargs={"max_new_tokens": 100},
        )
        model_provider = cast(HuggingFaceProvider, model_provider)
        assert model_provider.model == model_name

        for attempt in range(cls.max_retries + 1):
            try:
                result = model_provider.invoke(
                    messages=messages,
                    invoke_response_format=InvokeResponseFormat.STRING,
                )
                cls._check_string_response(
                    result, EXPECTED_RESULTS[0], model_provider.client.tokenizer
                )
                break
            except LLMContentMismatchError as e:
                if attempt == cls.max_retries:
                    raise
                print(
                    f"LLM content mismatch in STRING (attempt {attempt + 1}/{cls.max_retries + 1}): {e}"
                )

        if expected_torch_dtype:
            assert model_provider.client.model.dtype == expected_torch_dtype

        for attempt in range(cls.max_retries + 1):
            try:
                response = model_provider.invoke(
                    messages=messages,
                    max_new_tokens=50,
                )
                cls._check_full_response(
                    response,
                    formatted_messages[0],
                    EXPECTED_RESULTS[0],
                    model_provider.client.tokenizer,
                    min_tokens=45,
                    max_tokens=51,
                )
                break
            except LLMContentMismatchError as e:
                if attempt == cls.max_retries:
                    raise
                print(
                    f"LLM content mismatch in FULL (attempt {attempt + 1}/{cls.max_retries + 1}): {e}"
                )

        for attempt in range(cls.max_retries + 1):
            try:
                response = model_provider.invoke(
                    messages=messages,
                    max_new_tokens=50,
                    invoke_response_format=InvokeResponseFormat.USAGE,
                )
                validate_llm_single_response(
                    response,
                    EXPECTED_RESULTS[0],
                    model_provider.client.tokenizer,
                    min_tokens=45,
                    max_tokens=51,
                )
                break

            except LLMContentMismatchError as e:
                if attempt == cls.max_retries:
                    raise
                print(
                    f"LLM content mismatch in USAGE (attempt {attempt + 1}/{cls.max_retries + 1}): {e}"
                )

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

    @pytest.mark.parametrize(
        "invoke_response_format",
        [
            InvokeResponseFormat.STRING,
            InvokeResponseFormat.FULL,
            InvokeResponseFormat.USAGE,
        ],
    )
    @pytest.mark.parametrize("batch_size", [None, 4])
    def test_batch_invoke(self, invoke_response_format, batch_size):
        self.setup_datastore_profile()
        model_url = self.url_prefix + self.basic_llm_model
        default_invoke_kwargs = {"max_new_tokens": 100}
        # if not set, batch size is according to huggingface_default_batch_size in mlrun config
        if batch_size is not None:
            default_invoke_kwargs["batch_size"] = batch_size
        model_provider = mlrun.get_model_provider(
            url=model_url, default_invoke_kwargs=default_invoke_kwargs
        )

        model_provider = cast(HuggingFaceProvider, model_provider)

        messages_list = [[msg] for msg in formatted_messages]

        for attempt in range(self.max_retries + 1):
            try:
                results = model_provider.invoke(
                    messages=messages_list,
                    invoke_response_format=invoke_response_format,
                )

                assert isinstance(results, list)
                assert len(results) == len(formatted_messages)

                for i, result in enumerate(results):
                    if invoke_response_format == InvokeResponseFormat.STRING:
                        self._check_string_response(
                            result, EXPECTED_RESULTS[i], model_provider.client.tokenizer
                        )
                    elif invoke_response_format == InvokeResponseFormat.FULL:
                        self._check_full_response(
                            result,
                            formatted_messages[i],
                            EXPECTED_RESULTS[i],
                            model_provider.client.tokenizer,
                        )
                    elif invoke_response_format == InvokeResponseFormat.USAGE:
                        validate_llm_single_response(
                            result, EXPECTED_RESULTS[i], model_provider.client.tokenizer
                        )

                break

            except LLMContentMismatchError as e:
                if attempt == self.max_retries:
                    raise
                print(
                    f"LLM content mismatch in batch (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
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
        result = model_provider.invoke(
            messages=messages, invoke_response_format=InvokeResponseFormat.STRING
        )
        assert isinstance(result, str)
        result = result.strip()
        assert result
        assert " " not in result.strip()  # checking one-word answer

    @pytest.mark.parametrize("use_datastore_profile", [True, False])
    def test_custom_invoke(self, use_datastore_profile):
        model_name = self.image_classification_model
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

    def test_invoke_stream(self):
        """Streaming yields non-empty tokens that form a coherent answer."""
        model_url = self.url_prefix + self.basic_llm_model
        provider = mlrun.get_model_provider(
            url=model_url, default_invoke_kwargs={"max_new_tokens": 60}
        )
        messages = [formatted_messages[0]]
        tokens = list(provider.invoke_stream(messages=messages))
        assert len(tokens) > 1, "Expected multiple streamed tokens"
        full_text = "".join(tokens)
        assert EXPECTED_RESULTS[0] in full_text.lower()
        token_count = len(provider.client.tokenizer.encode(full_text))
        assert 50 <= token_count <= 70


class TestHuggingFaceMRS(TestBasicHuggingFaceProvider):
    @pytest.mark.parametrize(
        "execution_mechanism",
        ["naive", "process_pool", "dedicated_process", "thread_pool"],
    )
    def test_hf_model_runner(self, execution_mechanism):
        project = mlrun.new_project("test-hf-model", save=False)
        model_url = self.url_prefix + self.basic_llm_model
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            project,
            model_url,
            default_config={"max_new_tokens": 100},
            execution_mechanism=execution_mechanism,
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
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.basic_llm_model)

            def _test():
                response = server.test(body=BATCH_INPUT_DATA[0])["output"]
                validate_llm_single_response(response, EXPECTED_RESULTS[0], tokenizer)

            retry_on_content_mismatch(_test, self.max_retries + 1)

        finally:
            server.wait_for_completion()

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["process_pool", "dedicated_process", "naive", "thread_pool"],
    )
    def test_model_runner_batch_with_hf(self, execution_mechanism):
        """Test batch processing of multiple events with HuggingFace model"""
        project = mlrun.new_project("test-hf-model-batch", save=False)
        model_url = self.url_prefix + self.basic_llm_model
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            project,
            model_url,
            execution_mechanism=execution_mechanism,
            default_config={"max_new_tokens": 100},
        )
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
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.basic_llm_model)

            def _test():
                batch_response = server.test(body=BATCH_INPUT_DATA)
                validate_llm_batch_response_system(
                    batch_response, EXPECTED_RESULTS, tokenizer
                )

            retry_on_content_mismatch(_test, self.max_retries + 1)

        finally:
            server.wait_for_completion()

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["naive", "process_pool", "dedicated_process", "thread_pool"],
    )
    def test_model_runner_batch_step_with_hf(self, execution_mechanism):
        from transformers import AutoTokenizer

        project = mlrun.new_project("test-hf-batch-step", save=False)
        model_url = self.url_prefix + self.basic_llm_model

        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            project,
            model_url,
            execution_mechanism=execution_mechanism,
            default_config={"max_new_tokens": 100},
            batch_step=True,
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
            tokenizer = AutoTokenizer.from_pretrained(self.basic_llm_model)

            # Send events concurrently with staggered timing
            def send_event(event, delay):
                time.sleep(delay)
                return server.test(body=event)

            # Verify each response has correct structure
            def _test():
                with ThreadPoolExecutor(max_workers=len(BATCH_INPUT_DATA)) as executor:
                    futures = [
                        executor.submit(send_event, event, i * 0.1)
                        for i, event in enumerate(BATCH_INPUT_DATA)
                    ]
                    batch_response = [future.result() for future in futures]
                validate_llm_batch_response_system(
                    batch_response, EXPECTED_RESULTS, tokenizer
                )

            retry_on_content_mismatch(_test, self.max_retries + 1)

        finally:
            server.wait_for_completion()

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["naive", "process_pool", "dedicated_process", "thread_pool"],
    )
    def test_hf_custom_model_runner(self, execution_mechanism):
        project = mlrun.new_project("test-hf-model", save=False)
        self.setup_datastore_profile(task="image-classification")
        model_url = self.url_prefix + self.image_classification_model
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            project,
            model_url,
            execution_mechanism=execution_mechanism,
            model_class="tests.datastore.remote_model.remote_model_utils.MyHuggingFaceCustom",
            default_config={"top_k": 2},
            include_llm_artifact=False,
        )
        # # Mock needed since no artifact is saved in this test, so retrieval by URI isn't possible.
        # # Mocked function used to verify artifact URI is passed correctly.
        #
        mocked_get_store_artifact = create_mocked_get_store_artifact(
            {
                model_artifact.uri: model_artifact,
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
            results = server.test(body={"input": self.image_path})["result"]
            # Verify we got the expected number of results
            assert len(results) == 2
            # # Verify the top result contains 'cat' (assuming the test image is a cat)
            assert "cat" in results[0]["label"].lower()
            # Verify each result has the expected structure
            for result in results:
                assert "label" in result
                assert "score" in result
                assert isinstance(result["score"], float)
                assert 0 <= result["score"] <= 1
        finally:
            server.wait_for_completion()

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["naive", "process_pool", "dedicated_process", "thread_pool"],
    )
    def test_hf_2_models(self, execution_mechanism):
        proj_obj = mlrun.new_project("test-hf-model", save=False)
        llm_model2 = "google/gemma-2b-it"
        ep_name = "ep1"
        second_ep_name = "ep2"
        model_class = "mlrun.serving.states.LLModel"

        model_url = self.url_prefix + self.basic_llm_model
        second_model_url = self.url_prefix + llm_model2

        model1 = proj_obj.log_model(
            "model_key", model_url=model_url, default_config={"max_new_tokens": 50}
        )

        model2 = proj_obj.log_model(
            "model_key2",
            model_url=second_model_url,
            default_config={"max_new_tokens": 50},
        )
        llm_art1 = proj_obj.log_llm_prompt(
            "llm_artifact",
            prompt_template=PROMPT_TEMPLATE,
            description="remote_model_open_ai-llm-prompt-prompt",
            prompt_legend=PROMPT_LEGEND,
            model_artifact=model1,
        )

        llm_art2 = proj_obj.log_llm_prompt(
            "llm_artifact2",
            prompt_template=PROMPT_TEMPLATE,
            description="remote_model_open_ai-llm-prompt-prompt",
            prompt_legend=PROMPT_LEGEND,
            model_artifact=model2,
        )

        function = proj_obj.set_function(
            name="function_with_llm_hf",
            kind="serving",
            requirements=[
                "--extra-index-url",
                "https://download.pytorch.org/whl/cpu",
                "torch==2.7.1+cpu",
                "transformers==4.53.2",
                "pillow~=11.3",
            ],
        )
        model_runner_step = mlrun.serving.states.ModelRunnerStep(name="mrs")
        model_runner_step.add_model(
            endpoint_name=ep_name,
            model_class=model_class,
            execution_mechanism=execution_mechanism,
            model_artifact=llm_art1,
            result_path="output",
        )
        model_runner_step.add_model(
            endpoint_name=second_ep_name,
            model_class=model_class,
            execution_mechanism=execution_mechanism,
            model_artifact=llm_art2,
            result_path="output",
        )

        llm_graph = function.set_topology("flow", engine="async")
        llm_graph.to(model_runner_step).respond()
        mocked_get_store_artifact = create_mocked_get_store_artifact(
            {
                model1.uri: model1,
                model2.uri: model2,
                llm_art1.uri: llm_art1,
                llm_art2.uri: llm_art2,
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
            results = server.test(body=BATCH_INPUT_DATA[0])
            # Verify we got the expected number of results

            assert sorted(list(results.keys())) == sorted([ep_name, second_ep_name])
            for model_result in results.values():
                assert "paris" in model_result["output"]["answer"].lower()
        finally:
            server.wait_for_completion()

    def test_hf_model_runner_streaming(self):
        """Test streaming through MRS with HuggingFace provider."""
        project = mlrun.new_project("test-hf-streaming", save=False)
        model_url = self.url_prefix + self.basic_llm_model
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            project,
            model_url,
            default_config={"max_new_tokens": 60},
            execution_mechanism="naive",
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
            assert inspect.isgenerator(response), (
                f"Expected generator, got {type(response)}"
            )
            response = "".join(response)
            assert EXPECTED_RESULTS[0] in response.lower()
        finally:
            server.wait_for_completion()
