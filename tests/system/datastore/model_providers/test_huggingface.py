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

import json
import os

import pytest

import mlrun.serving.states
from mlrun.datastore.datastore_profile import (
    HuggingFaceProfile,
)
from tests.datastore.remote_model.remote_model_utils import (
    BATCH_INPUT_DATA,
    EXPECTED_RESULTS,
    PROMPT_LEGEND,
    PROMPT_TEMPLATE,
    retry_on_content_mismatch,
    setup_remote_model_test,
    validate_llm_batch_response_system,
    validate_llm_single_response,
)
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestHuggingFaceModelRunner(TestMLRunSystem):
    """Applying basic model endpoint CRUD operations through MLRun API"""

    project_name = "huggingface-system-test"
    image = "mlrun/mlrun"
    profile_name = "huggingface_profile"
    basic_llm_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    image_classification_model = "microsoft/resnet-50"

    def setup_datastore_profile(self, task=None, model_name=None):
        # noinspection PyAttributeOutsideInit
        self.profile = HuggingFaceProfile(
            name=self.profile_name,
            task=task or "text-generation",
            token=os.environ.get("HF_TOKEN"),
            device=os.environ.get("HF_DEVICE"),
            device_map=os.environ.get("HF_DEVICE_MAP"),
        )
        model_name = model_name or self.basic_llm_model
        self.project.register_datastore_profile(self.profile)
        self.url_prefix = f"ds://{self.profile_name}/"
        self.model_url = self.url_prefix + model_name

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["naive", "process_pool", "dedicated_process", "thread_pool"],
    )
    def test_basic_huggingface_model_runner(self, execution_mechanism):
        self.setup_datastore_profile()
        mlrun_model_name = "sync_invoke_model"
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            self.project,
            self.model_url,
            mlrun_model_name=mlrun_model_name,
            image=self.image,
            requirements=[
                "--extra-index-url",
                "https://download.pytorch.org/whl/cpu",
                "torch==2.8.0+cpu",
                "transformers==4.56.2",
                "pillow~=11.3",
            ],
            default_config={"max_new_tokens": 50},
            execution_mechanism=execution_mechanism,
        )

        # Running models requires higher CPU for this pod.
        # The default Nuclio resource configuration is:
        # {"requests": {"cpu": "25m", "memory": "1Mi"}, "limits": {"cpu": "2", "memory": "20Gi"}}
        function.spec.resources = {
            "limits": {"cpu": "6", "memory": "20Gi"},
            "requests": {"cpu": "25m", "memory": "1Mi"},
        }
        function.spec.max_replicas = (
            1  # to avoid allocating extended resources to multiple pods
        )
        # Set workers=None to avoid using the default value of 8 workers
        function.with_http(gateway_timeout=600, worker_timeout=500, workers=None)
        function.spec.readiness_timeout = 600

        function.deploy()

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.basic_llm_model)

        def _test_single():
            response = function.invoke(
                f"v2/models/{mlrun_model_name}/infer",
                json.dumps(BATCH_INPUT_DATA[0]),
            )["output"]
            validate_llm_single_response(
                response, EXPECTED_RESULTS[0], tokenizer, min_tokens=45, max_tokens=51
            )

        retry_on_content_mismatch(_test_single)

        def _test_batch():
            batch_response = function.invoke(
                f"v2/models/{mlrun_model_name}/infer",
                json.dumps(BATCH_INPUT_DATA),
            )
            validate_llm_batch_response_system(
                batch_response,
                EXPECTED_RESULTS,
                tokenizer,
                min_tokens=45,
                max_tokens=51,
            )

        retry_on_content_mismatch(_test_batch)

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["naive", "process_pool", "dedicated_process", "thread_pool"],
    )
    def test_custom_huggingface_model_runner(self, execution_mechanism):
        self.setup_datastore_profile(
            task="image-classification", model_name=self.image_classification_model
        )

        # cat.jpg – free for personal & commercial use (Unsplash license)
        # https://unsplash.com/photos/brown-tabby-cat-on-white-stairs-mJaD10XeD7w
        # https://unsplash.com/license
        image_local_path = os.path.join(self.assets_path, "cat.jpg")
        artifact = self.project.log_artifact(
            "my_artifact", local_path=image_local_path, upload=True
        )
        v3io_path = artifact.get_target_path()

        mlrun_model_name = "custom_hf_model"
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            self.project,
            self.model_url,
            mlrun_model_name=mlrun_model_name,
            image=self.image,
            requirements=[
                "--extra-index-url",
                "https://download.pytorch.org/whl/cpu",
                "torch==2.8.0+cpu",
                "transformers==4.56.2",
                "pillow~=11.3",
            ],
            default_config={"top_k": 2},
            execution_mechanism=execution_mechanism,
            model_class="MyHuggingFaceCustom",
            include_llm_artifact=False,
        )

        function.spec.max_replicas = 1  # to avoid allocating resources to multiple pods
        function.deploy()
        results = function.invoke(
            f"v2/models/{mlrun_model_name}/infer",
            {"input": v3io_path},
        )["result"]
        assert len(results) == 2
        # # Verify the top result contains 'cat' (assuming the test image is a cat)
        assert "cat" in results[0]["label"].lower()
        # Verify each result has the expected structure
        for result in results:
            assert "label" in result
            assert "score" in result
            assert isinstance(result["score"], float)
            assert 0 <= result["score"] <= 1

    @pytest.mark.parametrize(
        "execution_mechanism",
        ["naive", "process_pool", "dedicated_process", "thread_pool"],
    )
    def test_hf_2_models(self, execution_mechanism):
        if not os.getenv("HF_TOKEN"):
            pytest.skip("test_hf_2_models Requires HF_TOKEN")
        self.setup_datastore_profile()
        llm_model2 = "google/gemma-2b-it"
        ep_name = "ep"
        second_ep_name = "ep2"
        model_class = "mlrun.serving.states.LLModel"
        second_model_url = self.url_prefix + llm_model2

        model1 = self.project.log_model(
            "model_key", model_url=self.model_url, default_config={"max_new_tokens": 50}
        )

        model2 = self.project.log_model(
            "model_key2",
            model_url=second_model_url,
            default_config={"max_new_tokens": 50},
        )
        llm_art1 = self.project.log_llm_prompt(
            "llm_artifact",
            prompt_template=PROMPT_TEMPLATE,
            description="remote_model_open_ai-llm-prompt-prompt",
            prompt_legend=PROMPT_LEGEND,
            model_artifact=model1,
        )

        llm_art2 = self.project.log_llm_prompt(
            "llm_artifact2",
            prompt_template=PROMPT_TEMPLATE,
            description="remote_model_open_ai-llm-prompt-prompt",
            prompt_legend=PROMPT_LEGEND,
            model_artifact=model2,
        )

        function = self.project.set_function(
            name="function_with_llm_hf",
            kind="serving",
            requirements=[
                "--extra-index-url",
                "https://download.pytorch.org/whl/cpu",
                "torch==2.8.0+cpu",
                "transformers==4.56.2",
                "pillow~=11.3",
            ],
            image=self.image,
        )

        function.spec.resources = {
            "limits": {"cpu": "7", "memory": "20Gi"},
            "requests": {"cpu": "25m", "memory": "1Mi"},
        }
        function.spec.readiness_timeout = 600
        function.spec.max_replicas = (
            1  # to avoid allocating extended resources to multiple pods
        )
        # Set workers=None to avoid using the default value of 8 workers
        function.with_http(gateway_timeout=1100, worker_timeout=1000, workers=None)
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
        function.deploy()

        results = function.invoke("/", json.dumps(BATCH_INPUT_DATA[0]))
        # Verify we got the expected number of results

        assert sorted(list(results.keys())) == sorted([ep_name, second_ep_name])
        for model_result in results.values():
            assert "paris" in model_result["output"]["answer"].lower()
