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
from transformers import AutoTokenizer

from mlrun.datastore.datastore_profile import (
    HuggingFaceProfile,
)
from mlrun.datastore.model_provider.model_provider import UsageResponseKeys
from tests.datastore.remote_model.remote_model_utils import (
    EXPECTED_RESULTS,
    INPUT_DATA,
    setup_remote_model_test,
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
                "torch==2.7.1+cpu",
                "transformers==4.53.2",
                "pillow~=11.3",
            ],
            default_config={"max_new_tokens": 50},
            execution_mechanism=execution_mechanism,
        )

        # Running models requires higher CPU for this pod.
        # The default Nuclio resource configuration is:
        # {"requests": {"cpu": "25m", "memory": "1Mi"}, "limits": {"cpu": "2", "memory": "20Gi"}}
        function.spec.resources = {
            "limits": {"cpu": "5", "memory": "30Gi"},
            "requests": {"cpu": "3", "memory": "1Mi"},
        }
        function.spec.max_replicas = (
            1  # to avoid allocating extended resources to multiple pods
        )
        function.deploy()
        response = function.invoke(
            f"v2/models/{mlrun_model_name}/infer",
            json.dumps(INPUT_DATA[0]),
        )["output"]

        assert len(response) == 2
        answer = response[UsageResponseKeys.ANSWER]
        assert EXPECTED_RESULTS[0] in answer.lower()
        tokenizer = AutoTokenizer.from_pretrained(self.basic_llm_model)
        token_count = len(tokenizer.encode(answer))
        # Token count may be lower due to early stopping or slightly higher (e.g., 101)
        # due to internal EOS or tokenizer behavior, so we assert within this range.
        assert 45 <= token_count <= 51

        stats = response[UsageResponseKeys.USAGE]
        assert stats["completion_tokens"] == token_count
        assert stats["prompt_tokens"] > 0
        assert (
            stats["total_tokens"] == stats["completion_tokens"] + stats["prompt_tokens"]
        )

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
                "torch==2.7.1+cpu",
                "transformers==4.53.2",
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
