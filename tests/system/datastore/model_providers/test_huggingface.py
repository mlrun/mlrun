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
from transformers import AutoTokenizer

from mlrun.datastore.datastore_profile import (
    HuggingFaceProfile,
)
from tests.datastore.remote_model.remote_model_utils import (
    EXPECTED_RESULTS,
    INPUT_DATA,
    setup_remote_model_test,
)
from tests.system.base import TestMLRunSystem


@TestMLRunSystem.skip_test_if_env_not_configured
class TestOpenAIModelRunner(TestMLRunSystem):
    """Applying basic model endpoint CRUD operations through MLRun API"""

    project_name = "huggingface-system-test"
    image = "mlrun/mlrun"
    profile_name = "huggingface_profile"
    basic_llm_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

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

    def test_basic_openai_model_runner(self):
        self.setup_datastore_profile()
        mlrun_model_name = "sync_invoke_model"
        model_artifact, llm_prompt_artifact, function = setup_remote_model_test(
            self.project,
            self.model_url,
            mlrun_model_name=mlrun_model_name,
            image=self.image,
            requirements=["transformers==4.53.2", "torch==2.7.1"],
            default_config={"max_new_tokens": 100},
        )
        function.deploy()
        response = function.invoke(
            f"v2/models/{mlrun_model_name}/infer",
            json.dumps(INPUT_DATA[0]),
        )
        result = response["result"]
        assert EXPECTED_RESULTS[0] in result.lower()
        tokenizer = AutoTokenizer.from_pretrained(self.basic_llm_model)
        token_count = len(tokenizer.encode(result))
        # Extra token is due to the EOS token, which signals end of generation.
        assert token_count == 101
