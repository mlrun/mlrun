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
import tiktoken

import mlrun
import mlrun.artifacts
import mlrun.serving.states
from mlrun.datastore.datastore_profile import (
    OpenAIProfile,
)
from mlrun.serving import ModelRunnerStep
from tests.system.base import TestMLRunSystem


def get_missing_openai_env_variables():
    return [
        env_key
        for env_key in ["OPENAI_BASE_URL", "OPENAI_API_KEY"]
        if not os.environ.get(env_key)
    ]


@TestMLRunSystem.skip_test_if_env_not_configured
class TestOpenAIModelRunner(TestMLRunSystem):
    """Applying basic model endpoint CRUD operations through MLRun API"""

    project_name = "openai-system-test"
    image = "mlrun/mlrun"
    profile_name = "my_openai_profile"

    @classmethod
    def setup_class(cls):
        super().setup_class()
        missing_env_variables = get_missing_openai_env_variables()
        if missing_env_variables:
            pytest.skip(
                f"The following snowflake keys are missing: {missing_env_variables}"
            )
        cls.basic_llm_model = "gpt-4o"

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self):
        self.profile = OpenAIProfile(
            name=self.profile_name,
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization=os.environ.get("OPENAI_ORG_ID"),
            project=os.environ.get("OPENAI_PROJECT_ID"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            timeout=os.environ.get("OPENAI_TIMEOUT"),
            max_retries=os.environ.get("OPENAI_MAX_RETRIES"),
        )
        self.project.register_datastore_profile(self.profile)
        self.url_prefix = f"ds://{self.profile_name}/"
        self.model_url = self.url_prefix + self.basic_llm_model

    def test_basic_openai_model_runner(self):
        mlrun_model_name = "my_model"
        model_artifact = self.project.log_model(
            mlrun_model_name,
            model_url=self.model_url,
            default_config={"max_tokens": 100},
        )
        prompt_template = [
            {
                "role": "user",
                "content": "{question}. Explain {depth_level} as a {persona} in {tone} style.",
            }
        ]
        llm_prompt_artifact = self.project.log_llm_prompt(
            "my_llm_prompt",
            prompt_template=prompt_template,
            model_artifact=model_artifact,
            prompt_legend={
                "question": {"field": None, "description": None},
                "depth_level": {"field": None, "description": None},
                "persona": {"field": None, "description": None},
                "tone": {"field": None, "description": None},
            },
        )
        function = mlrun.code_to_function(
            name="tests",
            kind="serving",
            tag="latest",
            project=self.project_name,
            filename=os.path.relpath(str(self.assets_path / "models.py")),
            image=self.image,
            requirements=["openai==1.77.0"],
        )
        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(name="my_model_runner")
        model_runner_step.add_model(
            model_class="MyOpenAILLM",
            execution_mechanism="naive",
            endpoint_name="my_endpoint",
            model_artifact=llm_prompt_artifact,
        )
        graph.to(model_runner_step).respond()

        function.deploy()

        body = {
            "question": "What is the capital of France, and give a brief historical overview.",
            "depth_level": "detailed",
            "persona": "teacher",
            "tone": "casual",
        }
        response = function.invoke(
            f"v2/models/{mlrun_model_name}/infer",
            json.dumps(body),
        )
        result = response["result"]
        assert "paris" in result.lower()
        encoding = tiktoken.encoding_for_model(self.basic_llm_model)
        token_count = len(encoding.encode(result))
        assert token_count == 100
