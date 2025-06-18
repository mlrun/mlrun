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
    DatastoreProfileOpenAI,
    register_temporary_client_datastore_profile,
)
from mlrun.datastore.model_providers import ModelProvider
from mlrun.serving import ModelRunnerStep
from tests.system.base import TestMLRunSystem


class MyOpenAILLM(mlrun.serving.states.Model):
    execution_mechanism = "naive"

    def predict(self, body):
        if isinstance(
            self.invocation_artifact, mlrun.artifacts.LLMPromptArtifact
        ) and isinstance(self.model, ModelProvider):
            prompt = self.enrich_prompt(body)
            body["result"] = self.model.invoke(
                prompt=prompt,
                **(self.invocation_artifact.spec.model_configuration or {}),
            )
        return body

    def enrich_prompt(self, body) -> str:
        if isinstance(self.invocation_artifact, mlrun.artifacts.LLMPromptArtifact):
            prompt_template = self.invocation_artifact.spec.prompt_string
            needed_params = ["question", "depth_level", "persona", "tone"]
            sub_dict = {k: body[k] for k in needed_params if k in body}
            return prompt_template.format(**sub_dict)
        return body["prompt"]


def get_missing_openai_env_variables():
    return [env_key for env_key in ["OPENAI_BASE_URL", "OPENAI_API_KEY"] if not os.environ.get(env_key)]


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.parametrize("use_datastore_profile", [True, False])
class TestOpenAIModelRunner(TestMLRunSystem):
    """Applying basic model endpoint CRUD operations through MLRun API"""

    project_name = "openai-model-runner"
    image = "mlrun/mlrun"
    mandatory_env_vars = TestMLRunSystem.mandatory_env_vars + ["OPENAI_BASE_URL"]
    model = "gpt-4o"
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
        # cls.openai_url = os.environ.get("OPENAI_BASE_URL")

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile):
        self.profile = DatastoreProfileOpenAI(
            name=self.profile_name,
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization=os.environ.get("OPENAI_ORG_ID"),
            project=os.environ.get("OPENAI_PROJECT_ID"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            timeout=os.environ.get("OPENAI_TIMEOUT"),
            max_retries=os.environ.get("OPENAI_MAX_RETRIES"),
        )
        register_temporary_client_datastore_profile(self.profile)
        self.url_prefix = f"ds://{self.profile_name}/"
        # self.reset_env()
        self.model_url = self.url_prefix + self.basic_llm_model

    def test_basic_openai_model_runner(self):
        project = mlrun.new_project("system-test-openai-model", save=False)
        mlrun_model_name = "my_model"
        model_artifact = project.log_model(
            mlrun_model_name,
            model_url=self.model_url,
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
        model_runner_step.add_model(
            model_class="MyOpenAILLM",
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
        print(response)
