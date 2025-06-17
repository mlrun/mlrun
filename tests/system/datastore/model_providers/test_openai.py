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


class TestOpenAIModelRunner(TestMLRunSystem):
    """Applying basic model endpoint CRUD operations through MLRun API"""

    project_name = "openai_model_runner"
    image = "mlrun/mlrun"
    mandatory_env_vars = super().mandatory_env_vars
    model = "gpt-4o"
    profile_name = "my_openai_profile"

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.basic_llm_model = "gpt-4o"
        # cls.openai_url = os.environ.get("OPENAI_BASE_URL")

    @pytest.fixture(autouse=True)
    def setup_before_each_test(self, use_datastore_profile):
        # if use_datastore_profile:
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
        model_artifact = project.log_model(
            "my_model",
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
        # TODO replace mock with real operation

        #     server = function.to_mock_server()
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
