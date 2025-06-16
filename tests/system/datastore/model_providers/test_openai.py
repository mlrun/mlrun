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

import mlrun.artifacts
import mlrun.serving.states
from mlrun.serving import ModelRunnerStep
from tests.system.base import TestMLRunSystem


class MyOpenAILLM(mlrun.serving.states.ModelProviderModel):
    def predict(self, body):
        prompt = self.enrich_prompt(body)
        body["result"] = self.model.basic_llm_invoke(
            prompt=prompt, **self.invocation_artifact.spec.model_configuration
        )
        return body

    def enrich_prompt(self, body) -> str:
        if isinstance(self.invocation_artifact, mlrun.artifacts.LLMPromptArtifact):
            prompt_template = self.invocation_artifact.spec.prompt_string
            needed_params = ["question", "level", "length"]
            sub_dict = {k: body[k] for k in needed_params if k in body}
            return prompt_template.format(**sub_dict)
        return body["question"]


class TestOpenAIModelRunner(TestMLRunSystem):
    """Applying basic model endpoint CRUD operations through MLRun API"""

    project_name = "openai_model_runner"
    image = "mlrun/mlrun"
    mandatory_env_vars = super().mandatory_env_vars

    def setup_class(cls):
        super().setup_class()
        cls.model_url = os.environ.get("OPENAI_BASE_URL")

    # @pytest.fixture(autouse=True)
    # def setup_before_each_test(self, use_datastore_profile):
    #     if use_datastore_profile:
    #         self.profile = DatastoreProfileOpenAI(
    #             name=self.profile_name,
    #             api_key=self.env_secrets.get("OPENAI_API_KEY"),
    #             organization=self.env_secrets.get("OPENAI_ORG_ID"),
    #             project=self.env_secrets.get("OPENAI_PROJECT_ID"),
    #             base_url=self.env_secrets.get("OPENAI_BASE_URL"),
    #             timeout=self.env_secrets.get("OPENAI_TIMEOUT"),
    #             max_retries=self.env_secrets.get("OPENAI_MAX_RETRIES"),
    #         )
    #         register_temporary_client_datastore_profile(self.profile)
    #         self.url_prefix = f"ds://{self.profile_name}/"
    #         self.reset_env()
    #     else:
    #         for key, env_param in self.env_secrets.items():
    #             if env_param:
    #                 os.environ[key] = env_param
    #         model_provider_manager.reset_secrets()
    #         self.url_prefix = "openai://"

    def test_basic_openai_model_runner(self):
        project = mlrun.new_project("remote-model-project", save=False)
        model_artifact = project.log_model(
            "my_model",
            model_url="http://localhost:8080/v2/models/mymodel/infer",
            default_config={"model_version": "4"},
        )
        function = mlrun.new_function("tests", kind="serving")
        graph = function.set_topology("flow", engine="async")
        model_runner_step = ModelRunnerStep(name="my_model_runner")
        model_runner_step.add_model(
            model_class="MyOpenAILLM",
            endpoint_name="my_endpoint",
            model_artifact=model_artifact,
        )
        graph.to(model_runner_step).respond()
