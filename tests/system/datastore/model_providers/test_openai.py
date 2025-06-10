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


import mlrun.artifacts
import mlrun.serving.states
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

    def test_basic_openai_model_runner(self):
        # TODO
        pass
