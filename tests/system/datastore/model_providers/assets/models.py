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

import mlrun
import mlrun.artifacts
import mlrun.serving.states
from mlrun.datastore.model_provider.model_provider import ModelProvider


class MyOpenAILLM(mlrun.serving.states.Model):
    def predict(self, body):
        if isinstance(
            self.invocation_artifact, mlrun.artifacts.LLMPromptArtifact
        ) and isinstance(self.model_provider, ModelProvider):
            prompt = self.enrich_prompt(body)
            body["result"] = self.model_provider.invoke(
                prompt=prompt,
                **(self.invocation_artifact.spec.model_configuration or {}),
            )
        return body

    def enrich_prompt(self, body) -> str:
        # TODO: Update this once ML-8172 is completed
        if isinstance(self.invocation_artifact, mlrun.artifacts.LLMPromptArtifact):
            prompt_template = self.invocation_artifact.spec.prompt_string
            needed_params = ["question", "depth_level", "persona", "tone"]
            sub_dict = {k: body[k] for k in needed_params if k in body}
            return prompt_template.format(**sub_dict)
        return body["prompt"]
