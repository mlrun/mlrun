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
import asyncio
import time

import mlrun
import mlrun.artifacts
import mlrun.serving
from mlrun.datastore.model_provider.model_provider import ModelProvider

INPUT_DATA = {
    "input": [
        {
            "question": "What is the capital of France, and give a brief historical overview.",
            "depth_level": "detailed",
            "persona": "teacher",
            "tone": "casual",
        },
        {
            "question": "What is 2 + 2? Answer shortly and then explain with details.",
            "depth_level": "basic",
            "persona": "math teacher",
            "tone": "simple",
        },
        {
            "question": "Who wrote Hamlet? Answer shortly and then explain with details.",
            "depth_level": "basic",
            "persona": "literature professor",
            "tone": "formal",
        },
        {
            "question": "What color is the sky on a clear day? Answer shortly and then explain with details.",
            "depth_level": "basic",
            "persona": "child",
            "tone": "fun",
        },
        {
            "question": "What planet do we live on? Answer shortly and then explain with details.",
            "depth_level": "basic",
            "persona": "astronaut",
            "tone": "educational",
        },
    ],
}

EXPECTED_RESULTS = ["paris", "4", "shakespeare", "blue", "earth"]


async def timed(coro):
    start = time.perf_counter()
    result = await coro
    duration = time.perf_counter() - start
    return result, duration


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

    async def predict_async(self, body):
        if isinstance(
            self.invocation_artifact, mlrun.artifacts.LLMPromptArtifact
        ) and isinstance(self.model_provider, ModelProvider):
            prompt_parameters: list = body["input"]
            prompts = [
                self.enrich_prompt(single_prompt_parameters)
                for single_prompt_parameters in prompt_parameters
            ]

            tasks = [
                timed(
                    self.model_provider.async_invoke(
                        prompt,
                        **(self.invocation_artifact.spec.model_configuration or {}),
                    )
                )
                for prompt in prompts
            ]
            results_with_times = await asyncio.gather(*tasks)
            results = [r for r, _ in results_with_times]
            invoke_times = [t for _, t in results_with_times]
            body["results"] = results
            body["invoke_times"] = invoke_times
        return body

    def enrich_prompt(self, body) -> str:
        # TODO: Update this once ML-8172 is completed
        if isinstance(self.invocation_artifact, mlrun.artifacts.LLMPromptArtifact):
            prompt_template = self.invocation_artifact.spec.prompt_string
            needed_params = ["question", "depth_level", "persona", "tone"]
            sub_dict = {k: body[k] for k in needed_params if k in body}
            return prompt_template.format(**sub_dict)
        return ""
