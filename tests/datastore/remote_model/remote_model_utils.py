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
from mlrun.serving import ModelRunnerStep

INPUT_DATA = {
    "input": [
        {
            "question": "What is the capital of France, and give a historical overview.",
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

PROMPT_TEMPLATE = "{question}. Explain {depth_level} as a {persona} in {tone} style."

fixed_prompts = [
    PROMPT_TEMPLATE.format(**input_data) for input_data in INPUT_DATA["input"]
]


def setup_remote_model_test(
    project,
    model_url,
    mlrun_model_name="mymodel",
    execution_mechanism="naive",
    image=None,
    requirements=None,
):
    model_artifact = project.log_model(
        mlrun_model_name,
        model_url=model_url,
        default_config={"max_tokens": 100},
    )
    llm_prompt_artifact = project.log_llm_prompt(
        "my_llm_prompt",
        prompt_string=PROMPT_TEMPLATE,
        model_artifact=model_artifact.uri,
    )
    # function = mlrun.new_function("tests", kind="serving")
    function = mlrun.code_to_function(
        name="tests",
        kind="serving",
        tag="latest",
        project=project.name,
        filename=__file__,
        image=image,
        requirements=requirements,
    )
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    model_runner_step.add_model(
        model_class="MyOpenAILLM",
        endpoint_name="my_endpoint",
        execution_mechanism=execution_mechanism,
        model_artifact=llm_prompt_artifact,
    )
    graph.to(model_runner_step).respond()
    return model_artifact, llm_prompt_artifact, function


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
                as_str=True,
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
                        as_str=True,
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


def assert_async_invocations(results_with_times, model_name, total_duration):
    # Imported inside the function to avoid ImportError in pod while using MyOpenAILLM class.
    import tiktoken

    results = results_with_times["results"]
    invoke_times = results_with_times["invoke_times"]
    encoding = tiktoken.encoding_for_model(model_name)
    for i in range(len(EXPECTED_RESULTS)):
        assert EXPECTED_RESULTS[i] in results[i].lower()
        assert len(encoding.encode(results[i])) == 100
    assert total_duration < sum(invoke_times)
