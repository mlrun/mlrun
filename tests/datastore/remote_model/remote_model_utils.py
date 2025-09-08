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
from typing import Any, Optional

import mlrun
import mlrun.artifacts
import mlrun.serving
from mlrun.datastore.model_provider.model_provider import (
    InvokeResponseFormat,
    ModelProvider,
)
from mlrun.serving import ModelRunnerStep
from mlrun.serving.states import LLModel  # noqa

INPUT_DATA = [
    {
        "question": "What is the capital of France? Answer with one word first, then provide a historical overview.",
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
]

EXPECTED_RESULTS = ["paris", "4", "shakespeare", "blue", "earth"]

PROMPT_TEMPLATE = [
    {
        "role": "user",
        "content": "{question}. Explain {depth_level} as a {persona} in {tone} style.",
    }
]

formatted_messages = [
    {"role": prompt["role"], "content": prompt["content"].format(**input_data)}
    for input_data in INPUT_DATA
    for prompt in PROMPT_TEMPLATE
]


def setup_remote_model_test(
    project,
    model_url,
    mlrun_model_name="mymodel",
    execution_mechanism="naive",
    image=None,
    requirements=None,
    requirements_file=None,
    model_class: str = "LLModel",
    default_config: Optional[dict] = None,
):
    model_artifact = project.log_model(
        mlrun_model_name,
        model_url=model_url,
        default_config=default_config,
    )
    llm_prompt_artifact = project.log_llm_prompt(
        "my_llm_prompt",
        prompt_template=PROMPT_TEMPLATE,
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
        project=project.name,
        filename=__file__,
        image=image,
        requirements=requirements,
        requirements_file=requirements_file,
    )
    graph = function.set_topology("flow", engine="async")
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    model_runner_step.add_model(
        model_class=model_class,
        endpoint_name="my_endpoint",
        execution_mechanism=execution_mechanism,
        model_artifact=llm_prompt_artifact,
        result_path="output",
    )
    graph.to(model_runner_step).respond()
    return model_artifact, llm_prompt_artifact, function


async def timed(coro):
    start = time.perf_counter()
    result = await coro
    duration = time.perf_counter() - start
    return result, duration


class MyOpenAICustom(mlrun.serving.states.Model):
    def predict(self, body: Any, **kwargs) -> Any:
        if isinstance(
            self.invocation_artifact, mlrun.artifacts.LLMPromptArtifact
        ) and isinstance(self.model_provider, ModelProvider):
            result = self.model_provider.custom_invoke(
                operation=self.model_provider.client.embeddings.create,
                input=body["input"],
            )
            body["result"] = result.to_dict()
        return body

    async def predict_async(self, body: Any, **kwargs) -> Any:
        if isinstance(
            self.invocation_artifact, mlrun.artifacts.LLMPromptArtifact
        ) and isinstance(self.model_provider, ModelProvider):
            result = await self.model_provider.async_custom_invoke(
                operation=self.model_provider.async_client.embeddings.create,
                input=body["input"],
            )
            body["result"] = result.to_dict()
        return body


class MyOpenAIAsyncEvents(mlrun.serving.states.LLModel):
    async def run_async(
        self, body: Any, path: str, origin_name: Optional[str] = None
    ) -> Any:
        # Temporary workaround for testing purposes only, until events execution will be able to run in parallel
        model_configuration = {}
        all_messages = []
        for event in body["input"]:
            messages, model_configuration = self.enrich_prompt(
                event, origin_name, llm_prompt_artifact=self.invocation_artifact
            )
            all_messages.extend(messages)
        return await self.predict_async(
            body, messages=all_messages, model_configuration=model_configuration
        )

    async def predict_async(
        self,
        body: Any,
        messages: Optional[list[dict]] = None,
        model_configuration: Optional[dict] = None,
        **kwargs,
    ):
        if isinstance(
            self.invocation_artifact, mlrun.artifacts.LLMPromptArtifact
        ) and isinstance(self.model_provider, ModelProvider):
            # Load the client before using async operations
            self.model_provider.load_async_client()
            coros = [
                timed(
                    self.model_provider.async_invoke(
                        messages=[message],
                        invoke_response_format=InvokeResponseFormat.STRING,
                        **(model_configuration or {}),
                    )
                )
                for message in messages
            ]
            results_with_times = await asyncio.gather(*coros)
            results = [r for r, _ in results_with_times]
            invoke_times = [t for _, t in results_with_times]
            body["results"] = results
            body["invoke_times"] = invoke_times
        return body


def assert_async_invocations(results_with_times, model_name, total_duration):
    # Imported inside the function to avoid ImportError in pod while using MyOpenAIAsyncEvents class.
    import tiktoken  # noqa

    results = results_with_times["results"]
    invoke_times = results_with_times["invoke_times"]
    encoding = tiktoken.encoding_for_model(model_name)
    for i in range(len(EXPECTED_RESULTS)):
        assert EXPECTED_RESULTS[i] in results[i].lower()
        number_of_tokens = len(encoding.encode(results[i]))
        assert (
            number_of_tokens == 100
        ), f"Expected 100 tokens for input #{i}, but got {number_of_tokens}"
    assert total_duration < sum(invoke_times)
