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

import fsspec

import mlrun
import mlrun.artifacts
import mlrun.errors
import mlrun.serving
from mlrun.datastore.model_provider.model_provider import (
    InvokeResponseFormat,
    ModelProvider,
)
from mlrun.serving import ModelRunnerStep
from mlrun.serving.states import LLModel  # noqa


class LLMContentMismatchError(AssertionError):
    """Raised when LLM generates unexpected content (retriable error)."""

    pass


def retry_on_content_mismatch(func, max_attempts=3, *args, **kwargs):
    """
    Execute func with retry logic for LLMContentMismatchError.
    Other exceptions fail immediately.
    """
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except LLMContentMismatchError as e:
            if attempt == max_attempts - 1:
                raise
            print(f"LLM content mismatch (attempt {attempt + 1}/{max_attempts}): {e}")
    return None


def validate_llm_single_response(
    response,
    expected_result,
    encoding_or_tokenizer,
    min_tokens=95,
    max_tokens=105,
):
    from mlrun.datastore.model_provider.model_provider import UsageResponseKeys

    assert isinstance(response, dict), f"Expected dict response, got {type(response)}"
    assert len(response) == 2, f"Expected 2 keys in response, got {len(response)}"
    assert UsageResponseKeys.ANSWER in response
    assert UsageResponseKeys.USAGE in response

    answer = response[UsageResponseKeys.ANSWER]
    if expected_result not in answer.lower():
        raise LLMContentMismatchError(
            f"Expected '{expected_result}' in answer, got: {answer[:100]}..."
        )

    token_count = len(encoding_or_tokenizer.encode(answer))
    assert (
        min_tokens <= token_count <= max_tokens
    ), f"Token count {token_count} not in range [{min_tokens}, {max_tokens}]"

    stats = response[UsageResponseKeys.USAGE]
    assert isinstance(stats, dict)
    assert (
        min_tokens <= stats["completion_tokens"] <= max_tokens
    ), f"Completion tokens {stats['completion_tokens']} not in range [{min_tokens}, {max_tokens}]"
    assert stats["prompt_tokens"] > 0
    assert stats["total_tokens"] == stats["completion_tokens"] + stats["prompt_tokens"]


def validate_llm_batch_response_system(
    batch_response,
    expected_results,
    encoding_or_tokenizer,
    min_tokens=95,
    max_tokens=105,
):
    assert isinstance(
        batch_response, list
    ), f"Expected list response, got {type(batch_response)}"
    assert len(batch_response) == len(
        expected_results
    ), f"Expected {len(expected_results)} responses, got {len(batch_response)}"

    for i, full_result in enumerate(batch_response):
        result = full_result["output"]
        validate_llm_single_response(
            result,
            expected_results[i],
            encoding_or_tokenizer,
            min_tokens,
            max_tokens,
        )


def get_openai_encoding(model_name):
    """Get tiktoken encoding for OpenAI model."""
    import tiktoken

    return tiktoken.encoding_for_model(model_name)


def validate_openai_single_response(
    response, expected_result, model_name, min_tokens=95, max_tokens=105
):
    """OpenAI-specific single response validation."""
    encoding = get_openai_encoding(model_name)
    validate_llm_single_response(
        response, expected_result, encoding, min_tokens, max_tokens
    )


def validate_openai_batch_response(
    batch_response, expected_results, model_name, min_tokens=95, max_tokens=105
):
    """OpenAI-specific batch response validation."""
    encoding = get_openai_encoding(model_name)
    validate_llm_batch_response_system(
        batch_response, expected_results, encoding, min_tokens, max_tokens
    )


PROMPT_LEGEND = {
    "question": {"field": None, "description": None},
    "depth_level": {"field": None, "description": None},
    "persona": {"field": None, "description": None},
    "tone": {"field": None, "description": None},
}
BATCH_INPUT_DATA = [
    {
        "question": "What is the capital of France? Answer with one word first, then provide a historical overview."
        " Answer in detail with at least 200 words.",
        "depth_level": "detailed",
        "persona": "teacher",
        "tone": "casual",
    },
    {
        "question": "What is the largest planet in our solar system? First give a one-word answer, "
        "then provide a detailed explanation in at least 200 words.",
        "depth_level": "basic",
        "persona": "astronomy teacher",
        "tone": "simple",
    },
    {
        "question": "Who wrote Hamlet? Answer shortly and then explain with details.  "
        "Answer in detail with at least 200 words.",
        "depth_level": "basic",
        "persona": "literature professor",
        "tone": "formal",
    },
    {
        "question": "What color is the sky on a clear day? Answer shortly and then "
        "Answer in detail with at least 200 words.",
        "depth_level": "basic",
        "persona": "child",
        "tone": "fun",
    },
    {
        "question": "What planet do we live on? Answer shortly and then explain with details. "
        "Answer in detail with at least 200 words.",
        "depth_level": "basic",
        "persona": "astronaut",
        "tone": "educational",
    },
]

EXPECTED_RESULTS = ["paris", "jupiter", "shakespeare", "blue", "earth"]

PROMPT_TEMPLATE = [
    {
        "role": "user",
        "content": "{question}. Explain {depth_level} as a {persona} in {tone} style.",
    }
]

formatted_messages = [
    {"role": prompt["role"], "content": prompt["content"].format(**input_data)}
    for input_data in BATCH_INPUT_DATA
    for prompt in PROMPT_TEMPLATE
]

FLUSH_AFTER_SECONDS = 4


def create_mocked_get_store_artifact(uri_to_artifact: dict):
    def mocked_get_store_artifact(uri, **kwargs):
        artifact = uri_to_artifact.get(uri)
        if not artifact:
            raise mlrun.errors.MLRunInvalidArgumentError("Artifact uri not found")
        return artifact, None

    return mocked_get_store_artifact


def setup_remote_model_test(
    project,
    model_url,
    mlrun_model_name="mymodel",
    execution_mechanism="naive",
    image=None,
    requirements=None,
    model_class: str = "LLModel",
    default_config: Optional[dict] = None,
    include_llm_artifact=True,
    batch_step=False,
    flush_after_seconds=FLUSH_AFTER_SECONDS,
):
    model_artifact = project.log_model(
        mlrun_model_name,
        model_url=model_url,
        default_config=default_config,
    )
    if include_llm_artifact:
        llm_prompt_artifact = project.log_llm_prompt(
            "my_llm_prompt",
            prompt_template=PROMPT_TEMPLATE,
            model_artifact=model_artifact,
            prompt_legend=PROMPT_LEGEND,
        )
    else:
        llm_prompt_artifact = None

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
    if batch_step:
        # When deploying with batch_step in system tests, configure async HTTP via
        # function.with_http(async_spec=AsyncSpec()) outside this utility function
        graph = graph.to(
            "storey.Batch",
            "my_batching",
            max_events=2,
            flush_after_seconds=flush_after_seconds,
            full_event=True,
        )
    model_runner_step = ModelRunnerStep(name="my_model_runner")
    model_runner_step.add_model(
        model_class=model_class,
        endpoint_name="my_endpoint",
        execution_mechanism=execution_mechanism,
        model_artifact=llm_prompt_artifact or model_artifact,
        result_path="output",
    )
    step = graph.to(model_runner_step)
    if batch_step:
        step = step.to("storey.FlatMap", _fn="(event.body)", full_event=True)
    step.respond()

    return model_artifact, llm_prompt_artifact, function


async def timed(coro):
    start = time.perf_counter()
    result = await coro
    duration = time.perf_counter() - start
    return result, duration


class MyHuggingFaceCustom(mlrun.serving.states.Model):
    """Custom MLRun model wrapper for Hugging Face image classification that loads an image
    from a given path and returns predictions via HuggingFaceProvider."""

    def predict(self, body: Any, **kwargs) -> Any:
        if isinstance(self.model_provider, ModelProvider):
            # Imported here to avoid requiring Pillow in environments where it's not needed
            from PIL import Image

            with fsspec.open(body["input"], "rb") as f:
                image = Image.open(f)
                image.load()  # ensure image is fully read into memory

            result = self.model_provider.custom_invoke(
                inputs=image,
            )
            body["result"] = result
            return body


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
        invocation_config = {}
        all_messages = []
        for event in body["input"]:
            messages, invocation_config = self.enrich_prompt(
                event, origin_name, llm_prompt_artifact=self.invocation_artifact
            )
            all_messages.extend(messages)
        return await self.predict_async(
            body, messages=all_messages, invocation_config=invocation_config
        )

    async def predict_async(
        self,
        body: Any,
        messages: Optional[list[dict]] = None,
        invocation_config: Optional[dict] = None,
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
                        **(invocation_config or {}),
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
