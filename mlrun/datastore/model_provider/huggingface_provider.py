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
import threading
from typing import TYPE_CHECKING, Any, Optional, Union

import mlrun
from mlrun.datastore.model_provider.model_provider import (
    InvokeResponseFormat,
    ModelProvider,
    UsageResponseKeys,
)

if TYPE_CHECKING:
    from transformers.pipelines.base import Pipeline
    from transformers.pipelines.text_generation import ChatType


class HuggingFaceProvider(ModelProvider):
    """
    HuggingFaceProvider is a wrapper around the Hugging Face Transformers pipeline
    that provides an interface for interacting with a wide range of Hugging Face models.

    It supports synchronous operations, enabling flexible integration into various workflows.

    This class extends the ModelProvider base class and implements Hugging Face-specific
    functionality, including pipeline initialization, default text generation operations,
    and custom operations tailored to the Hugging Face Transformers pipeline API.

    Note: The pipeline object will download the model (if not already cached) and load it
    into memory for inference. Ensure you have the required CPU/GPU and memory to use this operation.
    """

    #  locks for threading use cases
    _client_lock = threading.Lock()

    def __init__(
        self,
        parent,
        schema,
        name,
        endpoint="",
        secrets: Optional[dict] = None,
        default_invoke_kwargs: Optional[dict] = None,
    ):
        endpoint = endpoint or mlrun.mlconf.model_providers.huggingface_default_model
        if schema != "huggingface":
            raise mlrun.errors.MLRunInvalidArgumentError(
                "HuggingFaceProvider supports only 'huggingface' as the provider kind."
            )
        super().__init__(
            parent=parent,
            kind=schema,
            name=name,
            endpoint=endpoint,
            secrets=secrets,
            default_invoke_kwargs=default_invoke_kwargs,
        )
        self.options = self.get_client_options()
        self._expected_operation_type = None
        self._download_model()

    @staticmethod
    def _extract_string_output(response: list[dict]) -> str:
        """
        Extracts the first generated string from Hugging Face pipeline output
        """
        if not isinstance(response, list) or len(response) == 0:
            raise ValueError("Empty or invalid pipeline output")
        if len(response) != 1:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "HuggingFaceProvider: extracting string from response is only supported for single-response outputs"
            )
        return response[0].get("generated_text")

    @classmethod
    def parse_endpoint_and_path(cls, endpoint, subpath) -> (str, str):
        if endpoint and subpath:
            endpoint = endpoint + subpath
            # In HuggingFace, "/" in a model name is part of the name — `subpath` is not used.
            subpath = ""
        return endpoint, subpath

    @property
    def client(self) -> Any:
        """
        Lazily return the HuggingFace-pipeline client.

        If the client has not been initialized yet, it will be created
        by calling `load_client`.
        """
        self.load_client()
        return self._client

    def _download_model(self):
        """
        Pre-downloads model files locally to prevent race conditions in multiprocessing.

        Uses snapshot_download with local_dir_use_symlinks=False to ensure proper
        file copying for safe concurrent access across multiple processes.

        :raises:
            ImportError: If huggingface_hub package is not installed.
        """
        try:
            from huggingface_hub import snapshot_download

            # Download the model and tokenizer files directly to the cache.
            snapshot_download(
                repo_id=self.model,
                local_dir_use_symlinks=False,
                token=self._get_secret_or_env("HF_TOKEN") or None,
            )
        except ImportError as exc:
            raise ImportError("huggingface_hub package is not installed") from exc

    def _response_handler(
        self,
        response: Union[str, list],
        invoke_response_format: InvokeResponseFormat = InvokeResponseFormat.FULL,
        messages: Union[str, list[str], "ChatType", list["ChatType"]] = None,
        **kwargs,
    ) -> Union[str, list, dict[str, Any]]:
        """
        Processes and formats the raw response from the HuggingFace pipeline according to the specified format.

        The response should exclude the user’s input (no repetition in the output).
        This can be accomplished by invoking the pipeline with `return_full_text=False`.

        :param response:                The raw response from the HuggingFace pipeline, typically a list of dictionaries
                                        containing generated text sequences.
        :param invoke_response_format:  Determines how the response should be processed and returned. Options:

                                       - STRING: Return only the main generated content as a string,
                                                 for single-answer responses.
                                       - USAGE: Return a dictionary combining the string response with
                                                token usage statistics:

                                       .. code-block:: json

                                       {
                                           "answer": "<generated_text>",
                                           "usage": {
                                               "prompt_tokens": <int>,
                                               "completion_tokens": <int>,
                                               "total_tokens": <int>
                                           }
                                       }

                                       Note: Token counts are estimated after answer generation and
                                       may differ from the actual tokens generated by the model due to
                                       internal decoding behavior and implementation details.

                                       - FULL: Return the full raw response object.

        :param messages:               The original input messages used for token count estimation in USAGE mode.
                                       Can be a string, list of strings, or chat format messages.
        :param kwargs:                 Additional parameters for response processing.

        :return:                       The processed response in the format specified by `invoke_response_format`.
                                       Can be a string, dictionary, or the original response object.

        :raises MLRunInvalidArgumentError: If extracting the string response fails.
        :raises MLRunRuntimeError: If applying the chat template to the model fails during token usage calculation.
        """
        if InvokeResponseFormat.is_str_response(invoke_response_format.value):
            str_response = self._extract_string_output(response)
            if invoke_response_format == InvokeResponseFormat.STRING:
                return str_response
            if invoke_response_format == InvokeResponseFormat.USAGE:
                tokenizer = self.client.tokenizer
                if not isinstance(messages, str):
                    try:
                        messages = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    except Exception as e:
                        raise mlrun.errors.MLRunRuntimeError(
                            f"Failed to apply chat template using the tokenizer for model '{self.model}'. "
                            "This may indicate that the tokenizer does not support chat formatting, "
                            "or that the input format is invalid. "
                            f"Original error: {e}"
                        )
                prompt_tokens = len(tokenizer.encode(messages))
                completion_tokens = len(tokenizer.encode(str_response))
                total_tokens = prompt_tokens + completion_tokens
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
                response = {
                    UsageResponseKeys.ANSWER: str_response,
                    UsageResponseKeys.USAGE: usage,
                }
        return response

    def load_client(self) -> None:
        """
        Initializes the Hugging Face pipeline using the provided options.

        This method imports the `pipeline` function from the `transformers` package,
        creates a pipeline instance with the specified task and model (from `self.options`),
        and assigns it to `self._client`.

        Note: Hugging Face pipelines are synchronous and do not support async invocation.

        :raises:
            ImportError: If the `transformers` package is not installed.
        """
        if self._client:
            return
        try:
            from transformers import pipeline, AutoModelForCausalLM  # noqa
            from transformers import AutoTokenizer  # noqa
            from transformers.pipelines.base import Pipeline  # noqa

            self.options["model_kwargs"] = self.options.get("model_kwargs", {})
            self.options["model_kwargs"]["local_files_only"] = True
            with self._client_lock:
                self._client = pipeline(model=self.model, **self.options)
            self._expected_operation_type = Pipeline
        except ImportError as exc:
            raise ImportError("transformers package is not installed") from exc

    def get_client_options(self):
        res = dict(
            task=self._get_secret_or_env("HF_TASK") or "text-generation",
            token=self._get_secret_or_env("HF_TOKEN"),
            device=self._get_secret_or_env("HF_DEVICE"),
            device_map=self._get_secret_or_env("HF_DEVICE_MAP"),
            trust_remote_code=self._get_secret_or_env("HF_TRUST_REMOTE_CODE"),
            model_kwargs=self._get_secret_or_env("HF_MODEL_KWARGS"),
        )
        return self._sanitize_options(res)

    def custom_invoke(
        self, operation: Optional["Pipeline"] = None, **invoke_kwargs
    ) -> Union[list, dict, Any]:
        """
        Invokes a HuggingFace pipeline operation with the given keyword arguments.

        This method provides flexibility to use a custom pipeline object for specific tasks
        (e.g., image classification, sentiment analysis).

        The operation must be a Pipeline object from the transformers library that accepts keyword arguments.

        Example:
            ```python
            from transformers import pipeline
            from PIL import Image

            # Using custom pipeline for image classification
            image = Image.open(image_path)
            pipeline_object = pipeline("image-classification", model="microsoft/resnet-50")
            result = hf_provider.custom_invoke(
                pipeline_object,
                inputs=image,
            )
            ```

        :param operation:      A Pipeline object from the transformers library.
                               If not provided, defaults to the provider's configured pipeline.
        :param invoke_kwargs:  Keyword arguments to pass to the pipeline operation.
                               These are merged with `default_invoke_kwargs` and may include
                               parameters such as `inputs`, `max_length`, `temperature`, or task-specific options.

        :return:               The full response returned by the pipeline operation.
                               Format depends on the pipeline task (list for text generation,
                               dict for classification, etc.).

        :raises MLRunInvalidArgumentError: If the operation is not a valid Pipeline object.

        """
        invoke_kwargs = self.get_invoke_kwargs(invoke_kwargs)
        if operation:
            if not isinstance(operation, self._expected_operation_type):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Huggingface operation must inherit" " from 'Pipeline' object"
                )
            return operation(**invoke_kwargs)
        else:
            return self.client(**invoke_kwargs)

    def invoke(
        self,
        messages: Union[str, list[str], "ChatType", list["ChatType"]],
        invoke_response_format: InvokeResponseFormat = InvokeResponseFormat.FULL,
        **invoke_kwargs,
    ) -> Union[str, list, dict[str, Any]]:
        """
        HuggingFace-specific implementation of model invocation using the synchronous pipeline client.
        Invokes a HuggingFace model operation for text generation tasks.

        Note: Ensure your environment has sufficient computational resources (CPU/GPU and memory) to run the model.

        :param messages:
            Input for the text generation model. Can be provided in multiple formats:

            - A single string: Direct text input for generation
            - A list of strings: Multiple text inputs for batch processing
            - Chat format: A list of dictionaries with "role" and "content" keys:

            .. code-block:: json

                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"}
                ]

        :param invoke_response_format: InvokeResponseFormat
            Specifies the format of the returned response. Options:

            - "string": Returns only the generated text content, extracted from a single response.
            - "usage":  Combines the generated text with metadata (e.g., token usage), returning a dictionary:

            .. code-block:: json
                {
                    "answer": "<generated_text>",
                    "usage": {
                        "prompt_tokens": <int>,
                        "completion_tokens": <int>,
                        "total_tokens": <int>
                    }
                }

            Note: For usage mode, the model tokenizer should support apply_chat_template.

            - "full":   Returns the raw response object from the HuggingFace model,
                        typically a list of generated sequences (dictionaries).
                        This format does not include token usage statistics.

        :param invoke_kwargs:
            Additional keyword arguments passed to the HuggingFace pipeline.

        :return:
            A string, dictionary, or list of model outputs, depending on `invoke_response_format`.

        :raises MLRunInvalidArgumentError:
            If the pipeline task is not "text-generation" or if the response contains multiple outputs when extracting
            string content.
        :raises MLRunRuntimeError:
            If using "usage" response mode and the model tokenizer does not support chat template formatting.
        """
        if self.client.task != "text-generation":
            raise mlrun.errors.MLRunInvalidArgumentError(
                "HuggingFaceProvider.invoke supports text-generation task only"
            )
        if InvokeResponseFormat.is_str_response(invoke_response_format.value):
            invoke_kwargs["return_full_text"] = False
        response = self.custom_invoke(text_inputs=messages, **invoke_kwargs)
        response = self._response_handler(
            messages=messages,
            response=response,
            invoke_response_format=invoke_response_format,
        )
        return response
