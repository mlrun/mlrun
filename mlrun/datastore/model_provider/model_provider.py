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
from collections.abc import Awaitable
from typing import Any, Callable, Optional, Union

import mlrun.errors
from mlrun.common.types import StrEnum
from mlrun.datastore.remote_client import (
    BaseRemoteClient,
)


class InvokeResponseFormat(StrEnum):
    STRING = "string"
    USAGE = "usage"
    FULL = "full"

    @classmethod
    def is_str_response(cls, invoke_response_format: str) -> bool:
        """
        Returns True if the response key corresponds to a string-based response (not a full generation object).
        """
        return invoke_response_format in {
            cls.USAGE,
            cls.STRING,
        }


class UsageResponseKeys(StrEnum):
    ANSWER = "answer"
    USAGE = "usage"

    @classmethod
    def fields(cls) -> list[str]:
        return [cls.ANSWER, cls.USAGE]


class ModelProvider(BaseRemoteClient):
    """
    The ModelProvider class is an abstract base for integrating with external
    model providers, primarily generative AI (GenAI) services.

    Designed to be subclassed, it defines a consistent interface and shared
    functionality for tasks such as text generation, embeddings, and invoking
    fine-tuned models. Subclasses should implement provider-specific logic,
    including SDK client initialization, model invocation, and custom operations.

    Key Features:
    - Establishes a consistent, reusable client management for model provider integrations.
    - Simplifies GenAI service integration by abstracting common operations.
    - Reduces duplication through shared components for common tasks.
    - Holds default invocation parameters (e.g., temperature, max_tokens) to avoid boilerplate
    code and promote consistency.
    """

    support_async = False

    def __init__(
        self,
        parent,
        kind,
        name,
        endpoint="",
        secrets: Optional[dict] = None,
        default_invoke_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            parent=parent, name=name, kind=kind, endpoint=endpoint, secrets=secrets
        )
        self.default_invoke_kwargs = default_invoke_kwargs or {}
        self._client = None
        self._async_client = None

    @staticmethod
    def _extract_string_output(response: Any) -> str:
        """
        Extracts string response from response object
        """
        pass

    def _response_handler(
        self,
        response: Any,
        invoke_response_format: InvokeResponseFormat = InvokeResponseFormat.FULL,
        **kwargs,
    ) -> Union[str, dict, Any]:
        """
        Handles the model response according to the specified response format.

        :param response: The raw response returned from the model invocation.
        :param invoke_response_format: Determines how the response should be processed and returned.
                                       Options include:

                                       - STRING: Return only the main generated content as a string,
                                                 typically for single-answer responses.
                                       - USAGE: Return a dictionary combining the string response with
                                                additional metadata or token usage statistics, in this format:
                                                {"answer": <string>, "usage": <dict>}

                                       - FULL: Return the full raw response object.

        :param kwargs:                  Additional parameters that may be required by specific implementations.

        :return:                        The processed response in the format specified by `invoke_response_format`.
                                        Can be a string, dictionary, or the original response object.
        """
        return None

    def get_client_options(self) -> dict:
        """
        Returns a dictionary containing credentials and configuration
        options required for client creation.

        :return:           A dictionary with client-specific settings.
        """
        return {}

    def load_client(self) -> None:
        """
        Initialize the SDK client for the model provider and assign it to an instance attribute.

        Subclasses should override this method to create and configure the provider-specific client.
        """

        raise NotImplementedError("load_client method is not implemented")

    def load_async_client(self) -> Any:
        raise NotImplementedError("load_async_client method is not implemented")

    @property
    def client(self) -> Any:
        return self._client

    @property
    def model(self) -> Optional[str]:
        """
        Returns the model identifier used by the underlying SDK.

        :return: A string representing the model ID, or None if not set.
        """
        return self.endpoint

    def get_invoke_kwargs(self, invoke_kwargs) -> dict:
        kwargs = self.default_invoke_kwargs.copy()
        kwargs.update(invoke_kwargs)
        return kwargs

    @property
    def async_client(self) -> Any:
        if not self.support_async:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"{self.__class__.__name__} does not support async operations"
            )
        return self._async_client

    def custom_invoke(
        self, operation: Optional[Callable] = None, **invoke_kwargs
    ) -> Any:
        """
        Invokes a model operation from a provider (e.g., OpenAI, Hugging Face, etc.) with the given keyword arguments.

        Useful for dynamically calling model methods like text generation, chat completions, or image generation.
        The operation must be a callable that accepts keyword arguments.

        :param operation:       A callable representing the model operation (e.g., a client method).
        :param invoke_kwargs:   Keyword arguments to pass to the operation.
        :return:                The full response returned by the operation.
        """
        raise NotImplementedError("custom_invoke method is not implemented")

    async def async_custom_invoke(
        self, operation: Optional[Callable[..., Awaitable[Any]]] = None, **invoke_kwargs
    ) -> Any:
        """
        Asynchronously invokes a model operation from a provider (e.g., OpenAI, Hugging Face, etc.)
        with the given keyword arguments.

        The operation must be an async callable (e.g., a method from an async client) that accepts keyword arguments.

        :param operation:       An async callable representing the model operation (e.g., an async_client method).
        :param invoke_kwargs:   Keyword arguments to pass to the operation.
        :return:                The full response returned by the awaited operation.
        """
        raise NotImplementedError("async_custom_invoke is not implemented")

    def invoke(
        self,
        messages: Union[list[dict], Any],
        invoke_response_format: InvokeResponseFormat = InvokeResponseFormat.FULL,
        **invoke_kwargs,
    ) -> Union[str, dict[str, Any], Any]:
        """
        Invokes a generative AI model with the provided messages and additional parameters.
        This method is designed to be a flexible interface for interacting with various
        generative AI backends (e.g., OpenAI, Hugging Face, etc.). It allows users to send
        a list of messages (following a standardized format) and receive a response.

        :param messages:            A list of dictionaries representing the conversation history or input messages.
                                    Each dictionary should follow the format::
                                    {"role": "system"| "user" | "assistant" ..., "content":
                                    "Message content as a string"}

                                    Example:

                                    .. code-block:: json

                                        [
                                            {"role": "system", "content": "You are a helpful assistant."},
                                            {"role": "user", "content": "What is the capital of France?"}
                                        ]

                                    This format is consistent across all backends. Defaults to None if no messages
                                    are provided.

        :param invoke_response_format:   Determines how the model response is returned:

                                    - string:   Returns only the generated text content from the model output,
                                                for single-answer responses only.

                                    - usage:    Combines the STRING response with additional metadata (token usage),
                                                and returns the result in a dictionary.

                                                Note: The usage dictionary may contain additional
                                                keys depending on the model provider:

                                    .. code-block:: json

                                    {
                                        "answer": "<generated_text>",
                                        "usage": {
                                        "prompt_tokens": <int>,
                                        "completion_tokens": <int>,
                                        "total_tokens": <int>
                                        }

                                    }

                                    - full:   Returns the full model output.

        :param invoke_kwargs:
                                    Additional keyword arguments to be passed to the underlying model API call.
                                    These can include parameters such as temperature, max tokens, etc.,
                                    depending on the capabilities of the specific backend being used.

        :return:                    The invoke result formatted according to the specified
                                    invoke_response_format parameter.

        """
        raise NotImplementedError("invoke method is not implemented")

    async def async_invoke(
        self,
        messages: list[dict],
        invoke_response_format=InvokeResponseFormat.FULL,
        **invoke_kwargs,
    ) -> Union[str, dict[str, Any], Any]:
        """
        Asynchronously invokes a generative AI model with the provided messages and additional parameters.
        This method is designed to be a flexible interface for interacting with various
        generative AI backends (e.g., OpenAI, Hugging Face, etc.). It allows users to send
        a list of messages (following a standardized format) and receive a response.

        :param messages:            A list of dictionaries representing the conversation history or input messages.
                                    Each dictionary should follow the format::
                                    {"role": "system"| "user" | "assistant" ..., "content":
                                    "Message content as a string"}

                                    Example:

                                    .. code-block:: json

                                        [
                                            {"role": "system", "content": "You are a helpful assistant."},
                                            {"role": "user", "content": "What is the capital of France?"}
                                        ]

                                    This format is consistent across all backends. Defaults to None if no messages
                                    are provided.

        :param invoke_response_format:   Determines how the model response is returned:

                                    - string:   Returns only the generated text content from the model output,
                                                for single-answer responses only.

                                    - usage:    Combines the STRING response with additional metadata (token usage),
                                                and returns the result in a dictionary.

                                                Note: The usage dictionary may contain additional
                                                keys depending on the model provider:

                                    .. code-block:: json

                                    {
                                        "answer": "<generated_text>",
                                        "usage": {
                                        "prompt_tokens": <int>,
                                        "completion_tokens": <int>,
                                        "total_tokens": <int>
                                        }

                                    }

                                    - full:   Returns the full model output.

        :param invoke_kwargs:
                                    Additional keyword arguments to be passed to the underlying model API call.
                                    These can include parameters such as temperature, max tokens, etc.,
                                    depending on the capabilities of the specific backend being used.

        :return:                    The invoke result formatted according to the specified
                                    invoke_response_format parameter.

        """
        raise NotImplementedError("async_invoke is not implemented")
