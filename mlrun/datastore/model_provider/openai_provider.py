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
import inspect
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import mlrun
from mlrun.datastore.model_provider.model_provider import (
    InvokeResponseFormat,
    ModelProvider,
    UsageResponseKeys,
)
from mlrun.datastore.utils import accepts_param

if TYPE_CHECKING:
    from openai._models import BaseModel  # noqa
    from openai.types.chat.chat_completion import ChatCompletion


class OpenAIProvider(ModelProvider):
    """
    OpenAIProvider is a wrapper around the OpenAI SDK that provides an interface
    for interacting with OpenAI's generative AI services.

    It supports both synchronous and asynchronous operations, allowing flexible
    integration into various workflows.

    This class extends the ModelProvider base class and implements OpenAI-specific
    functionality, including client initialization, model invocation, and custom
    operations tailored to the OpenAI API.
    """

    support_async = True
    response_class = None

    def __init__(
        self,
        parent,
        schema,
        name,
        endpoint="",
        secrets: Optional[dict] = None,
        default_invoke_kwargs: Optional[dict] = None,
    ):
        endpoint = endpoint or mlrun.mlconf.model_providers.openai_default_model
        if schema != "openai":
            raise mlrun.errors.MLRunInvalidArgumentError(
                "OpenAIProvider supports only 'openai' as the provider kind."
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

    @classmethod
    def _import_response_class(cls) -> None:
        if not cls.response_class:
            try:
                from openai.types.chat.chat_completion import ChatCompletion
            except ImportError as exc:
                raise ImportError("openai package is not installed") from exc
            cls.response_class = ChatCompletion

    @staticmethod
    def _extract_string_output(response: "ChatCompletion") -> str:
        """
        Extracts the text content of the first choice from an OpenAI ChatCompletion response.
        Only supports responses with a single choice. Raises an error if multiple choices exist.

        :param response: The ChatCompletion response from OpenAI.
        :return: The text content of the first message in the response.
        :raises MLRunInvalidArgumentError: If the response contains more than one choice.
        """
        if len(response.choices) != 1:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "OpenAIProvider: extracting string from response is only supported for single-response outputs"
            )
        return response.choices[0].message.content

    @classmethod
    def parse_endpoint_and_path(cls, endpoint, subpath) -> (str, str):
        if endpoint and subpath:
            endpoint = endpoint + subpath
            #  in openai there is no usage of subpath variable. if the model contains "/", it is part of the model name.
            subpath = ""
        return endpoint, subpath

    @property
    def client(self) -> Any:
        """
        Lazily return the synchronous OpenAI client.

        If the client has not been initialized yet, it will be created
        by calling `load_client`.
        """
        self.load_client()
        return self._client

    def load_client(self) -> None:
        """
        Lazily initialize the synchronous OpenAI client.

        The client is created only if it does not already exist.
        Raises ImportError if the openai package is not installed.
        """
        if self._client:
            return
        try:
            from openai import OpenAI  # noqa

            self._client = OpenAI(**self.options)
        except ImportError as exc:
            raise ImportError("openai package is not installed") from exc

    def load_async_client(self) -> None:
        """
        Lazily initialize the asynchronous OpenAI client.

        The client is created only if it does not already exist.
        Raises ImportError if the openai package is not installed.
        """
        if not self._async_client:
            try:
                from openai import AsyncOpenAI  # noqa

                self._async_client = AsyncOpenAI(**self.options)
            except ImportError as exc:
                raise ImportError("openai package is not installed") from exc

    @property
    def async_client(self) -> Any:
        """
        Return the asynchronous OpenAI client, creating it on first access.

        The client is lazily initialized via `load_async_client`.
        """
        self.load_async_client()
        return self._async_client

    def get_client_options(self) -> dict:
        res = dict(
            api_key=self._get_secret_or_env("OPENAI_API_KEY"),
            organization=self._get_secret_or_env("OPENAI_ORG_ID"),
            project=self._get_secret_or_env("OPENAI_PROJECT_ID"),
            base_url=self._get_secret_or_env("OPENAI_BASE_URL"),
            timeout=self._get_secret_or_env("OPENAI_TIMEOUT"),
            max_retries=self._get_secret_or_env("OPENAI_MAX_RETRIES"),
        )
        return self._sanitize_options(res)

    def custom_invoke(
        self, operation: Optional[Callable] = None, **invoke_kwargs
    ) -> Union["ChatCompletion", "BaseModel"]:
        """
        Invokes a model operation from the OpenAI client with the given keyword arguments.

        This method provides flexibility to either:
        - Call a specific OpenAI client operation (e.g., `client.images.generate`).
        - Default to `chat.completions.create` when no operation is provided.

        The operation must be a callable that accepts keyword arguments. If the callable
        does not accept a `model` parameter, it will be omitted from the call.

        Example:
            ```python
            result = openai_model_provider.custom_invoke(
                openai_model_provider.client.images.generate,
                prompt="A futuristic cityscape at sunset",
                n=1,
                size="1024x1024",
            )
            ```

        :param operation:       A callable representing the OpenAI operation to invoke.
                                If not provided, defaults to `client.chat.completions.create`.

        :param invoke_kwargs:   Additional keyword arguments to pass to the operation.
                                These are merged with `default_invoke_kwargs` and may
                                include parameters such as `temperature`, `max_tokens`,
                                or `messages`.

        :return:                The full response returned by the operation, typically
                                an OpenAI `ChatCompletion` or other OpenAI SDK model.
        """

        invoke_kwargs = self.get_invoke_kwargs(invoke_kwargs)
        model_kwargs = {"model": invoke_kwargs.pop("model", None) or self.model}

        if operation:
            if not callable(operation):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "OpenAI custom_invoke operation must be a callable"
                )
            if not accepts_param(operation, "model"):
                model_kwargs = {}
            return operation(**invoke_kwargs, **model_kwargs)
        else:
            return self.client.chat.completions.create(**invoke_kwargs, **model_kwargs)

    async def async_custom_invoke(
        self,
        operation: Optional[Callable[..., Awaitable[Any]]] = None,
        **invoke_kwargs,
    ) -> Union["ChatCompletion", "BaseModel"]:
        """
        Asynchronously invokes a model operation from the OpenAI client with the given keyword arguments.

        This method provides flexibility to either:
        - Call a specific async OpenAI client operation (e.g., `async_client.images.generate`).
        - Default to `chat.completions.create` when no operation is provided.

        The operation must be an async callable that accepts keyword arguments.
        If the callable does not accept a `model` parameter, it will be omitted from the call.

        Example:
            ```python
            result = await openai_model_provider.async_custom_invoke(
                openai_model_provider.async_client.images.generate,
                prompt="A futuristic cityscape at sunset",
                n=1,
                size="1024x1024",
            )
            ```

        :param operation:       An async callable representing the OpenAI operation to invoke.
                                If not provided, defaults to `async_client.chat.completions.create`.

        :param invoke_kwargs:   Additional keyword arguments to pass to the operation.
                                These are merged with `default_invoke_kwargs` and may
                                include parameters such as `temperature`, `max_tokens`,
                                or `messages`.

        :return:                The full response returned by the awaited operation,
                                typically an OpenAI `ChatCompletion` or other OpenAI SDK model.

        """
        invoke_kwargs = self.get_invoke_kwargs(invoke_kwargs)
        model_kwargs = {"model": invoke_kwargs.pop("model", None) or self.model}
        if operation:
            if not inspect.iscoroutinefunction(operation):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "OpenAI async_custom_invoke operation must be a coroutine function"
                )
            if not accepts_param(operation, "model"):
                model_kwargs = {}
            return await operation(**invoke_kwargs, **model_kwargs)
        else:
            return await self.async_client.chat.completions.create(
                **invoke_kwargs, **model_kwargs
            )

    def _response_handler(
        self,
        response: "ChatCompletion",
        invoke_response_format: InvokeResponseFormat = InvokeResponseFormat.FULL,
        **kwargs,
    ) -> ["ChatCompletion", str, dict[str, Any]]:
        if InvokeResponseFormat.is_str_response(invoke_response_format.value):
            str_response = self._extract_string_output(response)
            if invoke_response_format == InvokeResponseFormat.STRING:
                return str_response
            if invoke_response_format == InvokeResponseFormat.USAGE:
                usage = response.to_dict()["usage"]
                response = {
                    UsageResponseKeys.ANSWER: str_response,
                    UsageResponseKeys.USAGE: usage,
                }
        return response

    def invoke(
        self,
        messages: list[dict],
        invoke_response_format: InvokeResponseFormat = InvokeResponseFormat.FULL,
        **invoke_kwargs,
    ) -> Union[dict[str, Any], str, "ChatCompletion"]:
        """
        OpenAI-specific implementation of `ModelProvider.invoke`.
        Invokes an OpenAI model operation using the synchronous client.

        :param messages:
            A list of dictionaries representing the conversation history or input messages.
            Each dictionary should follow the format::
                {
                    "role": "system" | "user" | "assistant",
                    "content": "Message content as a string",
                }

            Example:

            .. code-block:: json

                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"}
                ]

            Defaults to None if no messages are provided.

        :param invoke_response_format:
            Specifies the format of the returned response. Options:

            - "string": Returns only the generated text content, taken from a single response.
            - "usage": Combines the generated text with metadata (e.g., token usage), returning a dictionary::

                .. code-block:: json
                   {
                       "answer": "<generated_text>",
                       "usage": <ChatCompletion>.to_dict()["usage"]
                   }

            - "full": Returns the full OpenAI `ChatCompletion` object.

        :param invoke_kwargs:
            Additional keyword arguments passed to the OpenAI client.

        :return:
            A string, dictionary, or `ChatCompletion` object, depending on `invoke_response_format`.
        """

        response = self.custom_invoke(messages=messages, **invoke_kwargs)
        return self._response_handler(
            messages=messages,
            invoke_response_format=invoke_response_format,
            response=response,
        )

    async def async_invoke(
        self,
        messages: list[dict],
        invoke_response_format=InvokeResponseFormat.FULL,
        **invoke_kwargs,
    ) -> Union[str, "ChatCompletion", dict]:
        """
        OpenAI-specific implementation of `ModelProvider.async_invoke`.
        Invokes an OpenAI model operation using the asynchronous client.

        :param messages:
            A list of dictionaries representing the conversation history or input messages.
            Each dictionary should follow the format::
                {
                    "role": "system" | "user" | "assistant",
                    "content": "Message content as a string",
                }

            Example:

            .. code-block:: json

                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"}
                ]

            Defaults to None if no messages are provided.

        :param invoke_response_format:
            Specifies the format of the returned response. Options:

            - "string": Returns only the generated text content, taken from a single response.
            - "usage": Combines the generated text with metadata (e.g., token usage), returning a dictionary::

                .. code-block:: json
                   {
                       "answer": "<generated_text>",
                       "usage": <ChatCompletion>.to_dict()["usage"]
                   }

            - "full": Returns the full OpenAI `ChatCompletion` object.

        :param invoke_kwargs:
            Additional keyword arguments passed to the OpenAI client.

        :return:
            A string, dictionary, or `ChatCompletion` object, depending on `invoke_response_format`.
        """
        response = await self.async_custom_invoke(messages=messages, **invoke_kwargs)
        return self._response_handler(
            messages=messages,
            invoke_response_format=invoke_response_format,
            response=response,
        )
