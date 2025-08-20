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
        self.load_client()

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
        Extracts the first generated string from Hugging Face pipeline output,
        regardless of whether it's plain text-generation or chat-style output.
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

    def load_client(self) -> None:
        """
        Initializes the OpenAI SDK client using the provided options.

        This method imports the `OpenAI` class from the `openai` package, instantiates
        a client with the given keyword arguments (`self.options`), and assigns it to
        `self._client` and `self._async_client`.

        Raises:
            ImportError: If the `openai` package is not installed.
        """
        try:
            from openai import OpenAI, AsyncOpenAI  # noqa

            self._client = OpenAI(**self.options)
            self._async_client = AsyncOpenAI(**self.options)
        except ImportError as exc:
            raise ImportError("openai package is not installed") from exc

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
        OpenAI-specific implementation of `ModelProvider.custom_invoke`.

        Invokes an OpenAI model operation using the sync client. For full details, see
        `ModelProvider.custom_invoke`.

        Example:
            ```python
            result = openai_model_provider.invoke(
                openai_model_provider.client.images.generate,
                prompt="A futuristic cityscape at sunset",
                n=1,
                size="1024x1024",
            )
            ```
        :param      operation:      Same as ModelProvider.custom_invoke.
        :param      invoke_kwargs:  Same as ModelProvider.custom_invoke.
        :return:                    Same as ModelProvider.custom_invoke.

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
        OpenAI-specific implementation of `ModelProvider.async_custom_invoke`.

        Invokes an OpenAI model operation using the async client. For full details, see
        `ModelProvider.async_custom_invoke`.

        Example:
        ```python
            result = openai_model_provider.invoke(
                openai_model_provider.async_client.images.generate,
                prompt="A futuristic cityscape at sunset",
                n=1,
                size="1024x1024",
            )
        ```

        :param operation:       Same as ModelProvider.async_custom_invoke.
        :param invoke_kwargs:   Same as ModelProvider.async_custom_invoke.
        :return:                Same as ModelProvider.async_custom_invoke.

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
                stats = response.to_dict()["usage"]
                response = {
                    UsageResponseKeys.ANSWER: str_response,
                    UsageResponseKeys.USAGE: stats,
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
        For full details, see `ModelProvider.invoke`.

        :param messages:
            Same as `ModelProvider.invoke`.

        :param invoke_response_format: InvokeResponseFormat
            Specifies the format of the returned response. Options:

            - "string": Returns only the generated text content, taken from a single response.
            - "stats": Combines the generated text with metadata (e.g., token usage), returning a dictionary:

              .. code-block:: json
                 {
                     "answer": "<generated_text>",
                     "stats": <ChatCompletion>.to_dict()["usage"]
                 }

            - "full": Returns the full OpenAI `ChatCompletion` object.

        :param invoke_kwargs:
            Additional keyword arguments passed to the OpenAI client. Same as in `ModelProvider.invoke`.

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
        Invokes an OpenAI model operation using the async client.
        For full details, see `ModelProvider.async_invoke` and `OpenAIProvider.invoke`.

        :param messages:    Same as `OpenAIProvider.invoke`.

        :param invoke_response_format: InvokeResponseFormat
                            Same as `OpenAIProvider.invoke`.

        :param invoke_kwargs:
                            Same as `OpenAIProvider.invoke`.
        :returns            Same as `ModelProvider.async_invoke`.

        """
        response = await self.async_custom_invoke(messages=messages, **invoke_kwargs)
        return self._response_handler(
            messages=messages,
            invoke_response_format=invoke_response_format,
            response=response,
        )
