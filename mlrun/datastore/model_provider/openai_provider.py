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
from mlrun.datastore.model_provider.model_provider import ModelProvider
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

    def invoke(
        self,
        messages: Optional[list[dict]] = None,
        as_str: bool = False,
        **invoke_kwargs,
    ) -> Union[str, "ChatCompletion"]:
        """
        OpenAI-specific implementation of `ModelProvider.invoke`.
        Invokes an OpenAI model operation using the sync client.
        For full details, see `ModelProvider.invoke`.

        :param messages:    Same as ModelProvider.invoke.

        :param as_str: bool
                            If `True`, returns only the main content of the first response
                            (`response.choices[0].message.content`).
                            If `False`, returns the full response object, whose type depends on
                            the specific OpenAI SDK operation used (e.g., chat completion, completion, etc.).

        :param invoke_kwargs:
                            Same as ModelProvider.invoke.
        :return:            Same as ModelProvider.invoke.

        """
        response = self.custom_invoke(messages=messages, **invoke_kwargs)
        if as_str:
            return response.choices[0].message.content
        return response

    async def async_invoke(
        self,
        messages: Optional[list[dict]] = None,
        as_str: bool = False,
        **invoke_kwargs,
    ) -> Union[str, "ChatCompletion"]:
        """
        OpenAI-specific implementation of `ModelProvider.async_invoke`.
        Invokes an OpenAI model operation using the async client.
        For full details, see `ModelProvider.async_invoke`.

        :param messages:    Same as ModelProvider.async_invoke.

        :param as_str: bool
                            If `True`, returns only the main content of the first response
                            (`response.choices[0].message.content`).
                            If `False`, returns the full awaited response object, whose type depends on
                            the specific OpenAI SDK operation used (e.g., chat completion, completion, etc.).

        :param invoke_kwargs:
                            Same as ModelProvider.async_invoke.
        :returns            Same as ModelProvider.async_invoke.

        """
        response = await self.async_custom_invoke(messages=messages, **invoke_kwargs)
        if as_str:
            return response.choices[0].message.content
        return response
