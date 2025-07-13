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
from typing import Callable, Optional, TypeVar, Union

import mlrun
from mlrun.datastore.model_provider.model_provider import ModelProvider

T = TypeVar("T")
ChatType = list[dict[str, str]]  # according to transformers.pipelines.text_generation


class HuggingFaceProvider(ModelProvider):
    """
    HuggingFaceProvider is a wrapper around the OpenAI SDK that provides an interface
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
        self.load_client()

    @classmethod
    def parse_endpoint_and_path(cls, endpoint, subpath) -> (str, str):
        if endpoint and subpath:
            endpoint = endpoint + subpath
            #  in hf there is no usage of subpath variable. if the model contains "/", it is part of the model name.
            subpath = ""
        return endpoint, subpath

    def load_client(self) -> None:
        try:
            from transformers import pipeline, AutoModelForCausalLM  # noqa
            from transformers import AutoTokenizer  # noqa

            self._client = pipeline(model=self.model, **self.options)
            self._default_operation = self._client

            # self._async_client = AsyncOpenAI(**self.options)
            # self._default_async_operation = self.async_client.chat.completions.create
        except ImportError as exc:
            raise ImportError("openai package is not installed") from exc

    def get_client_options(self):
        res = dict(
            task=self._get_secret_or_env("HF_TASK"),
            token=self._get_secret_or_env("HF_TOKEN"),
            device=self._get_secret_or_env("HF_DEVICE"),
            device_map=self._get_secret_or_env("HF_DEVICE_MAP"),
            # base_url=self._get_secret_or_env("OPENAI_BASE_URL"),
            # timeout=self._get_secret_or_env("OPENAI_TIMEOUT"),
            # max_retries=self._get_secret_or_env("OPENAI_MAX_RETRIES"),
        )
        return self._sanitize_options(res)

    def custom_invoke(
        self, operation: Optional[Callable[..., T]] = None, **invoke_kwargs
    ) -> Optional[T]:
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
        :param operation: A callable representing the model operation (e.g., a client method).
        :param invoke_kwargs: Keyword arguments to pass to the operation.
        :return: The full response returned by the operation.

        """
        invoke_kwargs = self.get_invoke_kwargs(invoke_kwargs)
        if operation:
            return operation(**invoke_kwargs)
        else:
            return self._default_operation(**invoke_kwargs)

    async def async_custom_invoke(
        self,
        operation: Optional[Callable[..., Awaitable[T]]] = None,
        **invoke_kwargs,
    ) -> Optional[T]:
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
        :param operation: An async callable representing the model operation (e.g., an async_client method).
        :param invoke_kwargs: Keyword arguments to pass to the operation.
        :return: The full response returned by the awaited operation.

        """
        invoke_kwargs = self.get_invoke_kwargs(invoke_kwargs)
        if operation:
            return await operation(**invoke_kwargs)
        else:
            return await self._default_async_operation(**invoke_kwargs)

    def invoke(
        self,
        messages: Union[str, list[str], ChatType, list[ChatType]] = None,
        as_str: bool = False,
        **invoke_kwargs,
    ) -> Optional[Union[str, T]]:
        invoke_kwargs = self.get_invoke_kwargs(invoke_kwargs)
        response = self._default_operation(messages, **invoke_kwargs)
        if as_str:
            return response[0]["generated_text"]
        return response

    async def async_invoke(
        self,
        messages: Optional[list[dict]] = None,
        as_str: bool = False,
        **invoke_kwargs,
    ) -> str:
        invoke_kwargs = self.get_invoke_kwargs(invoke_kwargs)
        response = await self._default_async_operation(
            model=self.endpoint, messages=messages, **invoke_kwargs
        )
        if as_str:
            return response.choices[0].message.content
        return response
