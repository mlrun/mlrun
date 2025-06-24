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
import warnings
from abc import ABC
from collections.abc import Awaitable
from typing import Callable, Optional, TypeVar

import mlrun
from mlrun.datastore.remote_client import (
    BaseRemoteClient,
)

T = TypeVar("T")


class ModelProvider(BaseRemoteClient, ABC):
    support_async = False

    def __init__(
        self,
        parent,
        name,
        kind,
        endpoint="",
        secrets: Optional[dict] = None,
        default_invoke_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            parent=parent, name=name, kind=kind, endpoint=endpoint, secrets=secrets
        )
        self.default_invoke_kwargs = default_invoke_kwargs or {}
        self._client = None
        self._default_operation = None

    def load_client(self) -> None:
        raise NotImplementedError("load_client method is not implemented")

    def invoke(self, prompt, **invoke_kwargs) -> str:
        raise NotImplementedError("invoke method is not implemented")

    def customized_invoke(
        self, operation: Optional[Callable[..., T]] = None, **invoke_kwargs
    ) -> Optional[T]:
        raise NotImplementedError("customized_invoke method is not implemented")

    @property
    def client(self):
        return self._client

    @classmethod
    def parse_endpoint_and_path(cls, endpoint, subpath) -> (str, str):
        return endpoint, subpath

    @property
    def model(self):
        return None

    def get_invoke_kwargs(self, invoke_kwargs):
        kwargs = self.default_invoke_kwargs.copy()
        kwargs.update(invoke_kwargs)
        return kwargs


class AsyncModelProvider(ModelProvider, ABC):
    support_async = True

    def __init__(
        self,
        parent,
        name,
        kind,
        endpoint="",
        secrets: Optional[dict] = None,
        default_invoke_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            parent=parent,
            name=name,
            kind=kind,
            endpoint=endpoint,
            secrets=secrets,
            default_invoke_kwargs=default_invoke_kwargs,
        )
        self._async_client = None
        self._default_async_operation = None

    @property
    def async_client(self):
        return self._async_client

    async def async_customized_invoke(self, **kwargs):
        raise NotImplementedError("async_customized_invoke is not implemented")

    async def async_invoke(self, prompt: str, **invoke_kwargs) -> Awaitable[str]:
        raise NotImplementedError("async_invoke is not implemented")


class OpenAIProvider(AsyncModelProvider):
    def __init__(
        self,
        parent,
        name,
        kind,
        endpoint="",
        secrets: Optional[dict] = None,
        default_invoke_kwargs: Optional[dict] = None,
    ):
        endpoint = endpoint or mlrun.mlconf.model_providers.openai_default_model
        super().__init__(
            parent=parent,
            name=name,
            kind=kind,
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

    @property
    def model(self):
        return self.endpoint

    def load_client(self) -> None:
        try:
            from openai import OpenAI, AsyncOpenAI  # noqa

            self._client = OpenAI(**self.options)
            self._default_operation = self.client.chat.completions.create

            self._async_client = AsyncOpenAI(**self.options)
            self._default_async_operation = self.async_client.chat.completions.create
        except ImportError as exc:
            raise ImportError("openai package is not installed") from exc

    def get_client_options(self):
        res = dict(
            api_key=self._get_secret_or_env("OPENAI_API_KEY"),
            organization=self._get_secret_or_env("OPENAI_ORG_ID"),
            project=self._get_secret_or_env("OPENAI_PROJECT_ID"),
            base_url=self._get_secret_or_env("OPENAI_BASE_URL"),
            timeout=self._get_secret_or_env("OPENAI_TIMEOUT"),
            max_retries=self._get_secret_or_env("OPENAI_MAX_RETRIES"),
        )
        return self._sanitize_options(res)

    def customized_invoke(
        self, operation: Optional[Callable[..., T]] = None, **invoke_kwargs
    ) -> Optional[T]:
        invoke_kwargs = self.get_invoke_kwargs(invoke_kwargs)
        if operation:
            return operation(**invoke_kwargs, model=self.model)
        else:
            return self._default_operation(**invoke_kwargs, model=self.model)

    async def async_customized_invoke(
        self,
        async_operation: Optional[Callable[..., Awaitable[T]]] = None,
        **invoke_kwargs,
    ) -> Awaitable[T]:
        invoke_kwargs = self.get_invoke_kwargs(invoke_kwargs)
        if async_operation:
            return async_operation(**invoke_kwargs, model=self.model)
        else:
            return self._default_async_operation(**invoke_kwargs, model=self.model)

    def get_messages_parameter(self, prompt: str, **invoke_kwargs) -> (str, dict):
        invoke_kwargs = self.get_invoke_kwargs(invoke_kwargs)
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        if invoke_kwargs.get("messages"):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "can not provide 'messages' as an customized_invoke argument in "
                "invoke"
            )
        return messages, invoke_kwargs

    async def async_invoke(self, prompt: str, **invoke_kwargs) -> Awaitable[str]:
        messages, invoke_kwargs = self.get_messages_parameter(
            prompt=prompt, **invoke_kwargs
        )
        response = await self._default_async_operation(
            model=self.endpoint, messages=messages, **invoke_kwargs
        )
        return response.choices[0].message.content

    def invoke(self, prompt: str, **invoke_kwargs) -> str:
        messages, invoke_kwargs = self.get_messages_parameter(
            prompt=prompt, **invoke_kwargs
        )
        response = self._default_operation(
            model=self.endpoint, messages=messages, **invoke_kwargs
        )
        return response.choices[0].message.content


def schema_to_model_provider(schema: str, raise_exception=True) -> type[ModelProvider]:
    #  TODO add hugging face and http
    schema_dict = {"openai": OpenAIProvider}
    provider_class = schema_dict.get(schema, None)
    if not provider_class:
        if raise_exception:
            raise ValueError(f"unsupported model provider schema ({schema})")
        else:
            warnings.warn(f"unsupported model provider schema: {schema}")
    return provider_class
