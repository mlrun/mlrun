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
from collections.abc import Awaitable
from typing import Callable, Optional, TypeVar

import mlrun.datastore.model_provider.openai_provider
from mlrun.datastore.remote_client import (
    BaseRemoteClient,
)

T = TypeVar("T")


class ModelProvider(BaseRemoteClient):
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


class AsyncModelProvider(ModelProvider):
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


def schema_to_model_provider(schema: str, raise_exception=True) -> type[ModelProvider]:
    #  TODO add hugging face and http
    schema_dict = {
        "openai": mlrun.datastore.model_provider.openai_provider.OpenAIProvider
    }
    provider_class = schema_dict.get(schema, None)
    if not provider_class:
        if raise_exception:
            raise ValueError(f"unsupported model provider schema ({schema})")
        else:
            warnings.warn(f"unsupported model provider schema: {schema}")
    return provider_class
