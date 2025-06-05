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
from abc import ABC
from collections.abc import Awaitable
from typing import Callable, Optional, TypeVar
import mlrun
from mlrun.datastore.abstract_base import BaseRemoteClient

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
        **default_invoke_kwargs,
    ):
        super().__init__(
            parent=parent, name=name, kind=kind, endpoint=endpoint, secrets=secrets
        )
        self.default_invoke_kwargs = default_invoke_kwargs
        self._client = None
        self._default_operation = None

    def load_client(self) -> None:
        raise NotImplementedError("load_client method is not implemented")

    @property
    def client(self):
        return self._client


class AsyncModelProvider(ModelProvider, ABC):
    support_async = True

    def __init__(
        self,
        parent,
        name,
        kind,
        endpoint="",
        secrets: Optional[dict] = None,
        **default_invoke_kwargs,
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


class OpenAIProvider(AsyncModelProvider):
    def __init__(
        self,
        parent,
        name,
        kind,
        endpoint="",
        secrets: Optional[dict] = None,
        **default_invoke_kwargs,
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

    def _validate_model_name(self):
        # endpoint represent model name
        pass

    def load_client(self) -> None:
        try:
            from openai import OpenAI, AsyncOpenAI  # noqa

            self._client = OpenAI(**self.options)
            self._default_operation = self.client.ChatCompletion.create

            self._async_client = AsyncOpenAI(**self.options)
            self._default_async_operation = self.async_client.chat.completions.create
        except ImportError as exc:
            raise ImportError("openai package not installed") from exc

    def get_client_options(self):
        res = dict(
            api_key=self._get_secret_or_env("OPENAI_API_KEY"),
            endpoint_url=self._get_secret_or_env("OPENAI_BASE_URL"),
            open_ai_project_id=self._get_secret_or_env("OPENAI_ORG_ID"),
            openai_org_id=self._get_secret_or_env("OPENAI_PROJECT_ID"),
            timeout=self._get_secret_or_env("OPENAI_TIMEOUT"),
            max_retries=self._get_secret_or_env("OPENAI_MAX_RETRIES"),
        )
        return self._sanitize_options(res)

    def invoke(
        self, operation: Optional[Callable[..., T]] = None, **invoke_kwargs
    ) -> Optional[T]:
        kwargs = self.default_invoke_kwargs.copy()
        kwargs.update(invoke_kwargs)
        if operation:
            return operation(**kwargs, model=self.endpoint)
        else:
            return self._default_operation(**invoke_kwargs, model=self.endpoint)

    async def async_invoke(
        self,
        async_operation: Optional[Callable[..., Awaitable[T]]] = None,
        **invoke_kwargs,
    ) -> Awaitable[T]:
        kwargs = self.default_invoke_kwargs.copy()
        kwargs.update(invoke_kwargs)
        if async_operation:
            return async_operation(**kwargs, model=self.endpoint)
        else:
            return self._default_async_operation(**invoke_kwargs, model=self.endpoint)
