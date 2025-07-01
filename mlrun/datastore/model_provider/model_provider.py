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
from typing import Callable, Optional, TypeVar

import mlrun.errors
from mlrun.datastore.remote_client import (
    BaseRemoteClient,
)

T = TypeVar("T")


class ModelProvider(BaseRemoteClient):
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
        self._default_operation = None
        self._async_client = None
        self._default_async_operation = None

    def load_client(self) -> None:
        raise NotImplementedError("load_client method is not implemented")

    def invoke(self, prompt: Optional[str] = None, **invoke_kwargs) -> str:
        raise NotImplementedError("invoke method is not implemented")

    def customized_invoke(
        self, operation: Optional[Callable[..., T]] = None, **invoke_kwargs
    ) -> Optional[T]:
        raise NotImplementedError("customized_invoke method is not implemented")

    @property
    def client(self):
        return self._client

    @property
    def model(self):
        return None

    def get_invoke_kwargs(self, invoke_kwargs):
        kwargs = self.default_invoke_kwargs.copy()
        kwargs.update(invoke_kwargs)
        return kwargs

    @property
    def async_client(self):
        if not self.support_async:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"{self.__class__.__name__} does not support async operations"
            )
        return self._async_client

    async def async_customized_invoke(self, **kwargs):
        raise NotImplementedError("async_customized_invoke is not implemented")

    async def async_invoke(self, prompt: str, **invoke_kwargs) -> Awaitable[str]:
        raise NotImplementedError("async_invoke is not implemented")
