# Copyright 2023 Iguazio
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

import asyncio
import typing
from abc import ABC, abstractmethod

import fastapi
from fastapi.concurrency import run_in_threadpool

import mlrun.common.schemas
import mlrun.errors
import mlrun.utils.singleton


class BaseClient(ABC, metaclass=mlrun.utils.singleton.AbstractSingleton):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._api_url = mlrun.mlconf.iguazio_api_url

    @property
    def is_sync(self) -> bool:
        """Indicates whether the client is synchronous."""
        return True


class BaseAsyncClient(BaseClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._run_in_threadpool_callback = run_in_threadpool
        self._async_session: typing.Optional[mlrun.utils.AsyncClientWithRetry] = None

    @property
    def is_sync(self):
        """
        False because client is asynchronous
        """
        return False

    def __getattribute__(self, name):
        """
        This method is called when trying to access an attribute of the class.
        We override it to make sure that all *public* methods that are not async will be run in a thread pool.
          by convention/norm - public methods are methods that don't start with an underscore.
          If the method name starts with an underscore - it's a private method that was called from a public method,
          which means that it's already running in a thread pool or runs asynchronously.
        If the method is async, we don't do anything and let the async machinery handle it.

        """
        attr = super().__getattribute__(name)
        if name.startswith("_") or not callable(attr):
            return attr

        # already a coroutine
        if asyncio.iscoroutinefunction(attr):
            return attr

        # not a coroutine, run in threadpool
        def wrapper(*args, **kwargs):
            return self._run_in_threadpool_callback(attr, *args, **kwargs)

        return wrapper

    @abstractmethod
    async def verify_request_session(
        self, request: fastapi.Request
    ) -> mlrun.common.schemas.AuthInfo:
        pass
