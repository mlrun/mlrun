# Copyright 2026 Iguazio
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
import threading
from collections.abc import Callable
from typing import Optional, TypeVar

T = TypeVar("T")


# Holds one object per thread
class ThreadLocalClient:
    def __init__(
        self,
        factory: Callable[[], T],
        close_callback: Optional[Callable[[T], None]] = None,
    ):
        """
        Create a thread-local client holder.

        Args:
            factory: Function to create a new instance for each thread
            close_callback: Optional function (sync or async) to close an instance
        """
        self._factory = factory
        self._close_callback = close_callback
        self._local = threading.local()

    def get(self) -> T:
        if not hasattr(self._local, "instance"):
            self._local.instance = self._factory()
        return self._local.instance

    async def async_close(self):
        """Close the current thread's instance, works for both sync and async callbacks."""
        if hasattr(self._local, "instance") and self._close_callback:
            result = self._close_callback(self._local.instance)
            # If it's a coroutine, await it
            if inspect.iscoroutine(result):
                await result
            delattr(self._local, "instance")
