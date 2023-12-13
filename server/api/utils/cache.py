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
#
import asyncio
import contextlib
import typing
import uuid


class CachedObject:
    """
    Base class for objects that are cached in the Cache class.
    """

    def __init__(
        self,
        object_id: str,
        lock: asyncio.Lock = None,
        expiry_delayed_call: asyncio.Handle = None,
    ):
        self._object_id = object_id
        self.lock = lock or asyncio.Lock()
        self._expiry_delayed_call: typing.Optional[asyncio.Handle] = expiry_delayed_call

    def matches(self, object_id: str) -> bool:
        return self._object_id == object_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._expiry_delayed_call:
            self._expiry_delayed_call.cancel()


class Cache:
    """
    A cache for objects that are created asynchronously and need to be locked while in use.
    """

    def __init__(self, cls):
        self.lock = asyncio.Lock()
        self.cls = cls
        self.cache: typing.Dict[str, CachedObject] = {}

    @contextlib.asynccontextmanager
    async def get_or_create_locked(
        self,
        key: str,
        ttl: float,
        cls_kwargs: dict = None,
    ) -> typing.Tuple[CachedObject, bool]:
        """
        Get an object from the cache, or create it if it doesn't exist.
        The object will be locked while the context is active.
        :param key: The object's key in the cache.
        :param ttl: The object's time to live in the cache.
        :param cls_kwargs: Keyword arguments to pass to the object's constructor.

        :return:    A CachedObject instance and a boolean indicating whether the object was created.
        """
        created = False
        async with self.lock:
            obj = self.cache.get(key)
            if not obj:
                created = True
                obj = self._create(key, ttl, cls_kwargs)

            # lock the run specific lock before releasing the global lock
            await obj.lock.acquire()

        yield obj, created
        obj.lock.release()

    async def create(
        self,
        key: str,
        ttl: float,
        lock: typing.Optional[asyncio.Lock] = None,
        cls_kwargs=None,
    ) -> CachedObject:
        """
        Create an object in the cache without locking it.
        :param key: The object's key in the cache.
        :param ttl: The object's time to live in the cache.
        :param lock: A lock to use for the object.
        :param cls_kwargs: Keyword arguments to pass to the object's constructor.

        :return:   A CachedObject instance.
        """
        async with self.lock:
            return self._create(key, ttl, lock, cls_kwargs)

    def remove_obj(self, key: str, object_id: str):
        """
        Remove an object from the cache if it exists and matches the given arguments.
        :param key:   The object's key in the cache.
        :param object_id:  The object's id.
        """
        with self.lock:
            obj = self.cache.get(key)
            if obj and obj.matches(object_id):
                del self.cache[key]

    def _create(
        self,
        key: str,
        ttl: float,
        lock: typing.Optional[asyncio.Lock] = None,
        cls_kwargs: dict = None,
    ):
        cls_kwargs = cls_kwargs or {}
        object_id = str(uuid.uuid4())

        # create a delayed call to remove the object from the cache after the ttl expires
        loop = asyncio.get_event_loop()
        expiry_delayed_call = loop.call_later(ttl, self.remove_obj, key, object_id)

        obj = self.cls(
            object_id, lock, expiry_delayed_call=expiry_delayed_call, **cls_kwargs
        )
        self.cache[key] = obj
        return obj
