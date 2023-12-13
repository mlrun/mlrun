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
import datetime
import typing
import uuid

import server.api.utils.periodic


class CachedObject:
    """
    Base class for objects that are cached in the Cache class.
    """

    def __init__(
        self,
        ttl: float,
        lock: asyncio.Lock = None,
    ):
        self._object_id = str(uuid.uuid4())
        self.ttl = datetime.datetime.now() + datetime.timedelta(seconds=ttl)
        self.lock = lock or asyncio.Lock()

    @property
    def object_id(self):
        return self._object_id

    def matches(self, object_id: str) -> bool:
        return self._object_id == object_id


class Cache:
    """
    A cache for objects that are created asynchronously and need to be locked while in use.
    """

    def __init__(self, cls):
        if not issubclass(cls, CachedObject):
            raise TypeError("cls must be a subclass of CachedObject")

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
        :param key:         The object's key in the cache.
        :param ttl:         The object's time to live in the cache.
        :param cls_kwargs:  Keyword arguments to pass to the object's constructor.

        :return:    A CachedObject instance and a boolean indicating whether the object was created.
        """
        created = False
        async with self.lock:
            obj = self.cache.get(key)
            now = datetime.datetime.now()
            if not obj or obj.ttl < now:
                created = True
                obj = self._create(key, ttl, cls_kwargs)

        # lock the object after releasing the global lock to avoid deadlocks
        async with obj.lock:
            # make sure that the object wasn't deleted while we were waiting for the lock
            # cleanup ensures that the object will not be deleted while it is locked
            _obj = self.cache.get(key)
            if not _obj.matches(obj.object_id):
                raise RuntimeError("Failed to acquire lock for operation")

            yield obj, created

    async def create(
        self,
        key: str,
        ttl: float,
        lock: typing.Optional[asyncio.Lock] = None,
        cls_kwargs=None,
    ) -> CachedObject:
        """
        Create an object in the cache without locking it.
        :param key:         The object's key in the cache.
        :param ttl:         The object's time to live in the cache.
        :param lock:        A lock to use for the object.
        :param cls_kwargs:  Keyword arguments to pass to the object's constructor.

        :return:   A CachedObject instance.
        """
        async with self.lock:
            return self._create(key, ttl, lock, cls_kwargs)

    def start_periodic_cleanup(self):
        server.api.utils.periodic.run_function_periodically(
            interval=30,  # TODO: make configurable
            name=f"{self.cls.__name__}{self._cleanup.__name__}",
            replace=False,
            function=self._cleanup,
        )

    def _create(
        self,
        key: str,
        ttl: float,
        lock: typing.Optional[asyncio.Lock] = None,
        cls_kwargs: dict = None,
    ):
        cls_kwargs = cls_kwargs or {}
        obj = self.cls(ttl, lock, **cls_kwargs)
        self.cache[key] = obj
        return obj

    async def _cleanup(self):
        async with self.lock:
            now = datetime.datetime.now()
            for key, obj in list(self.cache.items()):
                if obj.ttl < now and not obj.lock.locked():
                    del self.cache[key]
