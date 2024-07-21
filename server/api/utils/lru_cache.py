# Copyright 2024 Iguazio
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

import collections
import hashlib
from copy import deepcopy


class LRUCache:
    """LRU cache which solves some of lru_cache from functools deficiencies.
    Most importantly, the ability to remove/modify a cache value on top
    of the ability to clear entire cache.
    We strive to provide similar API to lru_cache.
    Note this class is used as is instead of using a decorator for performance reasons.
    If performance is not as important, a decorator can be added.
    Note this can be used as a decorator, but only with default parameters i.e. maxsize
    This code is based on https://pastebin.com/LDwMwtp8
    """

    class CacheInfo:
        def __init__(self, maxsize: int):
            self.maxsize = maxsize
            self.reset()

        def reset(self):
            self.hits = 0
            self.misses = 0
            self.currsize = 0

    def __init__(self, func, maxsize: int = 128):
        self.cache = collections.OrderedDict()
        self.func = func
        self.maxsize = maxsize
        self._cache_info = self.CacheInfo(maxsize)

    def __call__(self, *args, **kwargs):
        cache = self.cache
        key = self._gen_key(args, kwargs)
        if key in cache:
            self._cache_info.hits += 1
            cache.move_to_end(key)
            return cache[key]
        result = self.func(*args, **kwargs)
        cache[key] = result
        self._cache_info.misses += 1
        if len(cache) > self.maxsize:
            cache.popitem(last=False)
        return result

    def cache_info(self) -> CacheInfo:
        """Get cache statistics. We emulate lru_cache API.
        We return a deep copy of our internal CacheInfo object to make sure user
        does not accidentally modify our internal structures"""
        self._cache_info.currsize = len(self.cache)
        return deepcopy(self._cache_info)

    def cache_clear(self) -> None:
        """Remove all values from cache and reset statistics"""
        self.cache.clear()
        self._cache_info.reset()

    def cache_remove(self, *args, **kwargs) -> None:
        """Remove an item from the cache by passing the same args used in the call"""
        key = self._gen_key(args, kwargs)
        if key in self.cache:
            self.cache.pop(key)

    def cache_replace(self, value, *args, **kwargs) -> None:
        """Replace value only if in cache. Use cache_set() if you want to make sure the value is set in the cache"""
        cache = self.cache
        key = self._gen_key(args, kwargs)
        if key in cache:
            cache[key] = value
            cache.move_to_end(key)

    def cache_set(self, value, *args, **kwargs) -> None:
        """Set a single value in cache"""
        cache = self.cache
        key = self._gen_key(args, kwargs)
        is_cached = self.cached(key)
        cache[key] = value
        cache.move_to_end(key)
        if not is_cached and len(cache) > self.maxsize:
            cache.popitem(last=False)

    def cached(self, *args, **kwargs) -> bool:
        """Return if argument in cache"""
        key = self._gen_key(args, kwargs)
        return key in self.cache

    @staticmethod
    def _gen_key(args, kwargs) -> str:
        return hashlib.sha256(f"{args}/{sorted(kwargs.items())}".encode()).hexdigest()
