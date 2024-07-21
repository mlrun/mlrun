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

import unittest

import server.api.utils.lru_cache


class LRUTest(unittest.TestCase):
    def test_lru(self):
        lru = server.api.utils.lru_cache.LRUCache(self._func_getter, maxsize=3)
        lru("1")
        lru("2")
        lru("3")
        lru("4")
        lru("5")

        # test eviction
        self.assertFalse(lru.cached("1"))
        self.assertFalse(lru.cached("2"))
        self.assertTrue(lru.cached("3"))
        self.assertTrue(lru.cached("4"))
        self.assertTrue(lru.cached("5"))

        lru.cache_remove("5")
        self.assertFalse(lru.cached("5"))

        lru.cache_set(5, "5")
        self.assertTrue(lru.cached("5"))

        info = lru.cache_info()
        self.assertTrue(info.currsize == 3)
        self.assertTrue(info.misses == 5)
        self.assertTrue(info.maxsize == 3)
        self.assertTrue(info.hits == 0)

        lru("5")
        info2 = lru.cache_info()
        self.assertTrue(info2.currsize == 3)
        self.assertTrue(info2.misses == 5)
        self.assertTrue(info2.hits == 1)
        # check that returned cache_info is not the internal object
        self.assertFalse(info.hits == info2.hits)

        # make sure we don't care about kwargs order
        lru("6", increment=True, decrement=False)
        self.assertTrue(lru.cached("6", decrement=False, increment=True))
        self.assertFalse(lru.cached("6"))

        # verify replace semantics
        lru.cache_replace(123, "123")
        self.assertFalse(lru.cached("123"))

        # verify set semantics
        lru.cache_set(123, "123")
        self.assertTrue(lru.cached("123"))

        lru.cache_clear()
        info = lru.cache_info()
        self.assertTrue(info.hits == 0)
        self.assertTrue(info.misses == 0)
        self.assertTrue(info.currsize == 0)

    def _func_getter(self, arg, increment=False, decrement=False):
        result = int(arg)
        if increment:
            result += 1
        if decrement:
            result -= 1
