# Copyright 2022 Iguazio
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

from enum import Enum

import redis
import rediscluster

import mlrun

from .base import DataStore


class RedisType(Enum):
    STANDALONE = 1
    CLUSTER = 2


class RedisStore(DataStore):
    def __init__(self, parent, schema, name, endpoint=""):
        super().__init__(parent, name, schema, endpoint)
        self.endpoint = self.endpoint or mlrun.mlconf.redis_url
        self.headers = None

        self._type = RedisType.STANDALONE  # TODO support cluster

        if self.endpoint.startswith("redis://"):
            self.endpoint = self.endpoint[len("redis://") :]

        # TODO
        self.auth = None
        self.token = None
        self.secure = False

    @property
    def url(self):
        schema = "redis"
        return f"{schema}://{self.endpoint}"

    @property
    def redis(self):
        if hasattr(self, "_redis"):
            return self._redis
        if self._type is RedisType.STANDALONE:
            self._redis = redis.Redis.from_url(self.url, decode_responses=True)
        else:
            self._redis = rediscluster.RedisCluster.from_url(
                self.url, decode_response=True
            )
        return self._redis

    def get_filesystem(self, silent):
        return None  # no support for fsspec

    def upload(self, key, src_path):
        raise NotImplementedError()

    def get(self, key, size, offset):
        raise NotImplementedError()

    def put(self, key, data, append):
        raise NotImplementedError()

    def stat(self, key):
        raise NotImplementedError()

    def listdir(self, key):
        raise NotImplementedError()

    def rm(self, path, recursive=False, maxdepth=None):
        """delete all keys under a prefix"""
        count = 0
        new_path = "projects" + path[len("redis:///projects") :]
        ns_keys = "storey:" + new_path + "*"
        for key in self.redis.scan_iter(ns_keys):
            self.redis.delete(key)
            count += 1
        # print(f"deleted {count} keys")
