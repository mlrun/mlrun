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

import os

import redis
import redis.cluster
from storey.redis_driver import RedisType

import mlrun

from .base import DataStore


class RedisStore(DataStore):
    """
    Partial implementation of DataStore over the Redis KV store

    - no support for filesystem
    - key and value sizes are limited to 512MB
    """

    def __init__(
        self,
        parent,
        schema,
        name,
        endpoint="",
    ):
        super().__init__(parent, name, schema, endpoint)
        self.headers = None

        self.endpoint = self.endpoint or mlrun.mlconf.redis.url

        if self.endpoint.startswith("rediss://"):
            self.endpoint = self.endpoint[len("rediss://") :]
            self.secure = True
        elif self.endpoint.startswith("redis://"):
            self.endpoint = self.endpoint[len("redis://") :]
            self.secure = False
        elif self.endpoint == "":
            raise NotImplementedError(f"invalid endpoint: {endpoint}")

        self._redis_url = f"{schema}://{self.endpoint}"

        self._redis = None

        if os.environ.get("MLRUN_REDIS_TYPE") == "cluster":
            self._redis_type = RedisType.CLUSTER
        else:
            self._redis_type = RedisType.STANDALONE

    @property
    def redis(self):
        if self._redis is None:
            if self._redis_type is RedisType.STANDALONE:
                self._redis = redis.Redis.from_url(
                    self._redis_url, decode_responses=True
                )
            else:
                self._redis = redis.cluster.RedisCluster.from_url(
                    self._redis_url, decode_response=True
                )
        return self._redis

    def get_filesystem(self, silent):
        return None  # no support for fsspec

    def supports_isdir(self):
        return False

    def upload(self, key, src_path):
        with open(src_path, "rb") as f:
            while True:
                data = f.read(1000 * 1000)
                if not data:
                    break
                self.redis.append(key, data)

    def get(self, key, size=None, offset=0):
        if offset < 0:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "offset argument should be >= 0"
            )
        start_offset = offset
        if size is None:
            end_offset = -1
        elif size <= 0:
            raise mlrun.errors.MLRunInvalidArgumentError("size argument should be > 0")
        else:
            end_offset = start_offset + size - 1

        return self.redis.getrange(key, start_offset, end_offset)

    def put(self, key, data, append=False):
        if append:
            self.redis.append(key, data)
        else:
            self.redis.set(key, data)

    def stat(self, key):
        raise NotImplementedError()

    def listdir(self, key):
        """
        list all keys with prefix key
        """
        response = []
        key += "*" if key.endswith("/") else "/*"
        for key in self.redis.scan_iter(key):
            response.append(key)
        return response

    def rm(self, key, recursive=False, maxdepth=None):
        """
        delete keys, possibly recursively
        """
        if maxdepth is not None:
            raise NotImplementedError("maxdepth is not supported")

        if key.startswith("redis://"):
            key = key[len("redis://") :]
        elif key.startswith("rediss://"):
            key = key[len("redis://") :]

        if recursive:
            for key in self.redis.scan_iter(key):
                self.redis.delete(key)
        else:
            self.redis.delete(key)
