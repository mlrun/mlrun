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

import mlrun
from .base import DataStore
import redis
import redis.cluster
from storey.redis_driver import RedisType 


class RedisStore(DataStore):
    """
    Partial implementation of DataStore - for use only to delete keys
    """
    def __init__(self, parent, schema, name, endpoint="", redis_type: RedisType = RedisType.STANDALONE):
        super().__init__(parent, name, schema, endpoint)
        self.headers = None

        self.endpoint = self.endpoint or mlrun.mlconf.redis_url

        if self.endpoint.startswith("rediss://"):
            self.endpoint = self.endpoint[len("rediss://") :]
            self.secure = True
        elif self.endpoint.startswith("redis://"):
            self.endpoint = self.endpoint[len("redis://") :]
            self.secure = False
        else:
            raise NotImplementedError(f'invalid endpoint: {endpoint}')

        self._redis_url = f'{schema}://{self.endpoint}'

        self._redis = None
        self._type = redis_type


    @property
    def redis(self):
        if self._redis is None:
            if self._type is RedisType.STANDALONE:
                self._redis = redis.Redis.from_url(self._redis_url, decode_responses=True)
            else:
                self._redis = redis.cluster.RedisCluster.from_url(self._redis_url, decode_response=True)
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
        """
        delete keys, possibly recursively
        """
        if maxdepth is not None:
            raise NotImplementedError('maxdepth is not supported')

        path = path[len("redis://") :]
        count = 0
        if recursive:
            for key in self.redis.scan_iter(path + "*"):
                self.redis.delete(key)
                count += 1
        else:
            self.redis.delete(path)
            count += 1
        # print(f'deleted {count} redis keys', flush=True)
