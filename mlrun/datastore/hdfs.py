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
import os
from typing import Optional
from urllib.parse import urlparse

import fsspec

from mlrun.datastore.base import DataStore


class HdfsStore(DataStore):
    def __init__(
        self, parent, schema, name, endpoint="", secrets: Optional[dict] = None
    ):
        super().__init__(parent, name, schema, endpoint, secrets)

        self.host = self._get_secret_or_env("HDFS_HOST")
        self.port = self._get_secret_or_env("HDFS_PORT")
        self.http_port = self._get_secret_or_env("HDFS_HTTP_PORT")
        self.user = self._get_secret_or_env("HDFS_USER")
        if not self.user:
            self.user = os.environ.get("HADOOP_USER_NAME", os.environ.get("USER"))

        self._filesystem = None

    @property
    def filesystem(self):
        if not self._filesystem:
            self._filesystem = fsspec.filesystem(
                "webhdfs",
                host=self.host,
                port=self.http_port,
                user=self.user,
            )
        return self._filesystem

    @property
    def url(self):
        return f"webhdfs://{self.host}:{self.http_port}"

    @property
    def spark_url(self):
        return f"hdfs://{self.host}:{self.port}"

    def rm(self, url, recursive=False, maxdepth=None):
        path = urlparse(url).path
        self.filesystem.rm(path=path, recursive=recursive, maxdepth=maxdepth)
