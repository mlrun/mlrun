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

import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import oss2
from fsspec.registry import get_filesystem_class

import mlrun.errors

from .base import DataStore, FileStats, make_datastore_schema_sanitizer


class OSSStore(DataStore):
    using_bucket = True

    def __init__(
        self, parent, schema, name, endpoint="", secrets: Optional[dict] = None
    ):
        super().__init__(parent, name, schema, endpoint, secrets)
        # will be used in case user asks to assume a role and work through fsspec

        access_key_id = self._get_secret_or_env("ALIBABA_ACCESS_KEY_ID")
        secret_key = self._get_secret_or_env("ALIBABA_SECRET_ACCESS_KEY")
        endpoint_url = self._get_secret_or_env("ALIBABA_ENDPOINT_URL")
        if access_key_id and secret_key and endpoint_url:
            self.auth = oss2.Auth(access_key_id, secret_key)
            self.endpoint_url = endpoint_url
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "missing ALIBABA_ACCESS_KEY_ID or ALIBABA_SECRET_ACCESS_KEY ALIBABA_ENDPOINT_URL in environment"
            )

    @property
    def filesystem(self):
        """return fsspec file system object, if supported"""
        if self._filesystem:
            return self._filesystem
        try:
            import ossfs  # noqa
        except ImportError as exc:
            raise ImportError("ALIBABA ossfs not installed") from exc
        filesystem_class = get_filesystem_class(protocol=self.kind)
        self._filesystem = make_datastore_schema_sanitizer(
            filesystem_class,
            using_bucket=self.using_bucket,
            **self.get_storage_options(),
        )
        return self._filesystem

    def get_storage_options(self):
        res = dict(
            endpoint=self._get_secret_or_env("ALIBABA_ENDPOINT_URL"),
            key=self._get_secret_or_env("ALIBABA_ACCESS_KEY_ID"),
            secret=self._get_secret_or_env("ALIBABA_SECRET_ACCESS_KEY"),
        )
        return self._sanitize_storage_options(res)

    def get_bucket_and_key(self, key):
        path = self._join(key)[1:]
        return self.endpoint, path

    def upload(self, key, src_path):
        bucket, key = self.get_bucket_and_key(key)
        oss = oss2.Bucket(self.auth, self.endpoint_url, bucket)
        oss.put_object(key, open(src_path, "rb"))

    def get(self, key, size=None, offset=0):
        bucket, key = self.get_bucket_and_key(key)
        oss = oss2.Bucket(self.auth, self.endpoint_url, bucket)
        if size or offset:
            return oss.get_object(key, byte_range=self.get_range(size, offset)).read()
        return oss.get_object(key).read()

    def put(self, key, data, append=False):
        data, _ = self._prepare_put_data(data, append)
        bucket, key = self.get_bucket_and_key(key)
        oss = oss2.Bucket(self.auth, self.endpoint_url, bucket)
        oss.put_object(key, data)

    def stat(self, key):
        bucket, key = self.get_bucket_and_key(key)
        oss = oss2.Bucket(self.auth, self.endpoint_url, bucket)
        obj = oss.get_object_meta(key)
        size = obj.content_length
        modified = datetime.fromtimestamp(obj.last_modified)
        return FileStats(size, time.mktime(modified.timetuple()))

    def listdir(self, key):
        remote_path = self._convert_key_to_remote_path(key)
        if self.filesystem.isfile(remote_path):
            return key
        remote_path = f"{remote_path}/**"
        files = self.filesystem.glob(remote_path)
        key_length = len(key)
        files = [
            f.split("/", 1)[1][key_length:] for f in files if len(f.split("/")) > 1
        ]
        return files

    def delete(self, key):
        bucket, key = self.get_bucket_and_key(key)
        oss = oss2.Bucket(self.auth, self.endpoint_url, bucket)
        oss.delete_object(key)

    def _convert_key_to_remote_path(self, key):
        key = key.strip("/")
        schema = urlparse(key).scheme
        #  if called without passing dataitem - like in fset.purge_targets,
        #  key will include schema.
        if not schema:
            key = Path(self.endpoint, key).as_posix()
        return key

    @staticmethod
    def get_range(size, offset):
        if size:
            return [offset, size]
        return [offset, None]
