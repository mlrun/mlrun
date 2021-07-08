# Copyright 2018 Iguazio
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

import boto3
import fsspec

import mlrun.errors

from .base import DataStore, FileStats, get_range


class S3Store(DataStore):
    def __init__(self, parent, schema, name, endpoint=""):
        super().__init__(parent, name, schema, endpoint)
        region = None

        access_key = self._secret("AWS_ACCESS_KEY_ID")
        secret_key = self._secret("AWS_SECRET_ACCESS_KEY")

        if access_key or secret_key:
            self.s3 = boto3.resource(
                "s3",
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
        else:
            # from env variables
            self.s3 = boto3.resource("s3", region_name=region)

    def get_filesystem(self, silent=True):
        """return fsspec file system object, if supported"""
        if self._filesystem:
            return self._filesystem
        try:
            import s3fs  # noqa
        except ImportError as exc:
            if not silent:
                raise ImportError(
                    f"AWS s3fs not installed, run pip install s3fs, {exc}"
                )
            return None
        self._filesystem = fsspec.filesystem("s3", **self.get_storage_options())
        return self._filesystem

    def get_storage_options(self):
        return dict(
            anon=False,
            key=self._get_secret_or_env("AWS_ACCESS_KEY_ID"),
            secret=self._get_secret_or_env("AWS_SECRET_ACCESS_KEY"),
        )

    def upload(self, key, src_path):
        self.s3.Object(self.endpoint, self._join(key)[1:]).put(
            Body=open(src_path, "rb")
        )

    def get(self, key, size=None, offset=0):
        obj = self.s3.Object(self.endpoint, self._join(key)[1:])
        if size or offset:
            return obj.get(Range=get_range(size, offset))["Body"].read()
        return obj.get()["Body"].read()

    def put(self, key, data, append=False):
        self.s3.Object(self.endpoint, self._join(key)[1:]).put(Body=data)

    def stat(self, key):
        obj = self.s3.Object(self.endpoint, self._join(key)[1:])
        size = obj.content_length
        modified = obj.last_modified
        return FileStats(size, time.mktime(modified.timetuple()))

    def listdir(self, key):
        if not key.endswith("/"):
            key += "/"
        key_length = len(key)
        bucket = self.s3.Bucket(self.endpoint)
        return [obj.key[key_length:] for obj in bucket.objects.filter(Prefix=key)]


def parse_s3_bucket_and_key(s3_path):
    try:
        path_parts = s3_path.replace("s3://", "").split("/")
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
    except Exception as exc:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"failed to parse s3 bucket and key. {exc}"
        )

    return bucket, key
