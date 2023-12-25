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
from pathlib import Path
from urllib.parse import urlparse

from fsspec.registry import get_filesystem_class

import mlrun.errors
from mlrun.errors import err_to_str

from .base import DataStore, FileStats, makeDatastoreSchemaSanitizer

# Azure blobs will be represented with the following URL: az://<container name>. The storage account is already
# pointed to by the connection string, so the user is not expected to specify it in any way.


class AzureBlobStore(DataStore):
    using_bucket = True

    def __init__(self, parent, schema, name, endpoint="", secrets: dict = None):
        super().__init__(parent, name, schema, endpoint, secrets=secrets)
        self.get_filesystem()

    def get_filesystem(self, silent=True):
        """return fsspec file system object, if supported"""
        if self._filesystem:
            return self._filesystem
        try:
            import adlfs  # noqa
        except ImportError as exc:
            if not silent:
                raise ImportError(
                    f"Azure adlfs not installed, run pip install adlfs, {err_to_str(exc)}"
                )
            return None
        # in order to support az and wasbs kinds.
        filesystem_class = get_filesystem_class(protocol=self.kind)
        self._filesystem = makeDatastoreSchemaSanitizer(
            filesystem_class,
            using_bucket=self.using_bucket,
            **self.get_storage_options(),
        )
        return self._filesystem

    def get_storage_options(self):
        return dict(
            account_name=self._get_secret_or_env("account_name")
            or self._get_secret_or_env("AZURE_STORAGE_ACCOUNT_NAME"),
            account_key=self._get_secret_or_env("account_key")
            or self._get_secret_or_env("AZURE_STORAGE_KEY"),
            connection_string=self._get_secret_or_env("connection_string")
            or self._get_secret_or_env("AZURE_STORAGE_CONNECTION_STRING"),
            tenant_id=self._get_secret_or_env("tenant_id")
            or self._get_secret_or_env("AZURE_STORAGE_TENANT_ID"),
            client_id=self._get_secret_or_env("client_id")
            or self._get_secret_or_env("AZURE_STORAGE_CLIENT_ID"),
            client_secret=self._get_secret_or_env("client_secret")
            or self._get_secret_or_env("AZURE_STORAGE_CLIENT_SECRET"),
            sas_token=self._get_secret_or_env("sas_token")
            or self._get_secret_or_env("AZURE_STORAGE_SAS_TOKEN"),
            credential=self._get_secret_or_env("credential"),
        )

    def _convert_key_to_remote_path(self, key):
        key = key.strip("/")
        schema = urlparse(key).scheme
        #  if called without passing dataitem - like in fset.purge_targets,
        #  key will include schema.
        if not schema:
            key = Path(self.endpoint, key).as_posix()
        return key

    def upload(self, key, src_path):
        remote_path = self._convert_key_to_remote_path(key)
        self._filesystem.put_file(src_path, remote_path, overwrite=True)

    def get(self, key, size=None, offset=0):
        remote_path = self._convert_key_to_remote_path(key)
        end = offset + size if size else None
        blob = self._filesystem.cat_file(remote_path, start=offset, end=end)
        return blob

    def put(self, key, data, append=False):
        if append:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Append mode not supported for Azure blob datastore"
            )
        remote_path = self._convert_key_to_remote_path(key)
        if isinstance(data, bytes):
            mode = "wb"
        elif isinstance(data, str):
            mode = "w"
        else:
            raise TypeError("Data type unknown.  Unable to put in Azure!")
        with self._filesystem.open(remote_path, mode) as f:
            f.write(data)

    def stat(self, key):
        remote_path = self._convert_key_to_remote_path(key)
        files = self._filesystem.ls(remote_path, detail=True)
        if len(files) == 1 and files[0]["type"] == "file":
            size = files[0]["size"]
            modified = files[0]["last_modified"]
        elif len(files) == 1 and files[0]["type"] == "directory":
            raise FileNotFoundError("Operation expects a file not a directory!")
        else:
            raise ValueError("Operation expects to receive a single file!")
        return FileStats(size, time.mktime(modified.timetuple()))

    def listdir(self, key):
        remote_path = self._convert_key_to_remote_path(key)
        if self._filesystem.isfile(remote_path):
            return key
        remote_path = f"{remote_path}/**"
        files = self._filesystem.glob(remote_path)
        key_length = len(key)
        files = [
            f.split("/", 1)[1][key_length:] for f in files if len(f.split("/")) > 1
        ]
        return files

    def rm(self, path, recursive=False, maxdepth=None):
        path = self._convert_key_to_remote_path(key=path)
        super().rm(path=path, recursive=recursive, maxdepth=maxdepth)
