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

import fsspec
from azure.storage.blob import BlobServiceClient

from .base import DataStore, FileStats

# Azure blobs will be represented with the following URL: az://<container name>. The storage account is already
# pointed to by the connection string, so the user is not expected to specify it in any way.


class AzureBlobStore(DataStore):
    def __init__(self, parent, schema, name, endpoint=""):
        super().__init__(parent, name, schema, endpoint)

        con_string = self._get_secret_or_env("AZURE_STORAGE_CONNECTION_STRING")
        if con_string:
            self.bsc = BlobServiceClient.from_connection_string(con_string)
        else:
            self.bsc = None
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
                    f"Azure adlfs not installed, run pip install adlfs, {exc}"
                )
            return None
        self._filesystem = fsspec.filesystem("az", **self.get_storage_options())
        return self._filesystem

    def get_storage_options(self):
        return dict(
            account_name=self._get_secret_or_env("AZURE_STORAGE_ACCOUNT_NAME"),
            account_key=self._get_secret_or_env("AZURE_STORAGE_KEY"),
            connection_string=self._get_secret_or_env(
                "AZURE_STORAGE_CONNECTION_STRING"
            ),
            tenant_id=self._get_secret_or_env("AZURE_STORAGE_TENANT_ID"),
            client_id=self._get_secret_or_env("AZURE_STORAGE_CLIENT_ID"),
            client_secret=self._get_secret_or_env("AZURE_STORAGE_CLIENT_SECRET"),
            sas_token=self._get_secret_or_env("AZURE_STORAGE_SAS_TOKEN"),
        )

    def upload(self, key, src_path):
        if self.bsc:
            # Need to strip leading / from key
            blob_client = self.bsc.get_blob_client(
                container=self.endpoint, blob=key[1:]
            )
            with open(src_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
        else:
            remote_path = f"{self.endpoint}{key}"
            self._filesystem.put_file(src_path, remote_path)

    def get(self, key, size=None, offset=0):
        if self.bsc:
            blob_client = self.bsc.get_blob_client(
                container=self.endpoint, blob=key[1:]
            )
            size = size if size else None
            return blob_client.download_blob(offset, size).readall()
        else:
            blob = self._filesystem.cat_file(key, start=offset)
            return blob

    def put(self, key, data, append=False):
        if self.bsc:
            blob_client = self.bsc.get_blob_client(
                container=self.endpoint, blob=key[1:]
            )
            # Note that append=True is not supported. If the blob already exists, this call will fail
            blob_client.upload_blob(data, overwrite=True)
        else:
            path = f"{self.endpoint}{key}"
            with self._filesystem.open(path, "wb") as f:
                f.write(data)

    def stat(self, key):
        if self.bsc:
            blob_client = self.bsc.get_blob_client(
                container=self.endpoint, blob=key[1:]
            )
            props = blob_client.get_blob_properties()
            size = props.size
            modified = props.last_modified
        else:
            path = f"{self.endpoint}{key}"
            files = self._filesystem.ls(path, detail=True)
            if len(files) == 1 and files[0]["kind"] == "file":
                size = files[0]["size"]
                modified = files[0]["last_modified"]
            elif len(files) == 1 and files[0]["kind"] == "directory":
                raise FileNotFoundError("Operation expects a file not a directory!")
            else:
                raise ValueError("Operation expects to receive a single file!")
        return FileStats(size, time.mktime(modified.timetuple()))

    def listdir(self, key):
        if key and not key.endswith("/"):
            key = key[1:] + "/"
        if self.bsc:
            key_length = len(key)
            container_client = self.bsc.get_container_client(self.endpoint)
            blob_list = container_client.list_blobs(name_starts_with=key)
            return [blob.name[key_length:] for blob in blob_list]
        else:
            path = f"{self.endpoint}{key}"
            files = self._filesystem.ls(path)
            return files
