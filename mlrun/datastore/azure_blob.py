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
from pathlib import Path

import fsspec
from azure.storage.blob import BlobServiceClient

import mlrun.errors
from mlrun.errors import err_to_str

from .base import DataStore, FileStats

# Azure blobs will be represented with the following URL: az://<container name>. The storage account is already
# pointed to by the connection string, so the user is not expected to specify it in any way.


class AzureBlobStore(DataStore):
    def __init__(self, parent, schema, name, endpoint="", secrets: dict = None):
        super().__init__(parent, name, schema, endpoint, secrets=secrets)
        self.bsc = None

        con_string = self._get_secret_or_env("AZURE_STORAGE_CONNECTION_STRING")
        if con_string:
            self.bsc = BlobServiceClient.from_connection_string(con_string)
        else:
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
        self._filesystem = fsspec.filesystem(self.kind, **self.get_storage_options())
        return self._filesystem

    def get_storage_options(self):
        return dict(
            account_name=self._get_secret_or_env("AZURE_STORAGE_ACCOUNT_NAME")
            or self._get_secret_or_env("account_name"),
            account_key=self._get_secret_or_env("AZURE_STORAGE_KEY")
            or self._get_secret_or_env("account_key"),
            connection_string=self._get_secret_or_env("AZURE_STORAGE_CONNECTION_STRING")
            or self._get_secret_or_env("connection_string"),
            tenant_id=self._get_secret_or_env("AZURE_STORAGE_TENANT_ID")
            or self._get_secret_or_env("tenant_id"),
            client_id=self._get_secret_or_env("AZURE_STORAGE_CLIENT_ID")
            or self._get_secret_or_env("client_id"),
            client_secret=self._get_secret_or_env("AZURE_STORAGE_CLIENT_SECRET")
            or self._get_secret_or_env("client_secret"),
            sas_token=self._get_secret_or_env("AZURE_STORAGE_SAS_TOKEN")
            or self._get_secret_or_env("sas_token"),
            credential=self._get_secret_or_env("credential"),
        )

    def _convert_key_to_remote_path(self, key):
        key = key.strip("/")
        path = Path(self.endpoint, key).as_posix()
        return path

    def upload(self, key, src_path):
        if self.bsc:
            # Need to strip leading / from key
            with self.bsc.get_blob_client(
                container=self.endpoint, blob=key[1:]
            ) as blob_client:
                with open(src_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
        else:
            remote_path = self._convert_key_to_remote_path(key)
            self._filesystem.put_file(src_path, remote_path, overwrite=True)

    def get(self, key, size=None, offset=0):
        if self.bsc:
            with self.bsc.get_blob_client(
                container=self.endpoint, blob=key[1:]
            ) as blob_client:
                size = size if size else None
                blob = blob_client.download_blob(offset, size).readall()
                return blob
        else:
            remote_path = self._convert_key_to_remote_path(key)
            end = offset + size if size else None
            blob = self._filesystem.cat_file(remote_path, start=offset, end=end)
            return blob

    def put(self, key, data, append=False):
        if append:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Append mode not supported for Azure blob datastore"
            )
        if self.bsc:
            with self.bsc.get_blob_client(
                container=self.endpoint, blob=key[1:]
            ) as blob_client:
                # Note that append=True is not supported. If the blob already exists, this call will fail
                blob_client.upload_blob(data, overwrite=True)
        else:
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
        if self.bsc:
            with self.bsc.get_blob_client(
                container=self.endpoint, blob=key[1:]
            ) as blob_client:
                props = blob_client.get_blob_properties()
            size = props.size
            modified = props.last_modified
        else:
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
        if self.bsc:
            if key and not key.endswith("/"):
                key = key[1:] + "/"
            key_length = len(key)
            with self.bsc.get_container_client(self.endpoint) as container_client:
                blob_list = container_client.list_blobs(name_starts_with=key)
            return [blob.name[key_length:] for blob in blob_list]
        else:
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
