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
import json
import os
from pathlib import Path
from typing import Optional

from fsspec.registry import get_filesystem_class
from google.auth.credentials import Credentials
from google.cloud.storage import Client, transfer_manager
from google.oauth2 import service_account

import mlrun.errors
from mlrun.utils import logger

from .base import DataStore, FileStats, make_datastore_schema_sanitizer

# Google storage objects will be represented with the following URL: gcs://<bucket name>/<path> or gs://...


class GoogleCloudStorageStore(DataStore):
    using_bucket = True
    workers = 8
    chunk_size = 32 * 1024 * 1024

    def __init__(
        self, parent, schema, name, endpoint="", secrets: Optional[dict] = None
    ):
        super().__init__(parent, name, schema, endpoint, secrets=secrets)
        self._storage_client = None
        self._storage_options = None

    @property
    def storage_client(self):
        if self._storage_client:
            return self._storage_client

        token = self._get_credentials().get("token")
        access = "https://www.googleapis.com/auth/devstorage.full_control"
        if isinstance(token, str):
            if os.path.exists(token):
                credentials = service_account.Credentials.from_service_account_file(
                    token, scopes=[access]
                )
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "gcsfs authentication file not found!"
                )
        elif isinstance(token, dict):
            credentials = service_account.Credentials.from_service_account_info(
                token, scopes=[access]
            )
        elif isinstance(token, Credentials):
            credentials = token
        else:
            raise ValueError(f"Unsupported token type: {type(token)}")
        self._storage_client = Client(credentials=credentials)
        return self._storage_client

    @property
    def filesystem(self):
        """return fsspec file system object, if supported"""
        if not self._filesystem:
            filesystem_class = get_filesystem_class(protocol=self.kind)
            self._filesystem = make_datastore_schema_sanitizer(
                filesystem_class,
                using_bucket=self.using_bucket,
                **self.storage_options,
            )
        return self._filesystem

    @property
    def storage_options(self):
        if self._storage_options:
            return self._storage_options
        credentials = self._get_credentials()
        # due to caching problem introduced in gcsfs 2024.3.1 (ML-7636)
        credentials["use_listings_cache"] = False
        self._storage_options = credentials
        return self._storage_options

    def _get_credentials(self):
        credentials = self._get_secret_or_env(
            "GCP_CREDENTIALS"
        ) or self._get_secret_or_env("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials:
            try:
                # Try to handle credentials as a json connection string or do nothing if already a dict
                token = (
                    credentials
                    if isinstance(credentials, dict)
                    else json.loads(credentials)
                )
            except json.JSONDecodeError:
                # If it's not json, handle it as a filename
                token = credentials
            return self._sanitize_storage_options(dict(token=token))
        else:
            logger.info(
                "No GCS credentials available - auth will rely on auto-discovery of credentials"
            )
            return self._sanitize_storage_options(None)

    def get_storage_options(self):
        return self.storage_options

    def _make_path(self, key):
        key = key.strip("/")
        path = Path(self.endpoint, key).as_posix()
        return path

    def get(self, key, size=None, offset=0):
        path = self._make_path(key)

        end = offset + size if size else None
        blob = self.filesystem.cat_file(path, start=offset, end=end)
        return blob

    def put(self, key, data, append=False):
        path = self._make_path(key)

        if append:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Append mode not supported for Google cloud storage datastore"
            )
        data, mode = self._prepare_put_data(data, append)
        with self.filesystem.open(path, mode) as f:
            f.write(data)

    def upload(self, key, src_path):
        file_size = os.path.getsize(src_path)
        united_path = self._make_path(key)

        # Multiple upload limitation recommendations as described in
        # https://cloud.google.com/storage/docs/multipart-uploads#storage-upload-object-chunks-python

        if file_size <= self.chunk_size:
            self.filesystem.put_file(src_path, united_path, overwrite=True)
            return

        bucket = self.storage_client.bucket(self.endpoint)
        blob = bucket.blob(key.strip("/"))

        try:
            transfer_manager.upload_chunks_concurrently(
                src_path, blob, chunk_size=self.chunk_size, max_workers=self.workers
            )
        except Exception as upload_chunks_concurrently_exception:
            logger.warning(
                f"gcs: failed to concurrently upload {src_path},"
                f" exception: {upload_chunks_concurrently_exception}. Retrying with single part upload."
            )
            self.filesystem.put_file(src_path, united_path, overwrite=True)

    def stat(self, key):
        path = self._make_path(key)

        files = self.filesystem.ls(path, detail=True)
        if len(files) == 1 and files[0]["type"] == "file":
            size = files[0]["size"]
            modified = files[0]["updated"]
        elif len(files) == 1 and files[0]["type"] == "directory":
            raise FileNotFoundError("Operation expects a file not a directory!")
        else:
            raise ValueError("Operation expects to receive a single file!")
        return FileStats(size, modified)

    def listdir(self, key):
        path = self._make_path(key)
        if self.filesystem.isfile(path):
            return key
        remote_path = f"{path}/**"
        files = self.filesystem.glob(remote_path)
        key_length = len(key)
        files = [
            f.split("/", 1)[1][key_length:] for f in files if len(f.split("/")) > 1
        ]
        return files

    def rm(self, path, recursive=False, maxdepth=None):
        path = self._make_path(path)
        # in order to raise an error in case of a connection error (ML-7056)
        self.filesystem.exists(path)
        super().rm(path, recursive=recursive, maxdepth=maxdepth)

    def get_spark_options(self):
        res = {}
        st = self._get_credentials()
        if "token" in st:
            res = {"spark.hadoop.google.cloud.auth.service.account.enable": "true"}
            if isinstance(st["token"], str):
                # Token is a filename, read json from it
                with open(st["token"]) as file:
                    credentials = json.load(file)
            else:
                # Token is a dictionary, use it directly
                credentials = st["token"]

            if "project_id" in credentials:
                res["spark.hadoop.fs.gs.project.id"] = credentials["project_id"]
            if "private_key_id" in credentials:
                res["spark.hadoop.fs.gs.auth.service.account.private.key.id"] = (
                    credentials["private_key_id"]
                )
            if "private_key" in credentials:
                res["spark.hadoop.fs.gs.auth.service.account.private.key"] = (
                    credentials["private_key"]
                )
            if "client_email" in credentials:
                res["spark.hadoop.fs.gs.auth.service.account.email"] = credentials[
                    "client_email"
                ]
            if "client_id" in credentials:
                res["spark.hadoop.fs.gs.client.id"] = credentials["client_id"]
        return res

    @property
    def spark_url(self):
        return f"gs://{self.endpoint}"
