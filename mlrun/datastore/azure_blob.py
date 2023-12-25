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

from azure.storage.blob._shared.base_client import parse_connection_str
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

    def get_spark_options(self):
        res = {}
        st = self.get_storage_options()
        service = "blob"

        if st.get("connection_string"):
            primary_url, _, parsed_credential = parse_connection_str(
                st.get("connection_string"), credential=None, service=service
            )
            for key in ["account_name", "account_key"]:
                if parsed_credential.get(key):
                    if st[key] and st[key] != parsed_credential.get(key):
                        raise mlrun.errors.MLRunInvalidArgumentError(
                            f"'{key}' from the storage_options does not match corresponding connection string"
                        )
                    st[key] = parsed_credential.get(key)

        if not st.get("account_name"):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "'account_name' is missing in the settings"
            )
        account_name = st.get("account_name")
        if primary_url:
            endpoint = primary_url.replace("http://", "").replace("https://", "")
        else:
            endpoint = f"{account_name}.{service}.core.windows.net"
        if st.get("account_key"):
            res[f"spark.hadoop.fs.azure.account.key.{endpoint}"] = st.get("account_key")

        if st.get("client_secret") or st.get("client_id") or st.get("tenant_id"):
            res[f"spark.hadoop.fs.azure.account.auth.type.{endpoint}"] = "OAuth"
            res[
                f"spark.hadoop.fs.azure.account.oauth.provider.type.{endpoint}"
            ] = "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
            if st.get("client_id"):
                res[
                    f"spark.hadoop.fs.azure.account.oauth2.client.id.{endpoint}"
                ] = st.get("client_id")
            if st.get("client_secret"):
                res[
                    f"spark.hadoop.fs.azure.account.oauth2.client.secret.{endpoint}"
                ] = st.get("client_secret")
            if st.get("tenant_id"):
                tenant_id = st.get("tenant_id")
                res[
                    f"spark.hadoop.fs.azure.account.oauth2.client.endpoint.{endpoint}"
                ] = f"https://login.microsoftonline.com/{tenant_id}/oauth2/token"

        if st.get("sas_token"):
            res[f"spark.hadoop.fs.azure.account.auth.type.{endpoint}"] = "SAS"
            res[
                f"spark.hadoop.fs.azure.sas.token.provider.type.{endpoint}"
            ] = "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider"
            res[f"spark.hadoop.fs.azure.sas.fixed.token.{endpoint}"] = st.get(
                "sas_token"
            )
        return res
