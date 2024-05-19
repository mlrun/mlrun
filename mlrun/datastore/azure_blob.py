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

from .base import DataStore, FileStats, makeDatastoreSchemaSanitizer

# Azure blobs will be represented with the following URL: az://<container name>. The storage account is already
# pointed to by the connection string, so the user is not expected to specify it in any way.


class AzureBlobStore(DataStore):
    using_bucket = True

    def __init__(self, parent, schema, name, endpoint="", secrets: dict = None):
        super().__init__(parent, name, schema, endpoint, secrets=secrets)

    @property
    def filesystem(self):
        """return fsspec file system object, if supported"""
        if self._filesystem:
            return self._filesystem
        try:
            import adlfs  # noqa
        except ImportError as exc:
            raise ImportError("Azure adlfs not installed") from exc
        # in order to support az and wasbs kinds.
        filesystem_class = get_filesystem_class(protocol=self.kind)
        self._filesystem = makeDatastoreSchemaSanitizer(
            filesystem_class,
            using_bucket=self.using_bucket,
            **self.get_storage_options(),
        )
        return self._filesystem

    def get_storage_options(self):
        res = dict(
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
        return self._sanitize_storage_options(res)

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
        self.filesystem.put_file(src_path, remote_path, overwrite=True)

    def get(self, key, size=None, offset=0):
        remote_path = self._convert_key_to_remote_path(key)
        end = offset + size if size else None
        blob = self.filesystem.cat_file(remote_path, start=offset, end=end)
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
        with self.filesystem.open(remote_path, mode) as f:
            f.write(data)

    def stat(self, key):
        remote_path = self._convert_key_to_remote_path(key)
        files = self.filesystem.ls(remote_path, detail=True)
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
        if self.filesystem.isfile(remote_path):
            return key
        remote_path = f"{remote_path}/**"
        files = self.filesystem.glob(remote_path)
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
        primary_url = None
        if st.get("connection_string"):
            primary_url, _, parsed_credential = parse_connection_str(
                st.get("connection_string"), credential=None, service=service
            )
            for key in ["account_name", "account_key"]:
                parsed_value = parsed_credential.get(key)
                if parsed_value:
                    if key in st and st[key] != parsed_value:
                        if key == "account_name":
                            raise mlrun.errors.MLRunInvalidArgumentError(
                                f"Storage option for '{key}' is '{st[key]}',\
                                    which does not match corresponding connection string '{parsed_value}'"
                            )
                        else:
                            raise mlrun.errors.MLRunInvalidArgumentError(
                                f"'{key}' from storage options does not match corresponding connection string"
                            )
                    st[key] = parsed_value

        account_name = st.get("account_name")
        if primary_url:
            if primary_url.startswith("http://"):
                primary_url = primary_url[len("http://") :]
            if primary_url.startswith("https://"):
                primary_url = primary_url[len("https://") :]
            host = primary_url
        elif account_name:
            host = f"{account_name}.{service}.core.windows.net"
        else:
            return res

        if "account_key" in st:
            res[f"spark.hadoop.fs.azure.account.key.{host}"] = st["account_key"]

        if "client_secret" in st or "client_id" in st or "tenant_id" in st:
            res[f"spark.hadoop.fs.azure.account.auth.type.{host}"] = "OAuth"
            res[f"spark.hadoop.fs.azure.account.oauth.provider.type.{host}"] = (
                "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
            )
            if "client_id" in st:
                res[f"spark.hadoop.fs.azure.account.oauth2.client.id.{host}"] = st[
                    "client_id"
                ]
            if "client_secret" in st:
                res[f"spark.hadoop.fs.azure.account.oauth2.client.secret.{host}"] = st[
                    "client_secret"
                ]
            if "tenant_id" in st:
                tenant_id = st["tenant_id"]
                res[f"spark.hadoop.fs.azure.account.oauth2.client.endpoint.{host}"] = (
                    f"https://login.microsoftonline.com/{tenant_id}/oauth2/token"
                )

        if "sas_token" in st:
            res[f"spark.hadoop.fs.azure.account.auth.type.{host}"] = "SAS"
            res[f"spark.hadoop.fs.azure.sas.token.provider.type.{host}"] = (
                "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider"
            )
            res[f"spark.hadoop.fs.azure.sas.fixed.token.{host}"] = st["sas_token"]
        return res

    @property
    def spark_url(self):
        spark_options = self.get_spark_options()
        url = f"wasbs://{self.endpoint}"
        prefix = "spark.hadoop.fs.azure.account.key."
        if spark_options:
            for key in spark_options:
                if key.startswith(prefix):
                    account_key = key[len(prefix) :]
                    url += f"@{account_key}"
                    break
        return url
