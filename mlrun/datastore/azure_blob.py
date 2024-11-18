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
from typing import Optional
from urllib.parse import urlparse

from azure.storage.blob import BlobServiceClient
from azure.storage.blob._shared.base_client import parse_connection_str
from fsspec.registry import get_filesystem_class

import mlrun.errors

from .base import DataStore, FileStats, make_datastore_schema_sanitizer

# Azure blobs will be represented with the following URL: az://<container name>. The storage account is already
# pointed to by the connection string, so the user is not expected to specify it in any way.


class AzureBlobStore(DataStore):
    using_bucket = True
    max_concurrency = 100
    max_blocksize = 1024 * 1024 * 4
    max_single_put_size = (
        1024 * 1024 * 8
    )  # for service_client property only, does not affect filesystem

    def __init__(
        self, parent, schema, name, endpoint="", secrets: Optional[dict] = None
    ):
        super().__init__(parent, name, schema, endpoint, secrets=secrets)
        self._service_client = None
        self._storage_options = None

    def get_storage_options(self):
        return self.storage_options

    @property
    def storage_options(self):
        if not self._storage_options:
            res = dict(
                account_name=self._get_secret_or_env("account_name")
                or self._get_secret_or_env("AZURE_STORAGE_ACCOUNT_NAME"),
                account_key=self._get_secret_or_env("account_key")
                or self._get_secret_or_env("AZURE_STORAGE_ACCOUNT_KEY"),
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
            self._storage_options = self._sanitize_storage_options(res)
        return self._storage_options

    @property
    def filesystem(self):
        """return fsspec file system object, if supported"""
        try:
            import adlfs  # noqa
        except ImportError as exc:
            raise ImportError("Azure adlfs not installed") from exc

        if not self._filesystem:
            # in order to support az and wasbs kinds
            filesystem_class = get_filesystem_class(protocol=self.kind)
            self._filesystem = make_datastore_schema_sanitizer(
                filesystem_class,
                using_bucket=self.using_bucket,
                blocksize=self.max_blocksize,
                **self.storage_options,
            )
        return self._filesystem

    @property
    def service_client(self):
        try:
            import azure  # noqa
        except ImportError as exc:
            raise ImportError("Azure not installed") from exc

        if not self._service_client:
            self._do_connect()
        return self._service_client

    def _do_connect(self):
        """

        Creates a client for azure.
        Raises MLRunInvalidArgumentError if none of the connection details are available
        based on do_connect in AzureBlobFileSystem:
        https://github.com/fsspec/adlfs/blob/2023.9.0/adlfs/spec.py#L422
        """
        from azure.identity import ClientSecretCredential

        storage_options = self.storage_options
        connection_string = storage_options.get("connection_string")
        client_name = storage_options.get("account_name")
        account_key = storage_options.get("account_key")
        sas_token = storage_options.get("sas_token")
        client_id = storage_options.get("client_id")
        credential = storage_options.get("credential")

        credential_from_client_id = None
        if (
            credential is None
            and account_key is None
            and sas_token is None
            and client_id is not None
        ):
            credential_from_client_id = ClientSecretCredential(
                tenant_id=storage_options.get("tenant_id"),
                client_id=client_id,
                client_secret=storage_options.get("client_secret"),
            )
        try:
            if connection_string is not None:
                self._service_client = BlobServiceClient.from_connection_string(
                    conn_str=connection_string,
                    max_block_size=self.max_blocksize,
                    max_single_put_size=self.max_single_put_size,
                )
            elif client_name is not None:
                account_url = f"https://{client_name}.blob.core.windows.net"
                cred = credential_from_client_id or credential or account_key
                if not cred and sas_token is not None:
                    if not sas_token.startswith("?"):
                        sas_token = f"?{sas_token}"
                    account_url = account_url + sas_token
                self._service_client = BlobServiceClient(
                    account_url=account_url,
                    credential=cred,
                    max_block_size=self.max_blocksize,
                    max_single_put_size=self.max_single_put_size,
                )
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Must provide either a connection_string or account_name with credentials"
                )
        except Exception as e:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"unable to connect to account for {e}"
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
        container, remote_path = remote_path.split("/", 1)
        container_client = self.service_client.get_container_client(container=container)
        with open(file=src_path, mode="rb") as data:
            container_client.upload_blob(
                name=remote_path,
                data=data,
                overwrite=True,
                max_concurrency=self.max_concurrency,
            )

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
        data, mode = self._prepare_put_data(data, append)
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
        st = self.storage_options
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
                    if not url.endswith(account_key):
                        url += f"@{account_key}"
                    break
        return url
