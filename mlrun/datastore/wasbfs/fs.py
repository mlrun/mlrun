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

from typing import Optional
from urllib.parse import urlparse

from fsspec import AbstractFileSystem


class WasbFS(AbstractFileSystem):
    protocol = "wasb"

    def __init__(
        self,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        connection_string: Optional[str] = None,
        credential: Optional[str] = None,
        sas_token: Optional[str] = None,
        request_session=None,
        socket_timeout: Optional[int] = None,
        blocksize: Optional[int] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
        anon: bool = True,
        location_mode: Optional[str] = None,
        loop=None,
        asynchronous: bool = False,
        default_fill_cache: bool = True,
        default_cache_type: Optional[str] = None,
        **kwargs,
    ):
        from adlfs import AzureBlobFileSystem

        self.azure_blob_fs = AzureBlobFileSystem(
            account_name,
            account_key,
            connection_string,
            credential,
            sas_token,
            request_session,
            socket_timeout,
            blocksize,
            client_id,
            client_secret,
            tenant_id,
            anon,
            location_mode,
            loop,
            asynchronous,
            default_fill_cache,
            default_cache_type,
            **kwargs,
        )

    @staticmethod
    def _convert_wasb_schema_to_az(url):
        # convert wasbs schema url to az schema url. Used before passing the url to AzureBlobFS.
        # wasbs://mycontainer@myaccount/path/to/obj is equivalent to
        # az://mycontainer/path/to/obj
        az_path = url
        parsed_url = urlparse(url)
        if parsed_url.scheme:
            if (
                parsed_url.scheme.lower() == "wasb"
                or parsed_url.scheme.lower() == "wasbs"
            ):
                az_path = (
                    "az://"
                    + parsed_url.username
                    + ("/" if not parsed_url.path.startswith("/") else parsed_url.path)
                )
            else:
                raise ValueError("Operation expects to wasb or wasbs scheme only!")
        return az_path

    def info(self, path, refresh=False, **kwargs):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.info(az_path, refresh, **kwargs)

    def glob(self, path, **kwargs):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.glob(az_path, **kwargs)

    def ls(
        self,
        path: str,
        detail: bool = False,
        invalidate_cache: bool = False,
        delimiter: str = "/",
        return_glob: bool = False,
        **kwargs,
    ):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.ls(
            az_path,
            detail,
            invalidate_cache,
            delimiter,
            return_glob,
            **kwargs,
        )

    def find(self, path, withdirs=False, prefix="", **kwargs):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.find(az_path, withdirs, prefix, **kwargs)

    def mkdir(self, path, exist_ok=False):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.mkdir(az_path, exist_ok)

    def rmdir(self, path: str, delimiter="/", **kwargs):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.rmdir(az_path, delimiter, **kwargs)

    def size(self, path):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.size(az_path)

    def isfile(self, path):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.isfile(az_path)

    def isdir(self, path):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.isdir(az_path)

    def exists(self, path):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.exists(az_path)

    def cat(self, path, recursive=False, on_error="raise", **kwargs):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.cat(az_path, recursive, on_error, **kwargs)

    def url(self, path, expires=3600, **kwargs):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.url(az_path, expires, **kwargs)

    def expand_path(self, path, recursive=False, maxdepth=None):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.expand_path(az_path, recursive, maxdepth)

    def rm(self, path, recursive=False, maxdepth=None, **kwargs):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.rm(az_path, recursive, maxdepth, **kwargs)

    def open(self, path, mode="rb", block_size=None, cache_options=None, **kwargs):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.open(
            az_path, mode, block_size, cache_options, **kwargs
        )

    def touch(self, path, truncate=True, **kwargs):
        az_path = self._convert_wasb_schema_to_az(path)
        return self.azure_blob_fs.touch(az_path, truncate, **kwargs)

    @classmethod
    def _strip_protocol(cls, path: str):
        az_path = cls._convert_wasb_schema_to_az(path)
        from adlfs import AzureBlobFileSystem

        return AzureBlobFileSystem._strip_protocol(az_path)

    @staticmethod
    def _get_kwargs_from_urls(paths):
        from adlfs import AzureBlobFileSystem

        return AzureBlobFileSystem._get_kwargs_from_urls(paths)
