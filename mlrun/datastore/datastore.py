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

from urllib.parse import urlparse

import mlrun
import mlrun.errors

from ..utils import DB_SCHEMA, run_keys
from .base import DataItem, DataStore, HttpStore
from .filestore import FileStore
from .inmem import InMemoryStore
from .store_resources import get_store_resource, is_store_uri
from .v3io import V3ioStore

in_memory_store = InMemoryStore()


def parse_url(url):
    parsed_url = urlparse(url)
    schema = parsed_url.scheme.lower()
    endpoint = parsed_url.hostname
    if endpoint:
        # HACK - urlparse returns the hostname after in lower case - we want the original case:
        # the hostname is a substring of the netloc, in which it's the original case, so we find the indexes of the
        # hostname in the netloc and take it from there
        lower_hostname = parsed_url.hostname
        netloc = str(parsed_url.netloc)
        lower_netloc = netloc.lower()
        hostname_index_in_netloc = lower_netloc.index(str(lower_hostname))
        endpoint = netloc[
            hostname_index_in_netloc : hostname_index_in_netloc + len(lower_hostname)
        ]
    if parsed_url.port:
        endpoint += f":{parsed_url.port}"
    return schema, endpoint, parsed_url


def schema_to_store(schema):
    # import store classes inside to enable making their dependencies optional (package extras)
    if not schema or schema in ["file", "c", "d"]:
        return FileStore
    elif schema == "s3":
        try:
            from .s3 import S3Store
        except ImportError:
            raise mlrun.errors.MLRunMissingDependencyError(
                "s3 packages are missing, use pip install mlrun[s3]"
            )

        return S3Store
    elif schema == "az":
        try:
            from .azure_blob import AzureBlobStore
        except ImportError:
            raise mlrun.errors.MLRunMissingDependencyError(
                "azure blob storage packages are missing, use pip install mlrun[azure-blob-storage]"
            )

        return AzureBlobStore
    elif schema in ["v3io", "v3ios"]:
        return V3ioStore
    elif schema in ["http", "https"]:
        return HttpStore
    else:
        raise ValueError(f"unsupported store scheme ({schema})")


def uri_to_ipython(link):
    schema, endpoint, parsed_url = parse_url(link)
    if schema in [DB_SCHEMA, "memory"]:
        return ""
    return schema_to_store(schema).uri_to_ipython(endpoint, parsed_url.path)


class StoreManager:
    def __init__(self, secrets=None, db=None):
        self._stores = {}
        self._secrets = secrets or {}
        self._db = db

    def set(self, secrets=None, db=None):
        if db and not self._db:
            self._db = db
        if secrets:
            for key, val in secrets.items():
                self._secrets[key] = val
        return self

    def _get_db(self):
        if not self._db:
            self._db = mlrun.get_run_db(secrets=self._secrets)
        return self._db

    def from_dict(self, struct: dict):
        stor_list = struct.get(run_keys.data_stores)
        if stor_list and isinstance(stor_list, list):
            for stor in stor_list:
                schema, endpoint, parsed_url = parse_url(stor.get("url"))
                new_stor = schema_to_store(schema)(self, schema, stor["name"], endpoint)
                new_stor.subpath = parsed_url.path
                new_stor.secret_pfx = stor.get("secret_pfx")
                new_stor.options = stor.get("options", {})
                new_stor.from_spec = True
                self._stores[stor["name"]] = new_stor

    def to_dict(self, struct):
        struct[run_keys.data_stores] = [
            stor.to_dict() for stor in self._stores.values() if stor.from_spec
        ]

    def secret(self, key):
        return self._secrets.get(key)

    def _add_store(self, store):
        self._stores[store.name] = store

    def get_store_artifact(self, url, project=""):

        try:
            resource = get_store_resource(
                url, db=self._get_db(), secrets=self._secrets, project=project
            )
        except Exception as exc:
            raise OSError(f"artifact {url} not found, {exc}")
        target = resource.get_target_path()
        if not target:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"resource {url} does not have a valid/persistent offline target"
            )
        return resource, target

    def object(self, url, key="", project="") -> DataItem:
        meta = artifact_url = None
        if is_store_uri(url):
            artifact_url = url
            meta, url = self.get_store_artifact(url, project)

        store, subpath = self.get_or_create_store(url)
        return DataItem(key, store, subpath, url, meta=meta, artifact_url=artifact_url)

    def get_or_create_store(self, url) -> (DataStore, str):
        schema, endpoint, parsed_url = parse_url(url)
        subpath = parsed_url.path

        if schema == "memory":
            subpath = url[len("memory://") :]
            return in_memory_store, subpath

        if not schema and endpoint:
            if endpoint in self._stores.keys():
                return self._stores[endpoint], subpath
            else:
                raise ValueError(f"no such store ({endpoint})")

        store_key = f"{schema}://{endpoint}"
        if store_key in self._stores.keys():
            return self._stores[store_key], subpath

        store = schema_to_store(schema)(self, schema, store_key, endpoint)
        self._stores[store_key] = store
        return store, subpath
