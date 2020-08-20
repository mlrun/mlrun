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
from .azure_blob import AzureBlobStore
from .base import DataItem, HttpStore
from .filestore import FileStore
from .inmem import InMemoryStore
from .s3 import S3Store
from .v3io import V3ioStore
from ..config import config
from ..utils import run_keys, DB_SCHEMA

in_memory_store = InMemoryStore()


def get_object_stat(url, secrets=None):
    stores = StoreManager(secrets)
    return stores.object(url=url).stat()


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
        endpoint += ":{}".format(parsed_url.port)
    return schema, endpoint, parsed_url


def schema_to_store(schema):
    if not schema or schema in ["file", "c", "d"]:
        return FileStore
    elif schema == "s3":
        return S3Store
    elif schema == "az":
        return AzureBlobStore
    elif schema in ["v3io", "v3ios"]:
        return V3ioStore
    elif schema in ["http", "https"]:
        return HttpStore
    else:
        raise ValueError("unsupported store scheme ({})".format(schema))


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
            self._db = mlrun.get_run_db().connect(self._secrets)
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
        schema, endpoint, parsed_url = parse_url(url)
        if not parsed_url.path:
            raise ValueError("store url without a path {}".format(url))
        key = parsed_url.path[1:]
        project = endpoint or project or config.default_project
        tag = parsed_url.fragment if parsed_url.fragment else ""
        iteration = None
        if ":" in key:
            if tag:
                raise ValueError("Tag can not given both with : and with #")
            idx = key.rfind(":")
            tag = key[idx + 1 :]
            key = key[:idx]
        if "/" in key:
            idx = key.rfind("/")
            try:
                iteration = int(key[idx + 1 :])
            except ValueError:
                raise ValueError(
                    "illegal store path {}, iteration must be integer value".format(url)
                )
            key = key[:idx]

        db = self._get_db()
        try:
            meta = db.read_artifact(key, tag=tag, iter=iteration, project=project)
            if meta.get("kind", "") == "link":
                # todo: support other link types (not just iter, move this to the db/api layer
                meta = self._get_db().read_artifact(
                    parsed_url.path[1:],
                    tag=tag,
                    iter=meta.get("link_iteration", 0),
                    project=project,
                )

            meta = mlrun.artifacts.dict_to_artifact(meta)
        except Exception as e:
            raise OSError("artifact {} not found, {}".format(url, e))
        return meta, meta.target_path

    def object(self, url, key="", project=""):
        meta = artifact_url = None
        if url.startswith(DB_SCHEMA + "://"):
            artifact_url = url
            meta, url = self.get_store_artifact(url, project)

        store, subpath = self.get_or_create_store(url)
        return DataItem(key, store, subpath, url, meta=meta, artifact_url=artifact_url)

    def get_or_create_store(self, url):
        schema, endpoint, parsed_url = parse_url(url)
        subpath = parsed_url.path

        if schema == "memory":
            subpath = url[len("memory://") :]
            return in_memory_store, subpath

        if not schema and endpoint:
            if endpoint in self._stores.keys():
                return self._stores[endpoint], subpath
            else:
                raise ValueError("no such store ({})".format(endpoint))

        storekey = "{}://{}".format(schema, endpoint)
        if storekey in self._stores.keys():
            return self._stores[storekey], subpath

        store = schema_to_store(schema)(self, schema, storekey, endpoint)
        self._stores[storekey] = store
        return store, subpath
