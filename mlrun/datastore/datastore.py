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

from mergedeep import merge

import mlrun
import mlrun.errors
from mlrun.datastore.datastore_profile import datastore_profile_read
from mlrun.errors import err_to_str
from mlrun.utils.helpers import get_local_file_schema

from ..utils import DB_SCHEMA, RunKeys
from .base import DataItem, DataStore, HttpStore
from .filestore import FileStore
from .inmem import InMemoryStore
from .store_resources import get_store_resource, is_store_uri
from .v3io import V3ioStore

in_memory_store = InMemoryStore()


def parse_url(url):
    if url and url.startswith("v3io://") and not url.startswith("v3io:///"):
        url = url.replace("v3io://", "v3io:///", 1)
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

    if not schema or schema in get_local_file_schema():
        return FileStore
    elif schema == "s3":
        try:
            from .s3 import S3Store
        except ImportError:
            raise mlrun.errors.MLRunMissingDependencyError(
                "s3 packages are missing, use pip install mlrun[s3]"
            )

        return S3Store
    elif schema in ["az", "wasbs", "wasb"]:
        try:
            from .azure_blob import AzureBlobStore
        except ImportError:
            raise mlrun.errors.MLRunMissingDependencyError(
                "azure blob storage packages are missing, use pip install mlrun[azure-blob-storage]"
            )

        return AzureBlobStore
    elif schema in ["v3io", "v3ios"]:
        return V3ioStore
    elif schema in ["redis", "rediss"]:
        from .redis import RedisStore

        return RedisStore
    elif schema in ["http", "https"]:
        return HttpStore
    elif schema in ["gcs", "gs"]:
        try:
            from .google_cloud_storage import GoogleCloudStorageStore
        except ImportError:
            raise mlrun.errors.MLRunMissingDependencyError(
                "Google cloud storage packages are missing, use pip install mlrun[google-cloud-storage]"
            )
        return GoogleCloudStorageStore
    elif schema == "dbfs":
        from .dbfs_store import DBFSStore

        return DBFSStore
    elif schema in ["hdfs", "webhdfs"]:
        from .hdfs import HdfsStore

        return HdfsStore
    elif schema == "oss":
        from .alibaba_oss import OSSStore

        return OSSStore
    else:
        raise ValueError(f"unsupported store scheme ({schema})")


def uri_to_ipython(link):
    schema, endpoint, parsed_url = parse_url(link)
    if schema in [DB_SCHEMA, "memory", "ds"]:
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
        stor_list = struct.get(RunKeys.data_stores)
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
        struct[RunKeys.data_stores] = [
            stor.to_dict() for stor in self._stores.values() if stor.from_spec
        ]

    def secret(self, key):
        return self._secrets.get(key)

    def _add_store(self, store):
        self._stores[store.name] = store

    def get_store_artifact(
        self, url, project="", allow_empty_resources=None, secrets=None
    ):
        """
        This is expected to be run only on client side. server is not expected to load artifacts.
        """
        try:
            resource = get_store_resource(
                url,
                db=self._get_db(),
                secrets=self._secrets,
                project=project,
                data_store_secrets=secrets,
            )
        except Exception as exc:
            raise OSError(f"artifact {url} not found, {err_to_str(exc)}")
        target = resource.get_target_path()
        # the allow_empty.. flag allows us to have functions which dont depend on having targets e.g. a function
        # which accepts a feature vector uri and generate the offline vector (parquet) for it if it doesnt exist
        if not target and not allow_empty_resources:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Resource {url} does not have a valid/persistent offline target"
            )
        return resource, target or ""

    def object(
        self,
        url,
        key="",
        project="",
        allow_empty_resources=None,
        secrets: Optional[dict] = None,
    ) -> DataItem:
        meta = artifact_url = None
        if is_store_uri(url):
            artifact_url = url
            meta, url = self.get_store_artifact(
                url, project, allow_empty_resources, secrets
            )

        store, subpath, url = self.get_or_create_store(
            url, secrets=secrets, project_name=project
        )
        return DataItem(
            key,
            store,
            subpath,
            url,
            meta=meta,
            artifact_url=artifact_url,
        )

    def get_or_create_store(
        self, url, secrets: Optional[dict] = None, project_name=""
    ) -> (DataStore, str, str):
        schema, endpoint, parsed_url = parse_url(url)
        subpath = parsed_url.path
        store_key = f"{schema}://{endpoint}" if endpoint else f"{schema}://"

        if schema == "ds":
            datastore_profile = datastore_profile_read(url, project_name, secrets)
            if secrets and datastore_profile.secrets():
                secrets = merge(secrets, datastore_profile.secrets())
            else:
                secrets = secrets or datastore_profile.secrets()
            url = datastore_profile.url(subpath)
            schema, endpoint, parsed_url = parse_url(url)
            subpath = parsed_url.path

        if schema == "memory":
            subpath = url[len("memory://") :]
            return in_memory_store, subpath, url

        elif schema in get_local_file_schema():
            # parse_url() will drop the windows drive-letter from the path for url like "c:\a\b".
            # As a workaround, we set subpath to the url.
            subpath = url.replace("file://", "", 1)

        if not schema and endpoint:
            if endpoint in self._stores.keys():
                return self._stores[endpoint], subpath, url
            else:
                raise ValueError(f"no such store ({endpoint})")

        if not secrets and not mlrun.config.is_running_as_api():
            if store_key in self._stores.keys():
                return self._stores[store_key], subpath, url

        # support u/p embedding in url (as done in redis) by setting netloc as the "endpoint" parameter
        # when running on server we don't cache the datastore, because there are multiple users and we don't want to
        # cache the credentials, so for each new request we create a new store
        store = schema_to_store(schema)(
            self, schema, store_key, parsed_url.netloc, secrets=secrets
        )
        if not secrets and not mlrun.config.is_running_as_api():
            self._stores[store_key] = store
        return store, subpath, url

    def reset_secrets(self):
        self._secrets = {}
