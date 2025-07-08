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

import warnings
from functools import partial
from typing import Optional

from mergedeep import merge

import mlrun
import mlrun.errors
from mlrun.artifacts.llm_prompt import LLMPromptArtifact
from mlrun.artifacts.model import ModelArtifact
from mlrun.datastore.datastore_profile import datastore_profile_read
from mlrun.datastore.model_provider.model_provider import (
    ModelProvider,
)
from mlrun.datastore.remote_client import BaseRemoteClient
from mlrun.datastore.utils import (
    parse_url,
)
from mlrun.errors import err_to_str
from mlrun.utils.helpers import get_local_file_schema

from ..artifacts.base import verify_target_path
from ..utils import DB_SCHEMA, RunKeys
from .base import DataItem, DataStore, HttpStore
from .filestore import FileStore
from .inmem import InMemoryStore
from .model_provider.openai_provider import OpenAIProvider
from .store_resources import get_store_resource, is_store_uri
from .v3io import V3ioStore

in_memory_store = InMemoryStore()


def schema_to_store(schema) -> DataStore.__subclasses__():
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
    raise ValueError(f"unsupported store scheme ({schema})")


def schema_to_model_provider(
    schema: str, raise_missing_schema_exception=True
) -> type[ModelProvider]:
    #  TODO add hugging face and http
    schema_dict = {"openai": OpenAIProvider}
    provider_class = schema_dict.get(schema, None)
    if not provider_class:
        if raise_missing_schema_exception:
            raise ValueError(f"unsupported model provider schema ({schema})")
        else:
            warnings.warn(f"unsupported model provider schema: {schema}")
    return provider_class


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
        self,
        url,
        project="",
        allow_empty_resources=None,
        secrets=None,
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
        if not allow_empty_resources:
            if isinstance(resource, LLMPromptArtifact):
                if not resource.spec.model_uri:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"LLMPromptArtifact {url} does not contain model artifact uri"
                    )
            elif not target and not (
                isinstance(resource, ModelArtifact) and resource.model_url
            ):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Resource {url} does not have a valid/persistent offline target or model_url"
                )
        return resource, target or ""

    def object(
        self,
        url,
        key="",
        project="",
        allow_empty_resources=None,
        secrets: Optional[dict] = None,
        **kwargs,
    ) -> DataItem:
        meta = artifact_url = None
        if is_store_uri(url):
            artifact_url = url
            meta, url = self.get_store_artifact(
                url, project, allow_empty_resources, secrets
            )
            if not allow_empty_resources:
                verify_target_path(meta)

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

    def _get_or_create_remote_client(
        self,
        url,
        secrets: Optional[dict] = None,
        project_name="",
        cache: Optional[dict] = None,
        schema_to_class: callable = schema_to_store,
        **kwargs,
    ) -> (BaseRemoteClient, str, str):
        # The cache can be an empty dictionary ({}), even if it is a _stores object
        cache = cache if cache is not None else {}
        schema, endpoint, parsed_url = parse_url(url)
        subpath = parsed_url.path
        cache_key = f"{schema}://{endpoint}" if endpoint else f"{schema}://"

        if schema == "ds":
            datastore_profile = datastore_profile_read(url, project_name, secrets)
            secrets = merge(secrets or {}, datastore_profile.secrets() or {})
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
            if endpoint in cache.keys():
                return cache[endpoint], subpath, url
            else:
                raise ValueError(f"no such store ({endpoint})")

        if not secrets and not mlrun.config.is_running_as_api():
            if cache_key in cache.keys():
                return cache[cache_key], subpath, url

        # support u/p embedding in url (as done in redis) by setting netloc as the "endpoint" parameter
        # when running on server we don't cache the datastore, because there are multiple users and we don't want to
        # cache the credentials, so for each new request we create a new store
        remote_client_class = schema_to_class(schema)
        remote_client = None
        if remote_client_class:
            endpoint, subpath = remote_client_class.parse_endpoint_and_path(
                endpoint, subpath
            )
            remote_client = remote_client_class(
                self, schema, cache_key, parsed_url.netloc, secrets=secrets, **kwargs
            )
            if not secrets and not mlrun.config.is_running_as_api():
                cache[cache_key] = remote_client
        else:
            warnings.warn("scheme not found. Returning None")
        return remote_client, subpath, url

    def get_or_create_store(
        self,
        url,
        secrets: Optional[dict] = None,
        project_name="",
    ) -> (DataStore, str, str):
        datastore, sub_path, url = self._get_or_create_remote_client(
            url=url,
            secrets=secrets,
            project_name=project_name,
            cache=self._stores,
            schema_to_class=schema_to_store,
        )
        if not isinstance(datastore, DataStore):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "remote client by url is not datastore"
            )
        return datastore, sub_path, url

    def get_or_create_model_provider(
        self,
        url,
        secrets: Optional[dict] = None,
        project_name="",
        default_invoke_kwargs: Optional[dict] = None,
        raise_missing_schema_exception=True,
    ) -> ModelProvider:
        schema_to_provider_with_raise = partial(
            schema_to_model_provider,
            raise_missing_schema_exception=raise_missing_schema_exception,
        )
        model_provider, _, _ = self._get_or_create_remote_client(
            url=url,
            secrets=secrets,
            project_name=project_name,
            schema_to_class=schema_to_provider_with_raise,
            default_invoke_kwargs=default_invoke_kwargs,
        )
        if model_provider and not isinstance(model_provider, ModelProvider):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "remote client by url is not model_provider"
            )
        return model_provider

    def reset_secrets(self):
        self._secrets = {}

    def model_provider_object(
        self,
        url,
        project="",
        allow_empty_resources=None,
        secrets: Optional[dict] = None,
        default_invoke_kwargs: Optional[dict] = None,
        raise_missing_schema_exception=True,
    ) -> ModelProvider:
        if mlrun.datastore.is_store_uri(url):
            resource = self.get_store_artifact(
                url,
                project,
                allow_empty_resources,
                secrets,
            )
            if not isinstance(resource, ModelArtifact) or not resource.model_url:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "unable to create the model provider from the given resource URI"
                )
            url = resource.model_url
            default_invoke_kwargs = default_invoke_kwargs or resource.default_config
        model_provider = self.get_or_create_model_provider(
            url,
            secrets=secrets,
            project_name=project,
            default_invoke_kwargs=default_invoke_kwargs,
            raise_missing_schema_exception=raise_missing_schema_exception,
        )
        return model_provider
