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
#
import json

import mlrun.errors
import mlrun.utils.singleton
from mlrun.api.schemas.marketplace import (
    MarketplaceCatalog,
    MarketplaceItem,
    MarketplaceItemMetadata,
    MarketplaceItemSpec,
    MarketplaceSource,
    ObjectStatus,
)
from mlrun.api.utils.singletons.k8s import get_k8s
from mlrun.config import config
from mlrun.datastore import store_manager

from ..schemas import SecretProviderName
from .secrets import Secrets, SecretsClientType

# Using a complex separator, as it's less likely someone will use it in a real secret name
secret_name_separator = "-__-"


class Marketplace(metaclass=mlrun.utils.singleton.Singleton):
    def __init__(self):
        self._internal_project_name = config.marketplace.k8s_secrets_project_name
        self._catalogs = {}

    @staticmethod
    def _in_k8s():
        k8s_helper = get_k8s()
        return (
            k8s_helper is not None and k8s_helper.is_running_inside_kubernetes_cluster()
        )

    @staticmethod
    def _generate_credentials_secret_key(source, key=""):
        full_key = source + secret_name_separator + key
        return Secrets().generate_client_project_secret_key(
            SecretsClientType.marketplace, full_key
        )

    def add_source(self, source: MarketplaceSource):
        source_name = source.metadata.name
        credentials = source.spec.credentials
        if credentials:
            self._store_source_credentials(source_name, credentials)

    def remove_source(self, source_name):
        self._catalogs.pop(source_name, None)
        if not self._in_k8s():
            return

        source_credentials = self._get_source_credentials(source_name)
        if not source_credentials:
            return
        secrets_to_delete = [
            self._generate_credentials_secret_key(source_name, key)
            for key in source_credentials
        ]
        Secrets().delete_project_secrets(
            self._internal_project_name,
            SecretProviderName.kubernetes,
            secrets_to_delete,
            allow_internal_secrets=True,
        )

    def _store_source_credentials(self, source_name, credentials: dict):
        if not self._in_k8s():
            raise mlrun.errors.MLRunInvalidArgumentError(
                "MLRun is not configured with k8s, marketplace source credentials cannot be stored securely"
            )

        adjusted_credentials = {
            self._generate_credentials_secret_key(source_name, key): value
            for key, value in credentials.items()
        }
        Secrets().store_project_secrets(
            self._internal_project_name,
            mlrun.api.schemas.SecretsData(
                provider=SecretProviderName.kubernetes, secrets=adjusted_credentials
            ),
            allow_internal_secrets=True,
        )

    def _get_source_credentials(self, source_name):
        if not self._in_k8s():
            return {}

        secret_prefix = self._generate_credentials_secret_key(source_name)
        secrets = (
            Secrets()
            .list_project_secrets(
                self._internal_project_name,
                SecretProviderName.kubernetes,
                allow_secrets_from_k8s=True,
                allow_internal_secrets=True,
            )
            .secrets
        )

        source_secrets = {}
        for key, value in secrets.items():
            if key.startswith(secret_prefix):
                source_secrets[key[len(secret_prefix) :]] = value

        return source_secrets

    @staticmethod
    def _transform_catalog_dict_to_schema(source, catalog_dict):
        catalog_dict = catalog_dict.get("functions")
        if not catalog_dict:
            raise mlrun.errors.MLRunInternalServerError(
                "Invalid catalog file - no 'functions' section found."
            )

        catalog = MarketplaceCatalog(catalog=[])
        # Loop over channels, then per function extract versions.
        for channel_name in catalog_dict:
            channel_dict = catalog_dict[channel_name]
            for function_name in channel_dict:
                function_dict = channel_dict[function_name]
                for version_tag in function_dict:
                    version_dict = function_dict[version_tag]
                    function_details_dict = version_dict.copy()
                    spec_dict = function_details_dict.pop("spec", None)
                    metadata = MarketplaceItemMetadata(
                        channel=channel_name, tag=version_tag, **function_details_dict
                    )
                    item_uri = source.get_full_uri(metadata.get_relative_path())
                    spec = MarketplaceItemSpec(item_uri=item_uri, **spec_dict)
                    item = MarketplaceItem(
                        metadata=metadata, spec=spec, status=ObjectStatus()
                    )
                    catalog.catalog.append(item)

        return catalog

    def get_source_catalog(
        self,
        source: MarketplaceSource,
        channel=None,
        version=None,
        tag=None,
        force_refresh=False,
    ) -> MarketplaceCatalog:
        source_name = source.metadata.name
        if not self._catalogs.get(source_name) or force_refresh:
            url = source.get_catalog_uri()
            credentials = self._get_source_credentials(source_name)
            catalog_data = mlrun.run.get_object(url=url, secrets=credentials)
            catalog_dict = json.loads(catalog_data)
            catalog = self._transform_catalog_dict_to_schema(source, catalog_dict)
            self._catalogs[source_name] = catalog
        else:
            catalog = self._catalogs[source_name]

        result_catalog = MarketplaceCatalog(catalog=[])
        for item in catalog.catalog:
            if (
                (channel is None or item.metadata.channel == channel)
                and (tag is None or item.metadata.tag == tag)
                and (version is None or item.metadata.version == version)
            ):
                result_catalog.catalog.append(item)

        return result_catalog

    def get_item(
        self,
        source: MarketplaceSource,
        item_name,
        channel,
        version=None,
        tag=None,
        force_refresh=False,
    ) -> MarketplaceItem:
        catalog = self.get_source_catalog(source, channel, version, tag, force_refresh)
        items = [item for item in catalog.catalog if item.metadata.name == item_name]
        if not items:
            raise mlrun.errors.MLRunNotFoundError(
                f"Item not found. source={item_name}, channel={channel}, version={version}"
            )
        if len(items) > 1:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Query resulted in more than 1 catalog items. "
                + f"source={item_name}, channel={channel}, version={version}, tag={tag}"
            )
        return items[0]

    def get_item_object_using_source_credentials(self, source: MarketplaceSource, url):
        credentials = self._get_source_credentials(source.metadata.name)

        if not url.startswith(source.spec.path):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "URL to retrieve must be located in the source filesystem tree"
            )

        if url.endswith("/"):
            obj = store_manager.object(url=url, secrets=credentials)
            listdir = obj.listdir()
            return {
                "listdir": listdir,
            }
        else:
            catalog_data = mlrun.run.get_object(url=url, secrets=credentials)
        return catalog_data
