import json

import mlrun.api.api.utils
import mlrun.api.schemas
import mlrun.api.utils.projects.remotes.follower
import mlrun.api.utils.singletons.db
import mlrun.errors
import mlrun.runtimes
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

# Using double underscore, as it's less likely someone will use it in a real secret name
secret_name_separator = "__"


class MarketplaceItemsManager(metaclass=mlrun.utils.singleton.Singleton):
    def __init__(self):
        self._internal_project_name = config.marketplace.k8s_secrets_project_name
        self._catalogs = {}

    def add_source(self, source: MarketplaceSource):
        source_name = source.metadata.name
        credentials = source.spec.credentials
        if credentials:
            self._store_source_credentials(source_name, credentials)

    def remove_source(self, source_name):
        secret_prefix = source_name + secret_name_separator
        source_credentials = self._get_source_credentials(source_name)
        if not source_credentials:
            return
        secrets_to_delete = [secret_prefix + key for key in source_credentials]
        get_k8s().delete_project_secrets(self._internal_project_name, secrets_to_delete)
        self._catalogs.pop(source_name, None)

    def _store_source_credentials(self, source_name, credentials: dict):
        if not get_k8s():
            raise mlrun.errors.MLRunInvalidArgumentError(
                "MLRun is not configured with k8s, marketplace source credentials cannot be stored securely"
            )
        secret_prefix = source_name + secret_name_separator

        adjusted_credentials = {
            secret_prefix + key: value for key, value in credentials.items()
        }
        get_k8s().store_project_secrets(
            self._internal_project_name, adjusted_credentials
        )

    def _get_source_credentials(self, source_name):
        if not get_k8s():
            return {}

        secret_prefix = source_name + secret_name_separator
        secrets = get_k8s().get_project_secret_values(self._internal_project_name)
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
                    metadata_dict = version_dict.copy()
                    spec_dict = metadata_dict.pop("spec", None)
                    metadata = MarketplaceItemMetadata(
                        channel=channel_name, **metadata_dict
                    )
                    item_uri = source.get_full_uri(metadata.get_relative_path())
                    spec = MarketplaceItemSpec(item_uri=item_uri, **spec_dict)
                    item = MarketplaceItem(
                        metadata=metadata, spec=spec, status=ObjectStatus()
                    )
                    catalog.catalog.append(item)

        return catalog

    def get_source_catalog(
        self, source: MarketplaceSource, channel=None, version=None, force_refresh=False
    ) -> MarketplaceCatalog:
        source_name = source.metadata.name
        if not self._catalogs.get(source_name) or force_refresh:
            url = source.get_full_uri(config.marketplace.catalog_filename)
            credentials = self._get_source_credentials(source_name)
            catalog_data = mlrun.run.get_object(url=url, secrets=credentials)
            catalog_dict = json.loads(catalog_data)
            catalog = self._transform_catalog_dict_to_schema(source, catalog_dict)
            self._catalogs[source_name] = catalog
        else:
            catalog = self._catalogs[source_name]

        result_catalog = MarketplaceCatalog(catalog=[])
        for item in catalog.catalog:
            if (channel is None or item.metadata.channel == channel) and (
                version is None or item.metadata.version == version
            ):
                result_catalog.catalog.append(item)

        return result_catalog

    def get_item(
        self,
        source: MarketplaceSource,
        item_name,
        channel,
        version,
        force_refresh=False,
    ) -> MarketplaceItem:
        catalog = self.get_source_catalog(source, channel, version, force_refresh)
        items = [item for item in catalog.catalog if item.metadata.name == item_name]
        if not items:
            raise mlrun.errors.MLRunNotFoundError(
                f"Item not found. source={item_name}, channel={channel}, version={version}"
            )
        if len(items) > 1:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Query resulted in more than 1 catalog items. source={item_name}, channel={channel}, version={version}"
            )
        return items[0]

    def get_item_object(self, source: MarketplaceSource, item: MarketplaceItem):
        credentials = self._get_source_credentials(source.metadata.name)
        catalog_data = mlrun.run.get_object(url=item.spec.item_uri, secrets=credentials)
        return catalog_data
