import enum
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Extra, Field

import mlrun.errors
from mlrun.api.schemas.object import ObjectKind, ObjectSpec, ObjectStatus
from mlrun.config import config


# Defining a different base class (not ObjectMetadata), as there's no project and it differs enough to
# justify a new class
class MarketplaceObjectMetadata(BaseModel):
    name: str
    description: str = ""
    labels: Optional[dict]
    updated: Optional[datetime]
    created: Optional[datetime]

    class Config:
        extra = Extra.allow


# Currently only functions are supported. Will add more in the future.
class MarketplaceSourceType(str, enum.Enum):
    functions = "functions"


# Sources-related objects
class MarketplaceSourceSpec(ObjectSpec):
    path: str  # URL to base directory, should include schema (s3://, etc...)
    channel: str
    credentials: Optional[dict] = None


class MarketplaceSource(BaseModel):
    kind: ObjectKind = Field(ObjectKind.marketplace_source, const=True)
    metadata: MarketplaceObjectMetadata
    spec: MarketplaceSourceSpec
    status: Optional[ObjectStatus] = ObjectStatus(state="created")

    def get_full_uri(self, relative_path):
        return "{base}/{channel}/{relative_path}".format(
            base=self.spec.path, channel=self.spec.channel, relative_path=relative_path
        )

    def get_catalog_uri(self):
        return self.get_full_uri(config.marketplace.catalog_filename)

    @classmethod
    def generate_default_source(cls):
        if not config.marketplace.default_source.create:
            return None

        now = datetime.now(timezone.utc)
        hub_metadata = MarketplaceObjectMetadata(
            name=config.marketplace.default_source.name,
            description=config.marketplace.default_source.description,
            created=now,
            updated=now,
        )
        return cls(
            metadata=hub_metadata,
            spec=MarketplaceSourceSpec(
                path=config.marketplace.default_source.url,
                channel=config.marketplace.default_source.channel,
            ),
            status=ObjectStatus(state="created"),
        )


last_source_index = -1


class IndexedMarketplaceSource(BaseModel):
    index: int = last_source_index  # Default last. Otherwise must be > 0
    source: MarketplaceSource


# Item-related objects
class MarketplaceItemMetadata(MarketplaceObjectMetadata):
    source: MarketplaceSourceType = Field(MarketplaceSourceType.functions, const=True)
    channel: str
    version: str
    tag: Optional[str]

    def get_relative_path(self) -> str:
        if self.source == MarketplaceSourceType.functions:
            # This is needed since the marketplace deployment script modifies the paths to use _ instead of -.
            modified_name = self.name.replace("-", "_")
            # Prefer using the tag if exists. Otherwise use version.
            version = self.tag or self.version
            return f"{self.source.value}/{self.channel}/{modified_name}/{version}/"
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Bad source for marketplace item - {self.source}"
            )


class MarketplaceItemSpec(ObjectSpec):
    item_uri: str


class MarketplaceItem(BaseModel):
    kind: ObjectKind = Field(ObjectKind.marketplace_item, const=True)
    metadata: MarketplaceItemMetadata
    spec: MarketplaceItemSpec
    status: ObjectStatus


class MarketplaceCatalog(BaseModel):
    kind: ObjectKind = Field(ObjectKind.marketplace_catalog, const=True)
    catalog: List[MarketplaceItem]
