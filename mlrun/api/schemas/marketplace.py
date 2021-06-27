from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Extra, Field

from mlrun.api.schemas import ObjectKind, ObjectSpec, ObjectStatus
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


# Sources-related objects
class MarketplaceSourceSpec(ObjectSpec):
    path: str  # URL to base directory, should include schema (s3://, etc...)
    credentials: Optional[dict] = None


class MarketplaceSource(BaseModel):
    kind: ObjectKind = Field(ObjectKind.marketplace_source, const=True)
    metadata: MarketplaceObjectMetadata
    spec: MarketplaceSourceSpec
    status: ObjectStatus

    @classmethod
    def create_default_source(cls):
        if not config.marketplace_default_source.create:
            return None

        now = datetime.now(timezone.utc)
        hub_metadata = MarketplaceObjectMetadata(
            name=config.marketplace_default_source.name,
            description=config.marketplace_default_source.description,
            created=now,
            updated=now,
        )
        return cls(
            metadata=hub_metadata,
            spec=MarketplaceSourceSpec(path=config.marketplace_default_source.url),
            status=ObjectStatus(state="created"),
        )


class OrderedMarketplaceSource(BaseModel):
    order: int = -1  # Default last. Otherwise must be > 0
    source: MarketplaceSource


# Item-related objects
class MarketplaceItemMetadata(MarketplaceObjectMetadata):
    source: str
    channel: str


class MarketplaceItemSpec(ObjectSpec):
    item_uri: str


class MarketplaceItem(BaseModel):
    kind: ObjectKind = Field(ObjectKind.marketplace_item, const=True)
    metadata: MarketplaceObjectMetadata
    spec: MarketplaceItemSpec
    status: ObjectStatus


class MarketplaceCatalog(BaseModel):
    kind: ObjectKind = Field(ObjectKind.marketplace_catalog, const=True)
    catalog: List[MarketplaceItem]
