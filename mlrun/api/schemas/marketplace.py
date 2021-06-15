from typing import Optional, List

from pydantic import BaseModel, Field

from mlrun.api.schemas import ObjectKind, ObjectSpec, ObjectStatus, ObjectMetadata


class MarketplaceSourceMetadata(ObjectMetadata):
    credentials: Optional[dict] = None


class MarketplaceSourceSpec(ObjectSpec):
    name: str  # Unique source name
    path: str  # URL to base directory, should include schema (s3://, etc...)
    description: str = ""
    order: int = -1  # Default last


class MarketplaceSource(BaseModel):
    kind: ObjectKind = Field(ObjectKind.marketplace_source, const=True)
    metadata: MarketplaceSourceMetadata
    spec: MarketplaceSourceSpec
    status: ObjectStatus


class MarketplaceItemMetadata(ObjectMetadata):
    source: str
    channel: str


class MarketplaceItemSpec(ObjectSpec):
    item_uri: str


class MarketplaceItem(BaseModel):
    kind: ObjectKind = Field(ObjectKind.marketplace_item, const=True)
    metadata: ObjectMetadata
    spec: MarketplaceItemSpec
    status: ObjectStatus


class MarketplaceCatalog(BaseModel):
    kind: ObjectKind = Field(ObjectKind.marketplace_catalog, const=True)
    catalog: List[MarketplaceItem]
