from http import HTTPStatus
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

import mlrun
from mlrun.api.crud.function_marketplace import MarketplaceItemsManager
from mlrun.api.schemas.marketplace import (
    MarketplaceCatalog,
    MarketplaceItem,
    OrderedMarketplaceSource,
)
from mlrun.api.utils.singletons.db import get_db

router = APIRouter()


@router.post(
    path="/marketplace/sources",
    status_code=HTTPStatus.CREATED.value,
    response_model=OrderedMarketplaceSource,
)
def add_source(
    source: OrderedMarketplaceSource,
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
):
    get_db().create_marketplace_source(db_session, source)
    # Handle credentials if they exist
    MarketplaceItemsManager().add_source(source.source)
    return get_db().get_marketplace_source(db_session, source.source.metadata.name)


@router.get(
    path="/marketplace/sources", response_model=List[OrderedMarketplaceSource],
)
def list_sources(db_session: Session = Depends(mlrun.api.api.deps.get_db_session),):
    return get_db().list_marketplace_sources(db_session)


@router.delete(
    path="/marketplace/sources/{source_name}", status_code=HTTPStatus.NO_CONTENT.value,
)
def delete_source(
    source_name: str, db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
):
    get_db().delete_marketplace_source(db_session, source_name)
    MarketplaceItemsManager().remove_source(source_name)


@router.get(
    path="/marketplace/sources/{source_name}", response_model=OrderedMarketplaceSource,
)
def get_source(
    source_name: str, db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
):
    return get_db().get_marketplace_source(db_session, source_name)


@router.put(
    path="/marketplace/sources/{source_name}", response_model=OrderedMarketplaceSource
)
def store_source(
    source_name: str,
    source: OrderedMarketplaceSource,
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
):
    get_db().store_marketplace_source(db_session, source_name, source)
    # Handle credentials if they exist
    MarketplaceItemsManager().add_source(source.source)

    return get_db().get_marketplace_source(db_session, source_name)


@router.get(
    path="/marketplace/sources/{source_name}/items", response_model=MarketplaceCatalog,
)
def get_catalog(
    source_name: str,
    channel: Optional[str] = Query(None),
    version: Optional[str] = Query(None),
    force_refresh: Optional[bool] = Query(False, alias="force-refresh"),
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
):
    ordered_source = get_db().get_marketplace_source(db_session, source_name)
    return MarketplaceItemsManager().get_source_catalog(
        ordered_source.source, channel, version, force_refresh
    )


@router.get(
    "/marketplace/sources/{source_name}/items/{item_name}",
    response_model=MarketplaceItem,
)
def get_item(
    source_name: str,
    item_name: str,
    channel: Optional[str] = Query("development"),
    version: Optional[str] = Query("latest"),
    force_refresh: Optional[bool] = Query(False, alias="force-refresh"),
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
):
    ordered_source = get_db().get_marketplace_source(db_session, source_name)
    return MarketplaceItemsManager().get_item(
        ordered_source.source, item_name, channel, version, force_refresh
    )
