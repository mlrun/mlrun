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

import mimetypes
from http import HTTPStatus
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun
import mlrun.api.api.deps
import mlrun.api.crud
import mlrun.api.utils.auth.verifier
from mlrun.api.schemas import AuthorizationAction
from mlrun.api.schemas.hub import HubCatalog, HubItem, IndexedHubSource
from mlrun.api.utils.singletons.db import get_db

router = APIRouter()


@router.post(
    path="/hub/sources",
    status_code=HTTPStatus.CREATED.value,
    response_model=IndexedHubSource,
)
async def create_source(
    source: IndexedHubSource,
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.hub_source,
        AuthorizationAction.create,
        auth_info,
    )

    await run_in_threadpool(get_db().create_hub_source, db_session, source)
    # Handle credentials if they exist
    await run_in_threadpool(mlrun.api.crud.Hub().add_source, source.source)
    return await run_in_threadpool(
        get_db().get_hub_source, db_session, source.source.metadata.name
    )


@router.get(
    path="/hub/sources",
    response_model=List[IndexedHubSource],
)
async def list_sources(
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.hub_source,
        AuthorizationAction.read,
        auth_info,
    )

    return await run_in_threadpool(get_db().list_hub_sources, db_session)


@router.delete(
    path="/hub/sources/{source_name}",
    status_code=HTTPStatus.NO_CONTENT.value,
)
async def delete_source(
    source_name: str,
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.hub_source,
        AuthorizationAction.delete,
        auth_info,
    )

    await run_in_threadpool(get_db().delete_hub_source, db_session, source_name)
    await run_in_threadpool(mlrun.api.crud.Hub().remove_source, source_name)


@router.get(
    path="/hub/sources/{source_name}",
    response_model=IndexedHubSource,
)
async def get_source(
    source_name: str,
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    hub_source = await run_in_threadpool(
        get_db().get_hub_source, db_session, source_name
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.hub_source,
        AuthorizationAction.read,
        auth_info,
    )

    return hub_source


@router.put(path="/hub/sources/{source_name}", response_model=IndexedHubSource)
async def store_source(
    source_name: str,
    source: IndexedHubSource,
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.hub_source,
        AuthorizationAction.store,
        auth_info,
    )

    await run_in_threadpool(get_db().store_hub_source, db_session, source_name, source)
    # Handle credentials if they exist
    await run_in_threadpool(mlrun.api.crud.Hub().add_source, source.source)

    return await run_in_threadpool(get_db().get_hub_source, db_session, source_name)


@router.get(
    path="/hub/sources/{source_name}/items",
    response_model=HubCatalog,
)
async def get_catalog(
    source_name: str,
    version: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    force_refresh: Optional[bool] = Query(False, alias="force-refresh"),
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    ordered_source = await run_in_threadpool(
        get_db().get_hub_source, db_session, source_name
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.hub_source,
        AuthorizationAction.read,
        auth_info,
    )

    return await run_in_threadpool(
        mlrun.api.crud.Hub().get_source_catalog,
        ordered_source.source,
        version,
        tag,
        force_refresh,
    )


@router.get(
    "/hub/sources/{source_name}/items/{item_name}",
    response_model=HubItem,
)
async def get_item(
    source_name: str,
    item_name: str,
    version: Optional[str] = Query(None),
    tag: Optional[str] = Query("latest"),
    force_refresh: Optional[bool] = Query(False, alias="force-refresh"),
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    ordered_source = await run_in_threadpool(
        get_db().get_hub_source, db_session, source_name
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.hub_source,
        AuthorizationAction.read,
        auth_info,
    )

    return await run_in_threadpool(
        mlrun.api.crud.Hub().get_item,
        ordered_source.source,
        item_name,
        version,
        tag,
        force_refresh,
    )


@router.get(
    "/hub/sources/{source_name}/item-object",
)
async def get_object(
    source_name: str,
    url: str,
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
    auth_info: mlrun.api.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    ordered_source = await run_in_threadpool(
        get_db().get_hub_source, db_session, source_name
    )
    object_data = await run_in_threadpool(
        mlrun.api.crud.Hub().get_item_object_using_source_credentials,
        ordered_source.source,
        url,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.hub_source,
        AuthorizationAction.read,
        auth_info,
    )

    if url.endswith("/"):
        return object_data

    ctype, _ = mimetypes.guess_type(url)
    if not ctype:
        ctype = "application/octet-stream"
    return Response(content=object_data, media_type=ctype)
