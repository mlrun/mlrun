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
#


from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun
import mlrun.api.api.deps
import mlrun.api.crud
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.singletons.db
import mlrun.common.schemas

router = APIRouter()


@router.put(
    path="/projects/{project_name}/datastore_profiles",
)
async def store_datastore_profile(
    project_name: str,
    info: mlrun.common.schemas.DatastoreProfile,
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().get_project,
        db_session,
        project_name,
        auth_info.session,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.datastore_profile,
        mlrun.common.schemas.AuthorizationAction.create,
        auth_info,
    )
    # overwrite the project
    info.project = project_name
    await run_in_threadpool(
        mlrun.api.utils.singletons.db.get_db().store_datastore_profile,
        db_session,
        info,
    )

    return await run_in_threadpool(
        mlrun.api.utils.singletons.db.get_db().get_datastore_profile,
        db_session,
        info.name,
        project_name,
    )


@router.get(
    path="/projects/{project_name}/datastore_profiles",
)
async def list_datastore_profiles(
    project_name: str,
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().get_project,
        db_session,
        project_name,
        auth_info.session,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.datastore_profile,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    return await run_in_threadpool(
        mlrun.api.utils.singletons.db.get_db().list_datastore_profiles,
        db_session,
        project_name,
    )


@router.get(
    path="/projects/{project_name}/datastore_profiles/{profile}",
)
async def get_datastore_profile(
    project_name: str,
    profile: str,
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().get_project,
        db_session,
        project_name,
        auth_info.session,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.datastore_profile,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    return await run_in_threadpool(
        mlrun.api.utils.singletons.db.get_db().get_datastore_profile,
        db_session,
        profile,
        project_name,
    )


@router.delete(
    path="/projects/{project_name}/datastore_profiles/{profile}",
)
async def delete_datastore_profile(
    project_name: str,
    profile: str,
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
    auth_info: mlrun.common.schemas.AuthInfo = Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    await run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().get_project,
        db_session,
        project_name,
        auth_info.session,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_global_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.datastore_profile,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    return await run_in_threadpool(
        mlrun.api.utils.singletons.db.get_db().delete_datastore_profile,
        db_session,
        profile,
        project_name,
    )
