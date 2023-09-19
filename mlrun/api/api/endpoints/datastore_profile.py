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

from http import HTTPStatus

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun
import mlrun.api.api.deps
import mlrun.api.crud
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.singletons.db
import mlrun.common.schemas
from mlrun.api.api.utils import log_and_raise
from mlrun.datastore.datastore_profile import DatastoreProfile as ds

router = APIRouter()


@router.put(
    path="/projects/{project_name}/datastore-profiles",
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
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.datastore_profile,
        project_name,
        info.name,
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )
    # overwrite the project
    if info.project != project_name:
        log_and_raise(
            HTTPStatus.BAD_REQUEST.value,
            reason="The project name provided in the URI does not match the one specified in the DatastoreProfile",
        )
    project_ds_name_private = ds.generate_secret_key(info.name, project_name)

    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.secret,
        project_name,
        mlrun.common.schemas.SecretProviderName.kubernetes,
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )

    # TODO: Although embedding specialized business logic like the handling of sensitive information in secrets
    # directly within endpoint code is suboptimal, we currently lack a dedicated CRUD object for managing storage
    # profiles. Creating one solely for this purpose might introduce unnecessary overhead.
    # When new features are introduced that warrant a CRUD framework, the logic for handling storage
    # profiles should be relocated there.
    await run_in_threadpool(
        mlrun.api.crud.Secrets().store_project_secrets,
        project_name,
        mlrun.common.schemas.SecretsData(
            provider=mlrun.common.schemas.SecretProviderName.kubernetes,
            secrets={project_ds_name_private: info.private},
        ),
        True,
    )

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
    path="/projects/{project_name}/datastore-profiles",
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
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project_name,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    profiles = await run_in_threadpool(
        mlrun.api.utils.singletons.db.get_db().list_datastore_profiles,
        db_session,
        project_name,
    )
    if len(profiles) == 0:
        return profiles
    filtered_data = await mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.datastore_profile,
        profiles,
        lambda profile: (project_name, profile.name),
        auth_info,
    )
    return filtered_data


@router.get(
    path="/projects/{project_name}/datastore-profiles/{profile}",
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
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.datastore_profile,
        project_name,
        profile,
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
    path="/projects/{project_name}/datastore-profiles/{profile}",
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
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.datastore_profile,
        project_name,
        profile,
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.secret,
        project_name,
        mlrun.common.schemas.SecretProviderName.kubernetes,
        mlrun.common.schemas.AuthorizationAction.delete,
        auth_info,
    )
    project_ds_name_private = ds.generate_secret_key(profile, project_name)

    await run_in_threadpool(
        mlrun.api.crud.Secrets().delete_project_secret,
        project_name,
        mlrun.common.schemas.SecretProviderName.kubernetes,
        project_ds_name_private,
        allow_secrets_from_k8s=True,
        allow_internal_secrets=True,
    )

    return await run_in_threadpool(
        mlrun.api.utils.singletons.db.get_db().delete_datastore_profile,
        db_session,
        profile,
        project_name,
    )
