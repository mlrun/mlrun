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
import http

import fastapi
import fastapi.concurrency
import sqlalchemy.orm

import mlrun.api.api.deps
import mlrun.api.crud.tags
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.singletons.project_member
from mlrun.utils.helpers import tag_name_regex_as_string

router = fastapi.APIRouter()


@router.post("/projects/{project}/tags/{tag}", response_model=mlrun.api.schemas.Tag)
async def overwrite_object_tags_with_tag(
    project: str,
    tag: str = fastapi.Path(..., regex=tag_name_regex_as_string()),
    tag_objects: mlrun.api.schemas.TagObjects = fastapi.Body(...),
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )

    # check permission per object type
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        getattr(mlrun.api.schemas.AuthorizationResourceTypes, tag_objects.kind),
        project,
        resource_name="",
        # not actually overwriting objects, just overwriting the objects tags
        action=mlrun.api.schemas.AuthorizationAction.update,
        auth_info=auth_info,
    )

    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.crud.Tags().overwrite_object_tags_with_tag,
        db_session,
        project,
        tag,
        tag_objects,
    )
    return mlrun.api.schemas.Tag(name=tag, project=project)


@router.put("/projects/{project}/tags/{tag}", response_model=mlrun.api.schemas.Tag)
async def append_tag_to_objects(
    project: str,
    tag: str = fastapi.Path(..., regex=tag_name_regex_as_string()),
    tag_objects: mlrun.api.schemas.TagObjects = fastapi.Body(...),
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )

    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        getattr(mlrun.api.schemas.AuthorizationResourceTypes, tag_objects.kind),
        project,
        resource_name="",
        action=mlrun.api.schemas.AuthorizationAction.update,
        auth_info=auth_info,
    )

    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.crud.Tags().append_tag_to_objects,
        db_session,
        project,
        tag,
        tag_objects,
    )
    return mlrun.api.schemas.Tag(name=tag, project=project)


@router.delete(
    "/projects/{project}/tags/{tag}", status_code=http.HTTPStatus.NO_CONTENT.value
)
async def delete_tag_from_objects(
    project: str,
    tag: str,
    tag_objects: mlrun.api.schemas.TagObjects,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.utils.singletons.project_member.get_project_member().ensure_project,
        db_session,
        project,
        auth_info=auth_info,
    )

    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        getattr(mlrun.api.schemas.AuthorizationResourceTypes, tag_objects.kind),
        project,
        resource_name="",
        # not actually deleting objects, just deleting the objects tags
        action=mlrun.api.schemas.AuthorizationAction.update,
        auth_info=auth_info,
    )

    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.crud.Tags().delete_tag_from_objects,
        db_session,
        project,
        tag,
        tag_objects,
    )
    return fastapi.Response(status_code=http.HTTPStatus.NO_CONTENT.value)
