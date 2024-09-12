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
import mimetypes
from http import HTTPStatus

import fastapi
from fastapi.concurrency import run_in_threadpool

import mlrun
import mlrun.common.schemas
import server.api.api.deps
import server.api.crud
import server.api.utils.auth.verifier
import server.api.utils.singletons.k8s
from mlrun.datastore import store_manager
from mlrun.errors import err_to_str
from mlrun.utils import logger
from server.api.api.utils import get_obj_path, get_secrets, log_and_raise

router = fastapi.APIRouter()


@router.get("/projects/{project}/files")
async def get_files_with_project_secrets(
    project: str,
    schema: str = "",
    objpath: str = fastapi.Query("", alias="path"),
    user: str = "",
    size: int = 0,
    offset: int = 0,
    use_secrets: bool = fastapi.Query(True, alias="use-secrets"),
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )

    secrets = {}
    if use_secrets:
        secrets = await _verify_and_get_project_secrets(project, auth_info)

    return await run_in_threadpool(
        _get_files,
        schema,
        objpath,
        user,
        size,
        offset,
        auth_info,
        secrets=secrets,
        project=project,
    )


@router.get("/projects/{project}/filestat")
async def get_filestat_with_project_secrets(
    project: str,
    schema: str = "",
    path: str = "",
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        server.api.api.deps.authenticate_request
    ),
    user: str = "",
    use_secrets: bool = fastapi.Query(True, alias="use-secrets"),
):
    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )

    secrets = {}
    if use_secrets:
        secrets = await _verify_and_get_project_secrets(project, auth_info)

    return await run_in_threadpool(
        server.api.crud.Files().get_filestat,
        auth_info,
        path,
        schema,
        user,
        secrets,
    )


def _get_files(
    schema: str,
    objpath: str,
    user: str,
    size: int,
    offset: int,
    auth_info: mlrun.common.schemas.AuthInfo,
    secrets: dict = None,
    project: str = "",
):
    if size > mlrun.mlconf.artifacts.limits.max_chunk_size:
        log_and_raise(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value,
            err=f"chunk size {size} exceeds the maximum allowed chunk size "
            f"{mlrun.mlconf.artifacts.limits.max_chunk_size}",
        )

    _, filename = objpath.split(objpath)

    objpath = get_obj_path(schema, objpath, user=user)
    if not objpath:
        log_and_raise(
            HTTPStatus.NOT_FOUND.value,
            path=objpath,
            err="illegal path prefix or schema",
        )

    logger.debug("Got get files request", path=objpath)

    secrets = secrets or {}
    secrets.update(get_secrets(auth_info))

    body = None
    try:
        obj = store_manager.object(url=objpath, secrets=secrets, project=project)
        if objpath.endswith("/"):
            listdir = obj.listdir()
            return {
                "listdir": listdir,
            }

        body = obj.get(size, offset)
    except FileNotFoundError as exc:
        log_and_raise(HTTPStatus.NOT_FOUND.value, path=objpath, err=err_to_str(exc))

    if body is None:
        log_and_raise(HTTPStatus.NOT_FOUND.value, path=objpath)

    ctype, _ = mimetypes.guess_type(objpath)
    if not ctype:
        ctype = "application/octet-stream"
    return fastapi.Response(
        content=body, media_type=ctype, headers={"x-suggested-filename": filename}
    )


async def _verify_and_get_project_secrets(project, auth_info):
    # If running on Docker or locally, we cannot retrieve project secrets, so skip.
    if not server.api.utils.singletons.k8s.get_k8s_helper(
        silent=True
    ).is_running_inside_kubernetes_cluster():
        return {}

    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.secret,
        project,
        mlrun.common.schemas.SecretProviderName.kubernetes,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )

    secrets_data = await run_in_threadpool(
        server.api.crud.Secrets().list_project_secrets,
        project,
        mlrun.common.schemas.SecretProviderName.kubernetes,
        allow_secrets_from_k8s=True,
    )
    return secrets_data.secrets or {}
