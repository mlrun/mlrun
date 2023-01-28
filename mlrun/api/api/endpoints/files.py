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

import fastapi
from fastapi.concurrency import run_in_threadpool

import mlrun.api.api.deps
import mlrun.api.crud.secrets
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
from mlrun.api.api.utils import get_obj_path, get_secrets, log_and_raise
from mlrun.datastore import store_manager
from mlrun.errors import err_to_str
from mlrun.utils import logger

router = fastapi.APIRouter()


@router.get("/files")
def get_files(
    schema: str = "",
    objpath: str = fastapi.Query("", alias="path"),
    user: str = "",
    size: int = 0,
    offset: int = 0,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    return _get_files(schema, objpath, user, size, offset, auth_info)


@router.get("/projects/{project}/files")
async def get_files_with_project_secrets(
    project: str,
    schema: str = "",
    objpath: str = fastapi.Query("", alias="path"),
    user: str = "",
    size: int = 0,
    offset: int = 0,
    use_secrets: bool = fastapi.Query(True, alias="use-secrets"),
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )

    secrets = {}
    if use_secrets:
        secrets = await _verify_and_get_project_secrets(project, auth_info)

    return await run_in_threadpool(
        _get_files, schema, objpath, user, size, offset, auth_info, secrets=secrets
    )


@router.get("/filestat")
def get_filestat(
    schema: str = "",
    path: str = "",
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    user: str = "",
):
    return _get_filestat(schema, path, user, auth_info)


@router.get("/projects/{project}/filestat")
async def get_filestat_with_project_secrets(
    project: str,
    schema: str = "",
    path: str = "",
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    user: str = "",
    use_secrets: bool = fastapi.Query(True, alias="use-secrets"),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )

    secrets = {}
    if use_secrets:
        secrets = await _verify_and_get_project_secrets(project, auth_info)

    return await run_in_threadpool(
        _get_filestat, schema, path, user, auth_info, secrets=secrets
    )


def _get_files(
    schema: str,
    objpath: str,
    user: str,
    size: int,
    offset: int,
    auth_info: mlrun.api.schemas.AuthInfo,
    secrets: dict = None,
):
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
        obj = store_manager.object(url=objpath, secrets=secrets)
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


def _get_filestat(
    schema: str,
    path: str,
    user: str,
    auth_info: mlrun.api.schemas.AuthInfo,
    secrets: dict = None,
):
    _, filename = path.split(path)

    path = get_obj_path(schema, path, user=user)
    if not path:
        log_and_raise(
            HTTPStatus.NOT_FOUND.value, path=path, err="illegal path prefix or schema"
        )

    logger.debug("Got get filestat request", path=path)

    secrets = secrets or {}
    secrets.update(get_secrets(auth_info))

    stat = None
    try:
        stat = store_manager.object(url=path, secrets=secrets).stat()
    except FileNotFoundError as exc:
        log_and_raise(HTTPStatus.NOT_FOUND.value, path=path, err=err_to_str(exc))

    ctype, _ = mimetypes.guess_type(path)
    if not ctype:
        ctype = "application/octet-stream"

    return {
        "size": stat.size,
        "modified": stat.modified,
        "mimetype": ctype,
    }


async def _verify_and_get_project_secrets(project, auth_info):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.secret,
        project,
        mlrun.api.schemas.SecretProviderName.kubernetes,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )
    secrets_data = await run_in_threadpool(
        mlrun.api.crud.Secrets().list_project_secrets,
        project,
        mlrun.api.schemas.SecretProviderName.kubernetes,
        allow_secrets_from_k8s=True,
    )
    return secrets_data.secrets or {}
