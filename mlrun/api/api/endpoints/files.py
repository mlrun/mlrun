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

import mlrun.api.api.deps
import mlrun.api.schemas
from mlrun.api.api.utils import get_obj_path, get_secrets, log_and_raise
from mlrun.datastore import store_manager
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
    _, filename = objpath.split(objpath)

    objpath = get_obj_path(schema, objpath, user=user)
    if not objpath:
        log_and_raise(
            HTTPStatus.NOT_FOUND.value,
            path=objpath,
            err="illegal path prefix or schema",
        )

    logger.debug("Got get files request", path=objpath)

    secrets = get_secrets(auth_info)
    body = None
    try:
        stores = store_manager.set(secrets)
        obj = stores.object(url=objpath)
        if objpath.endswith("/"):
            listdir = obj.listdir()
            return {
                "listdir": listdir,
            }

        body = obj.get(size, offset)
    except FileNotFoundError as exc:
        log_and_raise(HTTPStatus.NOT_FOUND.value, path=objpath, err=str(exc))

    if body is None:
        log_and_raise(HTTPStatus.NOT_FOUND.value, path=objpath)

    ctype, _ = mimetypes.guess_type(objpath)
    if not ctype:
        ctype = "application/octet-stream"
    return fastapi.Response(
        content=body, media_type=ctype, headers={"x-suggested-filename": filename}
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
    _, filename = path.split(path)

    path = get_obj_path(schema, path, user=user)
    if not path:
        log_and_raise(
            HTTPStatus.NOT_FOUND.value, path=path, err="illegal path prefix or schema"
        )

    logger.debug("Got get filestat request", path=path)

    secrets = get_secrets(auth_info)
    stat = None
    try:
        stores = store_manager.set(secrets)
        stat = stores.object(url=path).stat()
    except FileNotFoundError as exc:
        log_and_raise(HTTPStatus.NOT_FOUND.value, path=path, err=str(exc))

    ctype, _ = mimetypes.guess_type(path)
    if not ctype:
        ctype = "application/octet-stream"

    return {
        "size": stat.size,
        "modified": stat.modified,
        "mimetype": ctype,
    }
