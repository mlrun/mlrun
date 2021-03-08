import mimetypes
from http import HTTPStatus

from fastapi import APIRouter, Query, Request, Response

from mlrun.api.api.utils import get_obj_path, get_secrets, log_and_raise
from mlrun.datastore import store_manager
from mlrun.utils import logger

router = APIRouter()


# curl http://localhost:8080/api/files?schema=s3&path=mybucket/a.txt
@router.get("/files")
def get_files(
    request: Request,
    schema: str = "",
    objpath: str = Query("", alias="path"),
    user: str = "",
    size: int = 0,
    offset: int = 0,
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

    secrets = get_secrets(request)
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
    return Response(
        content=body, media_type=ctype, headers={"x-suggested-filename": filename}
    )


# curl http://localhost:8080/api/filestat?schema=s3&path=mybucket/a.txt
@router.get("/filestat")
def get_filestat(request: Request, schema: str = "", path: str = "", user: str = ""):
    _, filename = path.split(path)

    path = get_obj_path(schema, path, user=user)
    if not path:
        log_and_raise(
            HTTPStatus.NOT_FOUND.value, path=path, err="illegal path prefix or schema"
        )

    logger.debug("Got get filestat request", path=path)

    secrets = get_secrets(request)
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
