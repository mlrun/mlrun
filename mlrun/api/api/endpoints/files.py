import mimetypes
import requests
from http import HTTPStatus

from fastapi import APIRouter, Query, Request, Response, status

from mlrun.api.api.utils import log_and_raise, get_obj_path, get_secrets
from mlrun.datastore.v3io import ForbiddenPathAccessException
from mlrun.datastore import get_object_stat, store_manager

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
            HTTPStatus.NOT_FOUND, path=objpath, err="illegal path prefix or schema"
        )

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
    except FileNotFoundError as e:
        log_and_raise(HTTPStatus.NOT_FOUND, path=objpath, err=str(e))
    except ForbiddenPathAccessException as e:
        log_and_raise(status.HTTP_403_FORBIDDEN, path=objpath, err=str(e))
    except requests.HTTPError as e:
        if e.response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
        ]:
            log_and_raise(status.HTTP_403_FORBIDDEN, path=objpath, err=str(e))
        raise e
    if body is None:
        log_and_raise(HTTPStatus.NOT_FOUND, path=objpath)

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
            HTTPStatus.NOT_FOUND, path=path, err="illegal path prefix or schema"
        )
    secrets = get_secrets(request)
    stat = None
    try:
        stat = get_object_stat(path, secrets)
    except FileNotFoundError as e:
        log_and_raise(HTTPStatus.NOT_FOUND, path=path, err=str(e))

    ctype, _ = mimetypes.guess_type(path)
    if not ctype:
        ctype = "application/octet-stream"

    return {
        "size": stat.size,
        "modified": stat.modified,
        "mimetype": ctype,
    }
