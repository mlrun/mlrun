import asyncio
from distutils.util import strtobool
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from mlrun.app.api import deps
from mlrun.app.api.utils import json_error, log_path
from mlrun.app.main import db
from mlrun.app.main import k8s
from mlrun.utils import get_in, now_date, update_in

router = APIRouter()


# curl -d@/path/to/log http://localhost:8080/log/prj/7?append=true
@router.post("/log/{project}/{uid}")
def store_log(
        request: Request,
        project: str,
        uid: str,
        append: str = "on"):
    append = strtobool(append)
    log_file = log_path(project, uid)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    body = asyncio.run(request.body())  # TODO: Check size
    mode = "ab" if append else "wb"
    with log_file.open(mode) as fp:
        fp.write(body)
    return {}


# curl http://localhost:8080/log/prj/7
@router.get("/log/{project}/{uid}")
def get_log(
        project: str,
        uid: str,
        size: int = -1,
        offset: int = 0,
        tag: str = "",
        db_session: Session = Depends(deps.get_db_session)):
    out = b""
    log_file = log_path(project, uid)
    if log_file.exists():
        with log_file.open("rb") as fp:
            fp.seek(offset)
            out = fp.read(size)
        status = ""
    else:
        data = db.read_run(db_session, uid, project)
        if not data:
            return json_error(HTTPStatus.NOT_FOUND,
                              project=project, uid=uid)

        status = get_in(data, "status.state", "")
        if k8s:
            pods = k8s.get_logger_pods(uid)
            if pods:
                pod, new_status = list(pods.items())[0]
                new_status = new_status.lower()

                # TODO: handle in cron/tracking
                if new_status != "pending":
                    resp = k8s.logs(pod)
                    if resp:
                        out = resp.encode()[offset:]
                    if status == "running":
                        now = now_date().isoformat()
                        update_in(data, "status.last_update", now)
                        if new_status == "failed":
                            update_in(data, "status.state", "error")
                            update_in(
                                data, "status.error", "error, check logs")
                            db.store_run(db_session, data, uid, project)
                        if new_status == "succeeded":
                            update_in(data, "status.state", "completed")
                            db.store_run(db_session, data, uid, project)
                status = new_status
            elif status == "running":
                update_in(data, "status.state", "error")
                update_in(
                    data, "status.error", "pod not found, maybe terminated")
                db.store_run(db_session, data, uid, project)
                status = "failed"
    return Response(content=out, media_type="text/plain", headers={"pod_status": status})
