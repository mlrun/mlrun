from fastapi import APIRouter, Depends, Request, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.crud as crud
from mlrun.api.api import deps

router = APIRouter()


# curl -d@/path/to/log http://localhost:8080/log/prj/7?append=true
@router.post("/log/{project}/{uid}")
async def store_log(request: Request, project: str, uid: str, append: bool = True):
    body = await request.body()
    await run_in_threadpool(crud.Logs.store_log, body, project, uid, append)
    return {}


# curl http://localhost:8080/log/prj/7
@router.get("/log/{project}/{uid}")
def get_log(
    project: str,
    uid: str,
    size: int = -1,
    offset: int = 0,
    db_session: Session = Depends(deps.get_db_session),
):
    out, status = crud.Logs.get_log(db_session, project, uid, size, offset)
    return Response(
        content=out, media_type="text/plain", headers={"pod_status": status}
    )
