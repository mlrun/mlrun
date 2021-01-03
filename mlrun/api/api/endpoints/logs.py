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
    run_state, log = crud.Logs.get_logs(db_session, project, uid, size, offset)
    headers = {
        "x-mlrun-run-state": run_state,
        # pod_status was changed x-mlrun-run-state in 0.5.3, keeping it here for backwards compatibility (so <0.5.3
        # clients will work with the API)
        # TODO: remove this in 0.7.0
        "pod_status": run_state,
    }
    return Response(content=log, media_type="text/plain", headers=headers)
