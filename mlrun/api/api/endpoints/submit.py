from http import HTTPStatus
from typing import Optional

from fastapi import APIRouter, Depends, Request, Header
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from mlrun.api.api import deps
from mlrun.api.api.utils import log_and_raise, submit
from mlrun.utils import logger

router = APIRouter()


# curl -d@/path/to/job.json http://localhost:8080/submit
@router.post("/submit")
@router.post("/submit/")
@router.post("/submit_job")
@router.post("/submit_job/")
async def submit_job(
    request: Request,
    username: Optional[str] = Header(None, alias="x-remote-user"),
    db_session: Session = Depends(deps.get_db_session),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    # enrich job task with the username from the request header
    if username:
        # if task is missing, we don't want to create one
        if "task" in data:
            labels = data["task"].setdefault("metadata", {}).setdefault("labels", {})
            # TODO: remove this duplication
            labels.setdefault("v3io_user", username)
            labels.setdefault("owner", username)

    logger.info("submit_job: {}".format(data))
    response = await run_in_threadpool(submit, db_session, data)
    return response
