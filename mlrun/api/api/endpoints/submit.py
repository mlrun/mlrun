from http import HTTPStatus
from typing import Optional

from fastapi import APIRouter, Cookie, Depends, Header, Request
from sqlalchemy.orm import Session

import mlrun.api.api.utils
from mlrun.api.api import deps
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
    iguazio_session: Optional[str] = Cookie(None, alias="session"),
    db_session: Session = Depends(deps.get_db_session),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        mlrun.api.api.utils.log_and_raise(
            HTTPStatus.BAD_REQUEST.value, reason="bad JSON body"
        )

    # enrich job task with the username from the request header
    if username:
        # if task is missing, we don't want to create one
        if "task" in data:
            labels = data["task"].setdefault("metadata", {}).setdefault("labels", {})
            # TODO: remove this duplication
            labels.setdefault("v3io_user", username)
            labels.setdefault("owner", username)

    logger.info("Submit run", data=data)
    response = await mlrun.api.api.utils.submit_run(db_session, data, iguazio_session)
    return response
