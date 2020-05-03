import asyncio
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from .. import deps
from ..utils import log_and_raise, submit
from mlrun.utils import logger

router = APIRouter()


# curl -d@/path/to/job.json http://localhost:8080/submit
@router.post("/submit")
@router.post("/submit/")
@router.post("/submit_job")
@router.post("/submit_job/")
def submit_job(
        request: Request,
        db_session: Session = Depends(deps.get_db_session)):
    data = None
    try:
        data = asyncio.run(request.json())
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST, reason="bad JSON body")

    logger.info("submit_job: {}".format(data))
    return submit(db_session, data)
