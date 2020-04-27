import asyncio
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from mlrun.app.api import deps
from mlrun.app.api.utils import json_error
from mlrun.app.api.utils import submit
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
    try:
        data = asyncio.run(request.json())
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason="bad JSON body")

    logger.info('submit_job: {}'.format(data))
    return submit(db_session, data)
