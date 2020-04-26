import asyncio
from http import HTTPStatus
from typing import List

from fastapi import APIRouter, Depends, Request, Query
from sqlalchemy.orm import Session

from mlrun.app.api import deps
from mlrun.app.api.utils import json_error
from mlrun.app.db.session import get_db_instance
from mlrun.config import config
from mlrun.utils import logger

router = APIRouter()


# curl -d@/path/to/func.json http://localhost:8080/func/prj/7?tag=0.3.2
@router.post("/func/{project}/{name}")
def store_function(
        request: Request,
        project: str,
        name: str,
        tag: str = "",
        db_session: Session = Depends(deps.get_db_session)):
    try:
        data = asyncio.run(request.json())
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason="bad JSON body")

    logger.debug(data)
    logger.info(
        "store function: project=%s, name=%s, tag=%s", project, name, tag)
    get_db_instance().store_function(db_session, data, name, project, tag=tag)
    return {}


# curl http://localhost:8080/log/prj/7?tag=0.2.3
@router.get("/func/{project}/{name}")
def get_function(
        project: str,
        name: str,
        tag: str = "",
        db_session: Session = Depends(deps.get_db_session)):
    func = get_db_instance().get_function(db_session, name, project, tag)
    return {
        "func": func,
    }


# curl http://localhost:8080/funcs?project=p1&name=x&label=l1&label=l2
@router.get("/funcs")
def list_functions(
        project: str = config.default_project,
        name: str = None,
        tag: str = None,
        labels: List[str] = Query([]),
        db_session: Session = Depends(deps.get_db_session)):
    funcs = get_db_instance().list_functions(db_session, name, project, tag, labels)
    return {
        "funcs": list(funcs),
    }
