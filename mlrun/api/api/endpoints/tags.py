from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from mlrun.api.api import deps
from mlrun.api.api.utils import log_and_raise
from mlrun.api.db.sqldb.helpers import table2cls
from mlrun.api.utils.singletons.db import get_db

router = APIRouter()


@router.post("/{project}/tag/{name}")
async def tag_objects(
    request: Request,
    project: str,
    name: str,
    db_session: Session = Depends(deps.get_db_session),
):
    data = None
    try:
        data = await request.json()
    except ValueError:
        log_and_raise(HTTPStatus.BAD_REQUEST.value, reason="bad JSON body")

    objs = await run_in_threadpool(_tag_objects, db_session, data, project, name)
    return {
        "project": project,
        "name": name,
        "count": len(objs),
    }


@router.delete("/{project}/tag/{name}")
def del_tag(
    project: str, name: str, db_session: Session = Depends(deps.get_db_session)
):
    count = get_db().del_tag(db_session, project, name)
    return {
        "project": project,
        "name": name,
        "count": count,
    }


@router.get("/{project}/tags")
def list_tags(project: str, db_session: Session = Depends(deps.get_db_session)):
    tags = get_db().list_tags(db_session, project)
    return {
        "project": project,
        "tags": tags,
    }


@router.get("/{project}/tag/{name}")
def get_tagged(
    project: str, name: str, db_session: Session = Depends(deps.get_db_session)
):
    objs = get_db().find_tagged(db_session, project, name)
    return {
        "project": project,
        "tag": name,
        "objects": [obj.to_dict() for obj in objs],
    }


def _tag_objects(db_session, data, project, name):
    objs = []
    for typ, query in data.items():
        cls = table2cls(typ)
        if cls is None:
            err = f"unknown type - {typ}"
            log_and_raise(HTTPStatus.BAD_REQUEST.value, reason=err)
        # {"name": "bugs"} -> [Function.name=="bugs"]
        db_query = [getattr(cls, key) == value for key, value in query.items()]
        # TODO: Change _query to query?
        # TODO: Not happy about exposing db internals to API
        objs.extend(db_session.query(cls).filter(*db_query))
    get_db().tag_objects(db_session, objs, project, name)
    return objs
