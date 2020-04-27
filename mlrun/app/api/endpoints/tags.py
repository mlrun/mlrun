import asyncio
from http import HTTPStatus

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from mlrun.app.api import deps
from mlrun.app.api.utils import json_error
from mlrun.app.main import db
from mlrun.db.sqldb import to_dict as db2dict, table2cls

router = APIRouter()


@router.post("/{project}/tag/{name}")
def tag_objects(
        request: Request,
        project: str,
        name: str,
        db_session: Session = Depends(deps.get_db_session)):
    try:
        data = asyncio.run(request.json())
    except ValueError:
        return json_error(HTTPStatus.BAD_REQUEST, reason="bad JSON body")

    objs = []
    for typ, query in data.items():
        cls = table2cls(typ)
        if cls is None:
            err = f"unknown type - {typ}"
            return json_error(HTTPStatus.BAD_REQUEST, reason=err)
        # {"name": "bugs"} -> [Function.name=="bugs"]
        db_query = [
            getattr(cls, key) == value for key, value in query.items()
        ]
        # TODO: Change _query to query?
        # TODO: Not happy about exposing db internals to API
        objs.extend(db_session.query(cls).filter(*db_query))
    db.tag_objects(db_session, objs, project, name)
    return {
        "project": project,
        "name": name,
        "count": len(objs),
    }


@router.delete("/{project}/tag/{name}")
def del_tag(
        project: str,
        name: str,
        db_session: Session = Depends(deps.get_db_session)):
    count = db.del_tag(db_session, project, name)
    return {
        "project": project,
        "name": name,
        "count": count,
    }


@router.get("/{project}/tags")
def list_tags(
        project: str,
        db_session: Session = Depends(deps.get_db_session)):
    tags = db.list_tags(db_session, project)
    return {
        "project": project,
        "tags": tags,
    }


@router.get("/{project}/tag/{name}")
def get_tagged(
        project: str,
        name: str,
        db_session: Session = Depends(deps.get_db_session)):
    objs = db.find_tagged(db_session, project, name)
    return {
        "project": project,
        "tag": name,
        "objects": [db2dict(obj) for obj in objs],
    }
