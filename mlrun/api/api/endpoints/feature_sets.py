from http import HTTPStatus
from typing import List

from fastapi import APIRouter, Depends, Response, Query
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.api.utils import log_and_raise
from mlrun.api.utils.singletons.db import get_db

router = APIRouter()


@router.post("/projects/{project}/feature_sets")
def add_feature_set(
    project: str,
    feature_set: schemas.FeatureSet,
    versioned: bool = False,
    db_session: Session = Depends(deps.get_db_session),
):
    fs_id = get_db().add_feature_set(db_session, project, feature_set.dict(), versioned)

    return {
        "id": fs_id,
        "name": feature_set.metadata.name,
    }


@router.put("/projects/{project}/feature_sets/{name}")
def update_feature_set(
    project: str,
    name: str,
    feature_set: schemas.FeatureSetUpdate,
    tag: str = None,
    uid: str = None,
    db_session: Session = Depends(deps.get_db_session),
):
    get_db().update_feature_set(db_session, project, name, feature_set.dict(), tag, uid)
    return Response(status_code=HTTPStatus.OK.value)


@router.get("/projects/{project}/feature_sets/{name}")
def get_feature_set(
    project: str,
    name: str,
    tag: str = None,
    hash_key: str = None,
    db_session: Session = Depends(deps.get_db_session),
):
    fs = get_db().get_feature_set(db_session, project, name, tag, hash_key)
    if not fs:
        log_and_raise(
            HTTPStatus.NOT_FOUND.value,
            reason="feature set doesn't exist {}/{}".format(project, name),
        )

    return {
        "feature_set": fs,
    }


@router.delete("/projects/{project}/feature_sets/{name}")
def delete_feature_set(
    project: str, name: str, db_session: Session = Depends(deps.get_db_session),
):
    get_db().delete_feature_set(db_session, project, name)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.get("/projects/{project}/feature_sets")
def list_feature_sets(
    project: str,
    name: str = None,
    state: str = None,
    tag: str = None,
    entities: List[str] = Query(None, alias="entity"),
    features: List[str] = Query(None, alias="feature"),
    labels: List[str] = Query(None, alias="label"),
    db_session: Session = Depends(deps.get_db_session),
):
    fs_list = get_db().list_feature_sets(
        db_session, project, name, tag, state, entities, features, labels
    )

    return {
        "feature_sets": fs_list,
    }
