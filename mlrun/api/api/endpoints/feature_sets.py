from operator import attrgetter
from http import HTTPStatus

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.api.utils import log_and_raise
from mlrun.api.db.sqldb.helpers import to_dict as db2dict
from mlrun.api.utils.singletons.db import get_db

router = APIRouter()


@router.post("/projects/{project}/feature_sets")
def add_feature_set(
        project: str,
        feature_set: schemas.FeatureSet,
        db_session: Session = Depends(deps.get_db_session),
):
    fs_id = get_db().add_feature_set(db_session, project, feature_set.dict())
    return {
        "id": fs_id,
        "name": feature_set.name,
    }


@router.put("/projects/{project}/feature_sets/{name}")
def update_feature_set(
        project: str,
        name: str,
        feature_set: schemas.FeatureSet,
        db_session: Session = Depends(deps.get_db_session),
):
    fs_id = get_db().update_feature_set(db_session, project, feature_set.dict())
    if not fs_id:
        log_and_raise(
            HTTPStatus.NOT_FOUND.value,
            reason="feature set doesn't exist {}/{}".format(project, name),
        )

    return {
        "id": fs_id,
        "name": feature_set.name,
    }


@router.get("/projects/{project}/feature_sets/{name}")
def get_feature_set(
    project: str,
    name: str,
    db_session: Session = Depends(deps.get_db_session),
):
    fs = get_db().get_feature_set(db_session, project, name)
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
        project: str,
        name: str,
        db_session: Session = Depends(deps.get_db_session),
):
    fs_id = get_db().delete_feature_set(db_session, project, name)
    if not fs_id:
        log_and_raise(
            HTTPStatus.NOT_FOUND.value,
            reason="feature set doesn't exist {}/{}".format(project, name),
        )


@router.get("/projects/{project}/feature_sets")
def list_feature_sets(
    project: str,
    db_session: Session = Depends(deps.get_db_session),
):
    fs_list = get_db().list_feature_sets(db_session, project)

    return {
        "feature_sets": fs_list,
    }
