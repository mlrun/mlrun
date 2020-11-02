from http import HTTPStatus
from typing import List

from fastapi import APIRouter, Depends, Response, Query
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.utils.singletons.db import get_db
from .utils import parse_reference

router = APIRouter()


@router.post(
    "/projects/{project}/feature_sets", response_model=schemas.FeatureSetCreateOutput
)
def create_feature_set(
    project: str,
    feature_set: schemas.FeatureSet,
    versioned: bool = False,
    db_session: Session = Depends(deps.get_db_session),
):
    feature_set_id = get_db().create_feature_set(
        db_session, project, feature_set, versioned
    )
    return schemas.FeatureSetCreateOutput(
        uid=feature_set_id, name=feature_set.metadata.name
    )


@router.put("/projects/{project}/feature_sets/{name}/references/{ref}")
def update_feature_set(
    project: str,
    name: str,
    feature_set_update: schemas.FeatureSetUpdate,
    ref: str,
    db_session: Session = Depends(deps.get_db_session),
):
    tag, uid = parse_reference(ref)
    get_db().update_feature_set(db_session, project, name, feature_set_update, tag, uid)
    return Response(status_code=HTTPStatus.OK.value)


@router.get(
    "/projects/{project}/feature_sets/{name}/references/{ref}",
    response_model=schemas.FeatureSet,
)
def get_feature_set(
    project: str,
    name: str,
    ref: str,
    db_session: Session = Depends(deps.get_db_session),
):
    tag, uid = parse_reference(ref)
    feature_set = get_db().get_feature_set(db_session, project, name, tag, uid)
    return feature_set


@router.delete("/projects/{project}/feature_sets/{name}")
def delete_feature_set(
    project: str, name: str, db_session: Session = Depends(deps.get_db_session),
):
    get_db().delete_feature_set(db_session, project, name)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.get(
    "/projects/{project}/feature_sets", response_model=schemas.FeatureSetsOutput
)
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
    feature_sets = get_db().list_feature_sets(
        db_session, project, name, tag, state, entities, features, labels
    )

    return feature_sets
