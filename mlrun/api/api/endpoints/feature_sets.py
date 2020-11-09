from http import HTTPStatus
from typing import List

from fastapi import APIRouter, Depends, Response, Query
from sqlalchemy.orm import Session

from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.utils.singletons.db import get_db
from mlrun.api.api.utils import parse_reference

router = APIRouter()


@router.post("/projects/{project}/feature_sets", response_model=schemas.FeatureSet)
def create_feature_set(
    project: str,
    feature_set: schemas.FeatureSet,
    versioned: bool = True,
    db_session: Session = Depends(deps.get_db_session),
):
    feature_set_uid = get_db().create_feature_set(
        db_session, project, feature_set, versioned
    )

    return get_db().get_feature_set(
        db_session,
        project,
        feature_set.metadata.name,
        tag=feature_set.metadata.tag,
        uid=feature_set_uid,
    )


@router.put("/projects/{project}/feature_sets/{name}/references/{reference}")
def update_feature_set(
    project: str,
    name: str,
    feature_set_update: schemas.FeatureSetUpdate,
    reference: str,
    db_session: Session = Depends(deps.get_db_session),
):
    tag, uid = parse_reference(reference)
    get_db().update_feature_set(db_session, project, name, feature_set_update, tag, uid)
    return Response(status_code=HTTPStatus.OK.value)


@router.get(
    "/projects/{project}/feature_sets/{name}/references/{reference}",
    response_model=schemas.FeatureSet,
)
def get_feature_set(
    project: str,
    name: str,
    reference: str,
    db_session: Session = Depends(deps.get_db_session),
):
    tag, uid = parse_reference(reference)
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


@router.get("/projects/{project}/features", response_model=schemas.FeaturesOutput)
def list_features(
    project: str,
    name: str = None,
    tag: str = None,
    entities: List[str] = Query(None, alias="entity"),
    labels: List[str] = Query(None, alias="label"),
    db_session: Session = Depends(deps.get_db_session),
):
    features = get_db().list_features(db_session, project, name, tag, entities, labels)
    return features
