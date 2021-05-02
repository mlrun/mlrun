from http import HTTPStatus
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, Query, Request, Response
from sqlalchemy.orm import Session

import mlrun.feature_store
from mlrun import mount_v3io
from mlrun.api import schemas
from mlrun.api.api import deps
from mlrun.api.api.utils import get_secrets, log_and_raise, parse_reference
from mlrun.api.utils.singletons.db import get_db
from mlrun.data_types import InferOptions
from mlrun.datastore.targets import get_default_prefix_for_target
from mlrun.feature_store.api import RunConfig, ingest
from mlrun.model import DataSource, DataTargetBase

router = APIRouter()


@router.post("/projects/{project}/feature-sets", response_model=schemas.FeatureSet)
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


@router.put(
    "/projects/{project}/feature-sets/{name}/references/{reference}",
    response_model=schemas.FeatureSet,
)
def store_feature_set(
    project: str,
    name: str,
    reference: str,
    feature_set: schemas.FeatureSet,
    versioned: bool = True,
    db_session: Session = Depends(deps.get_db_session),
):
    tag, uid = parse_reference(reference)
    uid = get_db().store_feature_set(
        db_session, project, name, feature_set, tag, uid, versioned
    )

    return get_db().get_feature_set(db_session, project, name, uid=uid,)


@router.patch("/projects/{project}/feature-sets/{name}/references/{reference}")
def patch_feature_set(
    project: str,
    name: str,
    feature_set_update: dict,
    reference: str,
    patch_mode: schemas.PatchMode = Header(
        schemas.PatchMode.replace, alias=schemas.HeaderNames.patch_mode
    ),
    db_session: Session = Depends(deps.get_db_session),
):
    tag, uid = parse_reference(reference)
    get_db().patch_feature_set(
        db_session, project, name, feature_set_update, tag, uid, patch_mode
    )
    return Response(status_code=HTTPStatus.OK.value)


@router.get(
    "/projects/{project}/feature-sets/{name}/references/{reference}",
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


@router.delete("/projects/{project}/feature-sets/{name}")
@router.delete("/projects/{project}/feature-sets/{name}/references/{reference}")
def delete_feature_set(
    project: str,
    name: str,
    reference: str = None,
    db_session: Session = Depends(deps.get_db_session),
):
    tag = uid = None
    if reference:
        tag, uid = parse_reference(reference)
    get_db().delete_feature_set(db_session, project, name, tag, uid)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.get(
    "/projects/{project}/feature-sets", response_model=schemas.FeatureSetsOutput
)
def list_feature_sets(
    project: str,
    name: str = None,
    state: str = None,
    tag: str = None,
    entities: List[str] = Query(None, alias="entity"),
    features: List[str] = Query(None, alias="feature"),
    labels: List[str] = Query(None, alias="label"),
    partition_by: schemas.FeatureStorePartitionByField = Query(
        None, alias="partition-by"
    ),
    rows_per_partition: int = Query(1, alias="rows-per-partition", gt=0),
    sort: schemas.SortField = Query(None, alias="partition-sort-by"),
    order: schemas.OrderType = Query(schemas.OrderType.desc, alias="partition-order"),
    db_session: Session = Depends(deps.get_db_session),
):
    feature_sets = get_db().list_feature_sets(
        db_session,
        project,
        name,
        tag,
        state,
        entities,
        features,
        labels,
        partition_by,
        rows_per_partition,
        sort,
        order,
    )

    return feature_sets


def _has_v3io_path(data_source, data_targets, feature_set):
    paths = []

    # If no data targets received, then use targets from the feature-set spec. In case it's empty as well, use
    # default targets (by calling set_targets())
    if not data_targets:
        if not feature_set.spec.targets:
            feature_set.set_targets()
        data_targets = feature_set.spec.targets

    if data_targets:
        for target in data_targets:
            # If the target does not have a path (i.e. default target), then retrieve the default path from config.
            paths.append(target.path or get_default_prefix_for_target(target.kind))

    source = data_source or feature_set.spec.source
    if source:
        paths.append(source.path)

    return any(
        path and (path.startswith("v3io://") or path.startswith("v3ios://"))
        for path in paths
    )


@router.post(
    "/projects/{project}/feature-sets/{name}/references/{reference}/ingest",
    response_model=schemas.FeatureSetIngestOutput,
    status_code=HTTPStatus.ACCEPTED.value,
)
def ingest_feature_set(
    request: Request,
    project: str,
    name: str,
    reference: str,
    ingest_parameters: Optional[
        schemas.FeatureSetIngestInput
    ] = schemas.FeatureSetIngestInput(),
    username: str = Header(None, alias="x-remote-user"),
    db_session: Session = Depends(deps.get_db_session),
):
    tag, uid = parse_reference(reference)
    feature_set_record = get_db().get_feature_set(db_session, project, name, tag, uid)

    feature_set = mlrun.feature_store.FeatureSet.from_dict(feature_set_record.dict())
    # Need to override the default rundb since we're in the server.
    feature_set._override_run_db(db_session)

    data_source = data_targets = None
    if ingest_parameters.source:
        data_source = DataSource.from_dict(ingest_parameters.source.dict())
    if ingest_parameters.targets:
        data_targets = [
            DataTargetBase.from_dict(data_target.dict())
            for data_target in ingest_parameters.targets
        ]

    run_config = RunConfig()

    # Try to deduce whether the ingest job will need v3io mount, by analyzing the paths to the source and
    # targets. If it needs it, apply v3io mount to the run_config. Note that the access-key and username are
    # user-context parameters, we cannot use the api context.
    if _has_v3io_path(data_source, data_targets, feature_set):
        secrets = get_secrets(request)
        access_key = secrets.get("V3IO_ACCESS_KEY", None)

        if not access_key or not username:
            log_and_raise(
                HTTPStatus.BAD_REQUEST.value,
                reason="Request needs v3io access key and username in header",
            )
        run_config = run_config.apply(mount_v3io(access_key=access_key, user=username))

    infer_options = ingest_parameters.infer_options or InferOptions.default()

    run_params = ingest(
        feature_set,
        data_source,
        data_targets,
        infer_options=infer_options,
        return_df=False,
        run_config=run_config,
    )
    # ingest may modify the feature-set contents, so returning the updated feature-set.
    result_feature_set = schemas.FeatureSet(**feature_set.to_dict())
    return schemas.FeatureSetIngestOutput(
        feature_set=result_feature_set, run_object=run_params.to_dict()
    )


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


@router.get("/projects/{project}/entities", response_model=schemas.EntitiesOutput)
def list_entities(
    project: str,
    name: str = None,
    tag: str = None,
    labels: List[str] = Query(None, alias="label"),
    db_session: Session = Depends(deps.get_db_session),
):
    features = get_db().list_entities(db_session, project, name, tag, labels)
    return features


@router.post(
    "/projects/{project}/feature-vectors", response_model=schemas.FeatureVector
)
def create_feature_vector(
    project: str,
    feature_vector: schemas.FeatureVector,
    versioned: bool = True,
    db_session: Session = Depends(deps.get_db_session),
):
    feature_vector_uid = get_db().create_feature_vector(
        db_session, project, feature_vector, versioned
    )

    return get_db().get_feature_vector(
        db_session,
        project,
        feature_vector.metadata.name,
        tag=feature_vector.metadata.tag,
        uid=feature_vector_uid,
    )


@router.get(
    "/projects/{project}/feature-vectors/{name}/references/{reference}",
    response_model=schemas.FeatureVector,
)
def get_feature_vector(
    project: str,
    name: str,
    reference: str,
    db_session: Session = Depends(deps.get_db_session),
):
    tag, uid = parse_reference(reference)
    return get_db().get_feature_vector(db_session, project, name, tag, uid)


@router.get(
    "/projects/{project}/feature-vectors", response_model=schemas.FeatureVectorsOutput
)
def list_feature_vectors(
    project: str,
    name: str = None,
    state: str = None,
    tag: str = None,
    labels: List[str] = Query(None, alias="label"),
    partition_by: schemas.FeatureStorePartitionByField = Query(
        None, alias="partition-by"
    ),
    rows_per_partition: int = Query(1, alias="rows-per-partition", gt=0),
    sort: schemas.SortField = Query(None, alias="partition-sort-by"),
    order: schemas.OrderType = Query(schemas.OrderType.desc, alias="partition-order"),
    db_session: Session = Depends(deps.get_db_session),
):
    feature_vectors = get_db().list_feature_vectors(
        db_session,
        project,
        name,
        tag,
        state,
        labels,
        partition_by,
        rows_per_partition,
        sort,
        order,
    )

    return feature_vectors


@router.put(
    "/projects/{project}/feature-vectors/{name}/references/{reference}",
    response_model=schemas.FeatureVector,
)
def store_feature_vector(
    project: str,
    name: str,
    reference: str,
    feature_vector: schemas.FeatureVector,
    versioned: bool = True,
    db_session: Session = Depends(deps.get_db_session),
):
    tag, uid = parse_reference(reference)
    uid = get_db().store_feature_vector(
        db_session, project, name, feature_vector, tag, uid, versioned
    )

    return get_db().get_feature_vector(db_session, project, name, uid=uid,)


@router.patch("/projects/{project}/feature-vectors/{name}/references/{reference}")
def patch_feature_vector(
    project: str,
    name: str,
    feature_vector_update: dict,
    reference: str,
    patch_mode: schemas.PatchMode = Header(
        schemas.PatchMode.replace, alias=schemas.HeaderNames.patch_mode
    ),
    db_session: Session = Depends(deps.get_db_session),
):
    tag, uid = parse_reference(reference)
    get_db().patch_feature_vector(
        db_session, project, name, feature_vector_update, tag, uid, patch_mode
    )
    return Response(status_code=HTTPStatus.OK.value)


@router.delete("/projects/{project}/feature-vectors/{name}")
@router.delete("/projects/{project}/feature-vectors/{name}/references/{reference}")
def delete_feature_vector(
    project: str,
    name: str,
    reference: str = None,
    db_session: Session = Depends(deps.get_db_session),
):
    tag = uid = None
    if reference:
        tag, uid = parse_reference(reference)
    get_db().delete_feature_vector(db_session, project, name, tag, uid)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)
