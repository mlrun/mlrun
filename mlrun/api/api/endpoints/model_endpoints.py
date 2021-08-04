from http import HTTPStatus
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

import mlrun.api.api.deps
import mlrun.api.utils.clients.opa
from mlrun.api.crud.model_endpoints import ModelEndpoints, get_access_key
from mlrun.api.schemas import ModelEndpoint, ModelEndpointList
from mlrun.errors import MLRunConflictError

router = APIRouter()


@router.put(
    "/projects/{project}/model-endpoints/{endpoint_id}",
    status_code=HTTPStatus.NO_CONTENT.value,
)
def create_or_patch(
    project: str,
    endpoint_id: str,
    model_endpoint: ModelEndpoint,
    auth_verifier: mlrun.api.api.deps.AuthVerifier = Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
) -> Response:
    """
    Either create or updates the kv record of a given ModelEndpoint object
    """
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.model_endpoint,
        project,
        endpoint_id,
        mlrun.api.schemas.AuthorizationAction.store,
        auth_verifier.auth_info,
    )
    access_key = get_access_key(auth_verifier.auth_info)
    if project != model_endpoint.metadata.project:
        raise MLRunConflictError(
            f"Can't store endpoint of project {model_endpoint.metadata.project} into project {project}"
        )
    if endpoint_id != model_endpoint.metadata.uid:
        raise MLRunConflictError(
            f"Mismatch between endpoint_id {endpoint_id} and ModelEndpoint.metadata.uid {model_endpoint.metadata.uid}."
            f"\nMake sure the supplied function_uri, and model are configured as intended"
        )
    ModelEndpoints.create_or_patch(
        db_session=db_session,
        access_key=access_key,
        model_endpoint=model_endpoint,
        auth_info=auth_verifier.auth_info,
    )
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.delete(
    "/projects/{project}/model-endpoints/{endpoint_id}",
    status_code=HTTPStatus.NO_CONTENT.value,
)
def delete_endpoint_record(
    project: str,
    endpoint_id: str,
    auth_verifier: mlrun.api.api.deps.AuthVerifier = Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
) -> Response:
    """
    Clears endpoint record from KV by endpoint_id
    """
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.model_endpoint,
        project,
        endpoint_id,
        mlrun.api.schemas.AuthorizationAction.delete,
        auth_verifier.auth_info,
    )
    ModelEndpoints.delete_endpoint_record(auth_verifier.auth_info, project, endpoint_id)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.get("/projects/{project}/model-endpoints", response_model=ModelEndpointList)
def list_endpoints(
    project: str,
    model: Optional[str] = Query(None),
    function: Optional[str] = Query(None),
    labels: List[str] = Query([], alias="label"),
    start: str = Query(default="now-1h"),
    end: str = Query(default="now"),
    metrics: List[str] = Query([], alias="metric"),
    auth_verifier: mlrun.api.api.deps.AuthVerifier = Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
) -> ModelEndpointList:
    """
     Returns a list of endpoints of type 'ModelEndpoint', supports filtering by model, function, tag and labels.
     Labels can be used to filter on the existence of a label:
     api/projects/{project}/model-endpoints/?label=mylabel

     Or on the value of a given label:
     api/projects/{project}/model-endpoints/?label=mylabel=1

     Multiple labels can be queried in a single request by either using "&" separator:
     api/projects/{project}/model-endpoints/?label=mylabel=1&label=myotherlabel=2

     Or by using a "," (comma) separator:
     api/projects/{project}/model-endpoints/?label=mylabel=1,myotherlabel=2
     """
    endpoints = ModelEndpoints.list_endpoints(
        auth_info=auth_verifier.auth_info,
        project=project,
        model=model,
        function=function,
        labels=labels,
        metrics=metrics,
        start=start,
        end=end,
    )
    allowed_endpoints = mlrun.api.utils.clients.opa.Client().filter_resources_by_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.model_endpoint,
        endpoints.endpoints,
        lambda _endpoint: (_endpoint.metadata.project, _endpoint.metadata.uid,),
        auth_verifier.auth_info,
    )
    endpoints.endpoints = allowed_endpoints
    return endpoints


@router.get(
    "/projects/{project}/model-endpoints/{endpoint_id}", response_model=ModelEndpoint
)
def get_endpoint(
    project: str,
    endpoint_id: str,
    start: str = Query(default="now-1h"),
    end: str = Query(default="now"),
    metrics: List[str] = Query([], alias="metric"),
    feature_analysis: bool = Query(default=False),
    auth_verifier: mlrun.api.api.deps.AuthVerifier = Depends(
        mlrun.api.api.deps.AuthVerifier
    ),
) -> ModelEndpoint:
    mlrun.api.utils.clients.opa.Client().query_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.model_endpoint,
        project,
        endpoint_id,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_verifier.auth_info,
    )
    endpoint = ModelEndpoints.get_endpoint(
        auth_info=auth_verifier.auth_info,
        project=project,
        endpoint_id=endpoint_id,
        metrics=metrics,
        start=start,
        end=end,
        feature_analysis=feature_analysis,
    )
    return endpoint
