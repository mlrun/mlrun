from http import HTTPStatus
from typing import List, Optional

from fastapi import APIRouter, Cookie, Depends, Query, Request, Response
from sqlalchemy.orm import Session

import mlrun.api.api
from mlrun.api.crud.model_endpoints import ModelEndpoints, get_access_key
from mlrun.api.schemas import ModelEndpoint, ModelEndpointList
from mlrun.errors import MLRunConflictError

router = APIRouter()


@router.put(
    "/projects/{project}/model-endpoints/{endpoint_id}",
    status_code=HTTPStatus.NO_CONTENT.value,
)
def create_or_patch(
    request: Request,
    project: str,
    endpoint_id: str,
    model_endpoint: ModelEndpoint,
    iguazio_session: Optional[str] = Cookie(None, alias="session"),
    db_session: Session = Depends(mlrun.api.api.deps.get_db_session),
) -> Response:
    """
    Either create or updates the kv record of a given ModelEndpoint object
    """
    access_key = get_access_key(request.headers)
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
        leader_session=iguazio_session,
    )
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.delete(
    "/projects/{project}/model-endpoints/{endpoint_id}",
    status_code=HTTPStatus.NO_CONTENT.value,
)
def delete_endpoint_record(
    request: Request, project: str, endpoint_id: str
) -> Response:
    """
    Clears endpoint record from KV by endpoint_id
    """
    access_key = get_access_key(request.headers)
    ModelEndpoints.delete_endpoint_record(
        access_key=access_key, project=project, endpoint_id=endpoint_id,
    )
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.get("/projects/{project}/model-endpoints", response_model=ModelEndpointList)
def list_endpoints(
    request: Request,
    project: str,
    model: Optional[str] = Query(None),
    function: Optional[str] = Query(None),
    labels: List[str] = Query([], alias="label"),
    start: str = Query(default="now-1h"),
    end: str = Query(default="now"),
    metrics: List[str] = Query([], alias="metric"),
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
    access_key = get_access_key(request.headers)
    endpoints = ModelEndpoints.list_endpoints(
        access_key=access_key,
        project=project,
        model=model,
        function=function,
        labels=labels,
        metrics=metrics,
        start=start,
        end=end,
    )
    return endpoints


@router.get(
    "/projects/{project}/model-endpoints/{endpoint_id}", response_model=ModelEndpoint
)
def get_endpoint(
    request: Request,
    project: str,
    endpoint_id: str,
    start: str = Query(default="now-1h"),
    end: str = Query(default="now"),
    metrics: List[str] = Query([], alias="metric"),
    feature_analysis: bool = Query(default=False),
) -> ModelEndpoint:
    access_key = get_access_key(request.headers)
    endpoint = ModelEndpoints.get_endpoint(
        access_key=access_key,
        project=project,
        endpoint_id=endpoint_id,
        metrics=metrics,
        start=start,
        end=end,
        feature_analysis=feature_analysis,
    )
    return endpoint
