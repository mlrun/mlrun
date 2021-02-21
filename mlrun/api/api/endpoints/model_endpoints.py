from http import HTTPStatus
from typing import List, Optional

from fastapi import APIRouter, Query, Response, Request

from mlrun.api.crud.model_endpoints import ModelEndpoints, get_access_key
from mlrun.api.schemas import (
    ModelEndpointStateList,
    ModelEndpointState,
)

router = APIRouter()


@router.post(
    "/projects/{project}/model-endpoints/{endpoint_id}/clear",
    status_code=HTTPStatus.NO_CONTENT.value,
)
def clear_endpoint_record(request: Request, project: str, endpoint_id: str):
    """
    Clears endpoint record from KV by endpoint_id
    """
    access_key = get_access_key(request)
    ModelEndpoints.clear_endpoint_record(access_key, project, endpoint_id)
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.get(
    "/projects/{project}/model-endpoints", response_model=ModelEndpointStateList
)
def list_endpoints(
    request: Request,
    project: str,
    model: Optional[str] = Query(None),
    function: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    labels: List[str] = Query([], alias="label"),
    start: str = Query(default="now-1h"),
    end: str = Query(default="now"),
    metrics: List[str] = Query([], alias="metric"),
):
    """
     Returns a list of endpoints of type 'ModelEndpoint', supports filtering by model, function, tag and labels.
     Lables can be used to filter on the existance of a label:
     `api/projects/{project}/model-endpoints/?label=mylabel`

     Or on the value of a given label:
     `api/projects/{project}/model-endpoints/?label=mylabel=1`

     Multiple labels can be queried in a single request by either using `&` seperator:
     `api/projects/{project}/model-endpoints/?label=mylabel=1&label=myotherlabel=2`

     Or by using a `,` (comma) seperator:
     `api/projects/{project}/model-endpoints/?label=mylabel=1,myotherlabel=2`
     """
    access_key = get_access_key(request)
    endpoint_list = ModelEndpoints.list_endpoints(
        access_key=access_key,
        project=project,
        model=model,
        function=function,
        tag=tag,
        labels=labels,
        metrics=metrics,
        start=start,
        end=end,
    )
    return ModelEndpointStateList(endpoints=endpoint_list)


@router.get(
    "/projects/{project}/model-endpoints/{endpoint_id}",
    response_model=ModelEndpointState,
)
def get_endpoint(
    request: Request,
    project: str,
    endpoint_id: str,
    start: str = Query(default="now-1h"),
    end: str = Query(default="now"),
    metrics: List[str] = Query([], alias="metric"),
    features: bool = Query(default=False),
):
    access_key = get_access_key(request)
    return ModelEndpoints.get_endpoint(
        access_key=access_key,
        project=project,
        endpoint_id=endpoint_id,
        metrics=metrics,
        start=start,
        end=end,
        features=features,
    )
