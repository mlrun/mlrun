from http import HTTPStatus
from typing import List, Optional

from fastapi import APIRouter, Query, Request, Response

from mlrun.api.crud.model_endpoints import ModelEndpoints, get_access_key
from mlrun.api.schemas import ModelEndpoint, ModelEndpointList
from mlrun.errors import MLRunConflictError, MLRunInvalidArgumentError

router = APIRouter()


@router.post(
    "/projects/{project}/model-endpoints/register",
    status_code=HTTPStatus.NO_CONTENT.value,
)
async def register_endpoint(request: Request, project: str):
    access_key = get_access_key(request.headers)
    payload = await request.json()

    # Required parameters
    model = get_or_raise(payload, "model")
    function = get_or_raise(payload, "function")
    tag = get_or_raise(payload, "tag")

    model_class = payload.get("model_class")
    labels = payload.get("labels")
    model_artifact = payload.get("model_artifact")
    feature_stats = payload.get("feature_stats")
    feature_names = payload.get("feature_names")
    stream_path = payload.get("stream_path")
    active = payload.get("active")

    ModelEndpoints.register_endpoint(
        access_key=access_key,
        project=project,
        model=model,
        function=function,
        tag=tag,
        model_class=model_class,
        labels=labels,
        model_artifact=model_artifact,
        feature_stats=feature_stats,
        feature_names=feature_names,
        stream_path=stream_path,
        active=active,
    )


@router.post(
    "/projects/{project}/model-endpoints/{endpoint_id}/update",
    status_code=HTTPStatus.NO_CONTENT.value,
)
async def update_endpoint(
    request: Request, project: str, endpoint_id: str,
):
    verify_endpoint(project, endpoint_id)
    access_key = get_access_key(request.headers)
    payload = await request.json()
    ModelEndpoints.update_endpoint_record(
        access_key=access_key, project=project, endpoint_id=endpoint_id, payload=payload
    )
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.post(
    "/projects/{project}/model-endpoints/{endpoint_id}/clear",
    status_code=HTTPStatus.NO_CONTENT.value,
)
def clear_endpoint_record(request: Request, project: str, endpoint_id: str):
    """
    Clears endpoint record from KV by endpoint_id
    """
    verify_endpoint(project, endpoint_id)
    access_key = get_access_key(request.headers)
    ModelEndpoints.clear_endpoint_record(
        access_key=access_key, project=project, endpoint_id=endpoint_id
    )
    return Response(status_code=HTTPStatus.NO_CONTENT.value)


@router.get("/projects/{project}/model-endpoints", response_model=ModelEndpointList)
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
    access_key = get_access_key(request.headers)
    endpoints = ModelEndpoints.list_endpoints(
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
):
    verify_endpoint(project, endpoint_id)
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


def get_or_raise(dictionary: dict, key: str):
    if key not in dictionary:
        raise MLRunInvalidArgumentError(
            f"Required argument '{key}' missing from json payload: {dictionary}"
        )
    return dictionary[key]


def verify_endpoint(project, endpoint_id):
    endpoint_id_project, _ = endpoint_id.split(".")
    if endpoint_id_project != project:
        raise MLRunConflictError(
            f"project: {project} and endpoint_id: {endpoint_id} mismatch."
        )
