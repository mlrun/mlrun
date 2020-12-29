import json
from enum import Enum
from typing import List, Optional, Dict, Any
import os
from fastapi import APIRouter, Query
from pandas import DataFrame
from v3io.dataplane import RaiseForStatus
import hashlib
from mlrun.api import schemas
from mlrun.config import config
from mlrun.errors import (
    MLRunConflictError,
    MLRunNotFoundError,
    MLRunInvalidArgumentError,
)
from mlrun.utils import logger
from mlrun.utils.v3io_clients import get_v3io_client, get_frames_client

ENDPOINTS = "endpoints"
ENDPOINT_EVENTS = "endpoint_events"


class MetricType(Enum):
    LATENCY = "latency_avg_1s"
    PREDICTIONS = "predictions_per_second_count_1s"

    # Will be supported later on
    # REQUESTS = "requests"
    # DRIFT = "drift_magnitude"
    # ACCURACY = "accuracy"

    @staticmethod
    def from_string(name: str):
        if name in {"microsec", "latency"}:
            return MetricType.LATENCY
        elif name in {"preds", "predictions"}:
            return MetricType.PREDICTIONS
        # elif name in {"reqs", "requests"}:
        #     return MetricType.REQUESTS
        # elif name in {"drift", "drift_magnitude", "drift-magnitude"}:
        #     return MetricType.DRIFT
        # elif name in {"acc", "accuracy"}:
        #     return MetricType.DRIFT

        raise NotImplementedError(
            f"Unsupported metric '{name}'. metric name must be one of {MetricType.list()}"
        )

    @staticmethod
    def list():
        return list(map(lambda c: c.value, MetricType))


router = APIRouter()


@router.post("/projects/{project}/model-endpoints", response_model=schemas.Endpoint)
def create_endpoint(project: str, endpoint: schemas.Endpoint):
    """
    Creates an endpoint record in KV, also parses labels into searchable KV fields.
    """

    endpoint_id = get_endpoint_id(endpoint)

    if _get_endpoint_kv_record_by_id(endpoint_id):
        url = f"/projects/{endpoint.metadata.project}/model-endpoints/{endpoint_id}"
        raise MLRunConflictError(f"Adding an already-existing ModelEndpoint - {url}")

    logger.info(f"Creating endpoint {endpoint_id} table...")

    get_v3io_client().kv.put(
        container=config.model_endpoint_monitoring_container,
        table_path=ENDPOINTS,
        key=endpoint_id,
        attributes={
            "project": project,
            "function": endpoint.spec.function,
            "model": endpoint.spec.model,
            "tag": endpoint.metadata.tag,
            "model_class": endpoint.spec.model_class,
            "labels": json.dumps(endpoint.metadata.labels),
            **{f"_{k}": v for k, v in endpoint.metadata.labels.items()},
        },
    )

    logger.info(f"Model endpoint {endpoint_id} table created.")

    return endpoint


@router.delete("/projects/{project}/model-endpoints/{endpoint_id}")
def delete_endpoint(project: str, endpoint_id: str):
    """
    Deletes endpoint record from KV by endpoint_id
    """
    logger.info(f"Deleting model endpoint {endpoint_id} table...")
    get_v3io_client().kv.delete(
        container=config.model_endpoint_monitoring_container,
        table_path=ENDPOINTS,
        key=endpoint_id,
    )
    logger.info(f"Model endpoint {endpoint_id} table deleted.")


@router.get(
    "/projects/{project}/model-endpoints", response_model=schemas.EndpointStateList
)
def list_endpoints(
    project: str,
    model: Optional[str] = Query(None),
    function: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    labels: List[str] = Query([], alias="label"),
):
    """
    Returns a list of endpoints of type 'schema.ModelEndpoint', support filtering by model, function, tag and labels.
    Labels are expected to be separated by '&' separator, for example:

    api/projects/{project}/model-endpoints/?label=mylabel==1&myotherlabel==150
    """
    # TODO: call async version of v3io_client
    client = get_v3io_client()
    cursor = client.kv.new_cursor(
        container=config.model_endpoint_monitoring_container,
        table_path=ENDPOINTS,
        attribute_names=[
            "project",
            "model",
            "function",
            "tag",
            "model_class",
            "labels",
            "first_request",
            "last_request",
            "error_count",
            "alert_count",
            "drift_status",
        ],
        filter_expression=_build_filter_expression(
            project, function, model, tag, labels
        ),
    )
    endpoints = cursor.all()

    endpoint_state_list = []
    for endpoint in endpoints:
        # Collect labels (by convention labels are labeled with underscore '_'), ignore builtin '__name' field
        state = schemas.EndpointState(
            endpoint=schemas.Endpoint(
                metadata=schemas.ObjectMetadata(
                    name="",
                    project=endpoint.get("project"),
                    tag=endpoint.get("tag"),
                    labels=json.loads(endpoint.get("labels")),
                    updated=None,
                    uid=None,
                ),
                spec=schemas.EndpointSpec(
                    model=endpoint.get("model"),
                    function=endpoint.get("function"),
                    model_class=endpoint.get("model_class"),
                ),
                status=schemas.ObjectStatus(state="active"),
            ),
            first_request=endpoint.get("first_request"),
            last_request=endpoint.get("last_request"),
            error_count=endpoint.get("error_count"),
            alert_count=endpoint.get("alert_count"),
            drift_status=endpoint.get("drift_status"),
            metrics=None,
            accuracy=None,
        )
        endpoint_state_list.append(state)

    return schemas.EndpointStateList(endpoints=endpoint_state_list)


@router.get(
    "/projects/{project}/model-endpoints/{endpoint_id}",
    response_model=schemas.EndpointState,
)
def get_endpoint(
    project: str,
    endpoint_id: str,
    start: str = Query("now-1h"),
    end: str = Query("now"),
):
    """
    Return the current state of an endpoint, meaning all additional data the is relevant to a specified endpoint.
    This function also takes into account the start and end times and uses the same time-querying as v3io-frames.
    """
    endpoint = _get_endpoint_kv_record_by_id(
        endpoint_id,
        [
            "project",
            "model",
            "function",
            "tag",
            "model_class",
            "labels",
            "first_request",
            "last_request",
            "error_count",
            "alert_count",
            "drift_status",
        ],
    )

    if not endpoint:
        url = f"/projects/{project}/model-endpoints/{endpoint_id}"
        raise MLRunNotFoundError(f"Endpoint {endpoint_id} not found - {url}")

    return schemas.EndpointState(
        endpoint=schemas.Endpoint(
            metadata=schemas.ObjectMetadata(
                name="",
                project=endpoint.get("project"),
                tag=endpoint.get("tag"),
                labels=json.loads(endpoint.get("labels")),
                updated=None,
                uid=None,
            ),
            spec=schemas.EndpointSpec(
                model=endpoint.get("model"),
                function=endpoint.get("function"),
                model_class=endpoint.get("model_class"),
            ),
            status=schemas.ObjectStatus(state="active"),
        ),
        first_request=endpoint.get("first_request"),
        last_request=endpoint.get("last_request"),
        error_count=endpoint.get("error_count"),
        alert_count=endpoint.get("alert_count"),
        drift_status=endpoint.get("drift_status"),
        metrics=None,
        feature_details=None,
    )


@router.get(
    "/projects/{project}/model-endpoints/{endpoint_id}/metrics",
    response_model=schemas.MetricList,
)
def get_endpoint_metrics(
    project: str,
    endpoint_id: str,
    start: str = Query("now-1h"),
    end: str = Query("now"),
    name: List[str] = Query([]),
):
    if _get_endpoint_kv_record_by_id(endpoint_id):
        url = f"projects/{project}/model-endpoints/{endpoint_id}/metrics"
        raise MLRunNotFoundError(f"Endpoint not found' - {url}")

    try:
        metrics = [MetricType.from_string(n) for n in name]
    except NotImplementedError as e:
        raise MLRunInvalidArgumentError(str(e))

    data = get_frames_client(container="monitoring").read(
        backend="tsdb",
        table=ENDPOINT_EVENTS,
        columns=[m.value for m in metrics],
        filter=f"endpoint_id=='{endpoint_id}'",
        start=start,
        end=end,
    )

    # TODO: Fix frames client not returning anything
    return schemas.MetricList(metrics=[])


def _decode_labels(labels: Dict[str, Any]) -> List[str]:
    if not labels:
        return []
    return [f"{lbl.lstrip('_')}=={val}" for lbl, val in labels.items()]


def _build_filter_expression(
    project: str,
    function: Optional[str],
    model: Optional[str],
    tag: Optional[str],
    labels: Optional[List[str]],
):
    filter_expression = [f"project=='{project}'"]

    if function:
        filter_expression.append(f"function=='{function}'")
    if model:
        filter_expression.append(f"model=='{model}'")
    if tag:
        filter_expression.append(f"tag=='{tag}'")
    if labels:
        for label in labels:

            if not label.startswith("_"):
                label = f"_{label}"

            if "==" in label:
                lbl, value = list(map(lambda x: x.strip(), label.split("==")))
                filter_expression.append(f"{lbl}=='{value}'")

    return " AND ".join(filter_expression)


def _get_endpoint_kv_record_by_id(
    endpoint_id: str, attribute_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    endpoint = (
        get_v3io_client()
        .kv.get(
            container=config.model_endpoint_monitoring_container,
            table_path=ENDPOINTS,
            key=endpoint_id,
            attribute_names=attribute_names or "*",
            raise_for_status=RaiseForStatus.never,
        )
        .output.item
    )
    return endpoint


def get_endpoint_id(endpoint: schemas.Endpoint) -> str:
    endpoint_unique_string = (
        f"{endpoint.spec.function}_{endpoint.spec.model}_{endpoint.metadata.tag}"
    )
    md5 = hashlib.md5(endpoint_unique_string.encode("utf-8")).hexdigest()
    return f"{endpoint.metadata.project}.{md5}"
