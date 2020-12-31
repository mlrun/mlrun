import hashlib
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import pandas as pd
from fastapi import APIRouter, Query
from v3io.dataplane import RaiseForStatus

from mlrun.api import schemas
from mlrun.config import config
from mlrun.errors import (
    MLRunConflictError,
    MLRunNotFoundError,
    MLRunInvalidArgumentError,
)
from mlrun.utils import logger
from mlrun.utils.v3io_clients import get_v3io_client, get_frames_client

ENDPOINTS = "monitoring/endpoints"
ENDPOINT_EVENTS = "monitoring/endpoint_events"


@dataclass
class TimeMetric:
    tsdb_column: str
    metric_name: str
    headers: List[str]

    def to_metric(self, data: pd.DataFrame) -> Optional[schemas.Metric]:
        if data.empty or self.tsdb_column not in data.columns:
            return None

        values = data[self.tsdb_column].reset_index().to_numpy()
        describe = data[self.tsdb_column].describe().to_dict()

        return schemas.Metric(
            name=self.metric_name,
            start_timestamp=str(data.index[0]),
            end_timestamp=str(data.index[-1]),
            headers=self.headers,
            values=[(str(timestamp), float(value)) for timestamp, value in values],
            min=describe["min"],
            avg=describe["mean"],
            max=describe["max"],
        )

    @staticmethod
    def from_string(name) -> "TimeMetric":
        if name in {"microsec", "latency"}:
            return TimeMetric(
                tsdb_column="latency_avg_1s",
                metric_name="average_latency",
                headers=["timestamp", "average"],
            )
        elif name in {"preds", "predictions"}:
            return TimeMetric(
                tsdb_column="predictions_per_second_count_1s",
                metric_name="predictions_per_second",
                headers=["timestamp", "count"],
            )
        else:
            raise NotImplementedError(f"Unsupported metric '{name}'")


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
    start: str = Query(default="now-1h"),
    end: str = Query(default="now"),
    metrics: bool = Query(default=False),
    features: bool = Query(default=False),
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
            "features"
        ],
    )

    if not endpoint:
        url = f"/projects/{project}/model-endpoints/{endpoint_id}"
        raise MLRunNotFoundError(f"Endpoint {endpoint_id} not found - {url}")

    endpoint_metrics = None
    if metrics:
        endpoint_metrics = _get_endpoint_metrics(
            endpoint_id=endpoint_id,
            start=start,
            end=end,
            name=["predictions", "latency"],
        )

    endpoint_features = None
    if features:
        endpoint_features = _get_endpoint_features(
            project=project, endpoint_id=endpoint_id, features=endpoint.get("features")
        )

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
        metrics=endpoint_metrics,
        features=endpoint_features,
    )


@router.get(
    "/projects/{project}/model-endpoints/{endpoint_id}/metrics",
    response_model=schemas.MetricList,
)
def get_endpoint_metrics(
    project: str,
    endpoint_id: str,
    name: List[str] = Query([]),
    start: str = Query("now-1h"),
    end: str = Query("now"),
):
    if not _get_endpoint_kv_record_by_id(endpoint_id):
        url = f"projects/{project}/model-endpoints/{endpoint_id}/metrics"
        raise MLRunNotFoundError(f"Endpoint not found' - {url}")
    return _get_endpoint_metrics(endpoint_id, name, start, end)


def _get_endpoint_metrics(
    endpoint_id: str,
    name: List[str],
    start: str = "now-1h",
    end: str = "now",
) -> schemas.MetricList:
    if not name:
        raise MLRunInvalidArgumentError("Metric names must be provided")

    try:
        metrics = [TimeMetric.from_string(n) for n in name]
    except NotImplementedError as e:
        raise MLRunInvalidArgumentError(str(e))

    columns = ["endpoint_id"]

    for metric in metrics:
        columns.append(metric.tsdb_column)

    data = get_frames_client(container=config.model_endpoint_monitoring_container).read(
        backend="tsdb",
        table=ENDPOINT_EVENTS,
        columns=columns,
        filter=f"endpoint_id=='{endpoint_id}'",
        start=start,
        end=end,
    )

    metrics = [time_metric.to_metric(data) for time_metric in metrics]
    metrics = [metric for metric in metrics if metric is not None]
    return schemas.MetricList(metrics=metrics)


def _get_endpoint_features(
    project: str, endpoint_id: str, features: Optional[str]
) -> schemas.FeatureList:
    if not features:
        url = f"projects/{project}/model-endpoints/{endpoint_id}/features"
        raise MLRunNotFoundError(f"Endpoint features not found' - {url}")

    features = json.loads(features)
    features = [schemas.Features(**feature) for feature in features]
    return schemas.FeatureList(features=features)


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
