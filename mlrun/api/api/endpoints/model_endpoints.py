import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import List, Optional, Dict, Any

import pandas as pd
from fastapi import APIRouter, Query, Response, Request
from v3io.dataplane import RaiseForStatus

from mlrun.api.schemas import (
    ModelEndpointMetadata,
    ModelEndpointSpec,
    ModelEndpoint,
    ModelEndpointStateList,
    ModelEndpointState,
    Features,
    Metric,
    ObjectStatus,
    GrafanaTable,
    GrafanaColumn,
)
from mlrun.config import config
from mlrun.errors import (
    MLRunConflictError,
    MLRunNotFoundError,
    MLRunInvalidArgumentError,
    MLRunBadRequestError,
)
from mlrun.utils.helpers import logger
from mlrun.utils.v3io_clients import get_v3io_client, get_frames_client

ENDPOINTS_TABLE_PATH = "model-endpoints"
ENDPOINT_EVENTS_TABLE_PATH = "endpoint-events"
ENDPOINT_TABLE_ATTRIBUTES = [
    "project",
    "model",
    "function",
    "tag",
    "model_class",
    "labels",
    "first_request",
    "last_request",
    "error_count",
    "drift_status",
]


router = APIRouter()


@dataclass
class TimeMetric:
    tsdb_column: str
    metric_name: str
    headers: List[str]

    def transform_df_to_metric(self, data: pd.DataFrame) -> Optional[Metric]:
        if data.empty or self.tsdb_column not in data.columns:
            return None

        values = data[self.tsdb_column].reset_index().to_numpy()
        describe = data[self.tsdb_column].describe().to_dict()

        return Metric(
            name=self.metric_name,
            start_timestamp=str(data.index[0]),
            end_timestamp=str(data.index[-1]),
            headers=self.headers,
            values=[(str(timestamp), float(value)) for timestamp, value in values],
            min=describe["min"],
            avg=describe["mean"],
            max=describe["max"],
        )

    @classmethod
    def from_string(cls, name: str) -> "TimeMetric":
        if name in {"microsec", "latency"}:
            return cls(
                tsdb_column="latency_avg_1s",
                metric_name="average_latency",
                headers=["timestamp", "average"],
            )
        elif name in {"preds", "predictions"}:
            return cls(
                tsdb_column="predictions_per_second_count_1s",
                metric_name="predictions_per_second",
                headers=["timestamp", "count"],
            )
        else:
            raise NotImplementedError(f"Unsupported metric '{name}'")


@router.get("/projects/grafana-proxy/model-endpoints", status_code=HTTPStatus.OK.value)
def grafana_list_endpoints(request: Request):
    _get_access_key(request)
    return Response(status_code=HTTPStatus.OK.value)


@router.post(
    "/projects/grafana-proxy/model-endpoints/query", response_model=List[GrafanaTable]
)
async def grafana_list_endpoints(request: Request):
    body = await request.json()
    targets = body.get("targets", [])
    target = targets[0] if targets else {}
    target = target.get("target") if target else {}

    if not target:
        raise Exception(f"target missing in request body:\n {body}")

    parameters = dict(t.split("=") for t in target.split(";"))
    project = parameters.get("project")
    model = parameters.get("model", None)
    function = parameters.get("function", None)
    tag = parameters.get("tag", None)
    labels = parameters.get("labels", "")
    labels = labels.split(",") if labels else []

    metrics = parameters.get("metrics", "")
    metrics = metrics.split(",") if metrics else []

    start = body.get("rangeRaw", {}).get("start", "now-1h")
    end = body.get("rangeRaw", {}).get("end", "now")

    endpoint_list: ModelEndpointStateList = list_endpoints(
        request, project, model, function, tag, labels, start, end, metrics,
    )

    columns = [
        GrafanaColumn(text="endpoint_id", type="string"),
        GrafanaColumn(text="endpoint_function", type="string"),
        GrafanaColumn(text="endpoint_model", type="string"),
        GrafanaColumn(text="endpoint_model_class", type="string"),
        GrafanaColumn(text="endpoint_tag", type="string"),
        GrafanaColumn(text="first_request", type="time"),
        GrafanaColumn(text="last_request", type="time"),
        GrafanaColumn(text="accuracy", type="number"),
        GrafanaColumn(text="error_count", type="number"),
        GrafanaColumn(text="drift_status", type="number"),
    ]

    metric_columns = []

    found_metrics = set()
    for endpoint_state in endpoint_list.endpoints:
        if endpoint_state.metrics:
            for key in endpoint_state.metrics.keys():
                if key not in found_metrics:
                    found_metrics.add(key)
                    metric_columns.append(GrafanaColumn(text=key, type="number"))

    columns = columns + metric_columns

    rows = []
    for endpoint_state in endpoint_list.endpoints:
        row = [
            endpoint_state.endpoint.id,
            endpoint_state.endpoint.spec.function,
            endpoint_state.endpoint.spec.model,
            endpoint_state.endpoint.spec.model_class,
            endpoint_state.endpoint.metadata.tag,
            endpoint_state.first_request,
            endpoint_state.last_request,
            endpoint_state.accuracy,
            endpoint_state.error_count,
            endpoint_state.drift_status,
        ]

        if metric_columns and endpoint_state.metrics:
            for metric_column in metric_columns:
                row.append(endpoint_state.metrics[metric_column.text])

        rows.append(row)

    return [GrafanaTable(columns=columns, rows=rows)]


@router.get(
    "/projects/grafana-proxy/endpoint-features", status_code=HTTPStatus.OK.value
)
def grafana_endpoint_features(request: Request):
    _get_access_key(request)
    return Response(status_code=HTTPStatus.OK.value)


@router.post(
    "/projects/grafana-proxy/endpoint-features/query", response_model=List[GrafanaTable]
)
async def grafana_endpoint_features(request: Request):
    body = await request.json()
    targets = body.get("targets", [])
    target = targets[0] if targets else {}
    target = target.get("target") if target else {}

    if not target:
        raise Exception(f"target missing in request body:\n {body}")

    parameters = dict(t.split("=") for t in target.split(";"))
    endpoint_id = parameters.get("endpoint_id")
    project = parameters.get("project")

    start = body.get("rangeRaw", {}).get("start", "now-1h")
    end = body.get("rangeRaw", {}).get("end", "now")

    frames_client = get_frames_client(
        token=_get_access_key(request),
        address=config.v3io_framesd,
        container=config.model_endpoint_monitoring.container,
    )

    # v3io_client = get_v3io_client(endpoint=config.v3io_api)
    # features = v3io_client.kv.get(
    #     container="projects",
    #     table_path=f"{project}/model-describe",
    #     key="endpoint_id"
    # ).output.item
    #
    # if features:
    #     features = features.get("features", {})

    results = frames_client.read(
        "tsdb",
        "test/endpoint-features",
        filter=f"endpoint_id=='{endpoint_id}'",
        start=start,
        end=end,
    )

    if results.empty:
        return []

    results.drop(["endpoint_id", "prediction"], inplace=True, axis=1)

    columns = [
        GrafanaColumn(text="feature_name", type="string"),
        GrafanaColumn(text="actual_min", type="number"),
        GrafanaColumn(text="actual_mean", type="number"),
        GrafanaColumn(text="actual_max", type="number"),
        GrafanaColumn(text="expected_min", type="number"),
        GrafanaColumn(text="expected_mean", type="number"),
        GrafanaColumn(text="expected_max", type="number"),
    ]

    rows = []
    if not results.empty:
        describes = results.describe().to_dict()
        for feature, describe in describes.items():
            rows.append(
                [
                    feature,
                    describe["min"],
                    describe["mean"],
                    describe["max"],
                    None,  # features.get(feature, {}).get("min", None),
                    None,  # features.get(feature, {}).get("mean", None),
                    None,  # features.get(feature, {}).get("max", None),
                ]
            )

    return [GrafanaTable(columns=columns, rows=rows)]


@router.post(
    "/projects/{project}/model-endpoints/{endpoint_id}/clear",
    status_code=HTTPStatus.NO_CONTENT.value,
)
def clear_endpoint_record(request: Request, project: str, endpoint_id: str):
    """
    Clears endpoint record from KV by endpoint_id
    """

    _verify_endpoint(project, endpoint_id)

    logger.info("Clearing model endpoint table", endpoint_id=endpoint_id)
    client = get_v3io_client(endpoint=config.v3io_api)
    client.kv.delete(
        container=config.model_endpoint_monitoring.container,
        table_path=f"{project}/{ENDPOINTS_TABLE_PATH}",
        key=endpoint_id,
        access_key=_get_access_key(request),
    )
    logger.info("Model endpoint table deleted", endpoint_id=endpoint_id)

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
    access_key = _get_access_key(request)
    client = get_v3io_client(endpoint=config.v3io_api)
    cursor = client.kv.new_cursor(
        container=config.model_endpoint_monitoring.container,
        table_path=f"{project}/{ENDPOINTS_TABLE_PATH}",
        access_key=access_key,
        attribute_names=ENDPOINT_TABLE_ATTRIBUTES,
        filter_expression=_build_kv_cursor_filter_expression(
            project, function, model, tag, labels
        ),
    )
    endpoints = cursor.all()

    endpoint_state_list = []
    for endpoint in endpoints:
        endpoint_metrics = {}
        if metrics:
            endpoint_metrics = _get_endpoint_metrics(
                access_key=access_key,
                project=project,
                endpoint_id=endpoint.get("id"),
                name=metrics,
                start=start,
                end=end,
            )

        # Collect labels (by convention labels are labeled with underscore '_'), ignore builtin '__name' field
        state = ModelEndpointState(
            endpoint=ModelEndpoint(
                metadata=ModelEndpointMetadata(
                    project=endpoint.get("project"),
                    tag=endpoint.get("tag"),
                    labels=json.loads(endpoint.get("labels")),
                ),
                spec=ModelEndpointSpec(
                    model=endpoint.get("model"),
                    function=endpoint.get("function"),
                    model_class=endpoint.get("model_class"),
                ),
                status=ObjectStatus(state="active"),
            ),
            first_request=endpoint.get("first_request"),
            last_request=endpoint.get("last_request"),
            error_count=endpoint.get("error_count"),
            drift_status=endpoint.get("drift_status"),
            metrics=endpoint_metrics,
        )
        endpoint_state_list.append(state)

    return ModelEndpointStateList(endpoints=endpoint_state_list)


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
    """
    Return the current state of an endpoint, meaning all additional data the is relevant to a specified endpoint.
    This function also takes into account the start and end times and uses the same time-querying as v3io-frames.
    """

    _verify_endpoint(project, endpoint_id)
    access_key = _get_access_key(request)
    endpoint = get_endpoint_kv_record_by_id(
        access_key, project, endpoint_id, ENDPOINT_TABLE_ATTRIBUTES,
    )

    if not endpoint:
        url = f"/projects/{project}/model-endpoints/{endpoint_id}"
        raise MLRunNotFoundError(f"Endpoint {endpoint_id} not found - {url}")

    endpoint_metrics = None
    if metrics:
        endpoint_metrics = _get_endpoint_metrics(
            access_key=access_key,
            project=project,
            endpoint_id=endpoint_id,
            start=start,
            end=end,
            name=metrics,
        )

    # endpoint_features = None
    # if features:
    #     endpoint_features = _get_endpoint_features(
    #         project=project, endpoint_id=endpoint_id, features=endpoint.get("features")
    #     )

    return ModelEndpointState(
        endpoint=ModelEndpoint(
            metadata=ModelEndpointMetadata(
                project=endpoint.get("project"),
                tag=endpoint.get("tag"),
                labels=json.loads(endpoint.get("labels", "")),
            ),
            spec=ModelEndpointSpec(
                model=endpoint.get("model"),
                function=endpoint.get("function"),
                model_class=endpoint.get("model_class"),
            ),
            status=ObjectStatus(state="active"),
        ),
        first_request=endpoint.get("first_request"),
        last_request=endpoint.get("last_request"),
        error_count=endpoint.get("error_count"),
        drift_status=endpoint.get("drift_status"),
        metrics=endpoint_metrics,
        features=[],
    )


def _get_endpoint_metrics(
    access_key: str,
    project: str,
    endpoint_id: str,
    name: List[str],
    start: str = "now-1h",
    end: str = "now",
) -> Dict[str, Metric]:

    if not name:
        raise MLRunInvalidArgumentError("Metric names must be provided")

    try:
        metrics = [TimeMetric.from_string(n) for n in name]
    except NotImplementedError as e:
        raise MLRunInvalidArgumentError(str(e))

    # Columns must have at least an endpoint_id attribute for frames' filter expression
    columns = ["endpoint_id"]

    for metric in metrics:
        columns.append(metric.tsdb_column)

    client = get_frames_client(
        token=access_key,
        address=config.v3io_framesd,
        container=config.model_endpoint_monitoring.container,
    )

    data = client.read(
        backend="tsdb",
        table=f"{project}/{ENDPOINT_EVENTS_TABLE_PATH}",
        columns=columns,
        filter=f"endpoint_id=='{endpoint_id}'",
        start=start,
        end=end,
    )

    metrics = [time_metric.transform_df_to_metric(data) for time_metric in metrics]
    metrics = [metric for metric in metrics if metric is not None]
    metrics = {metric.name: metric for metric in metrics}
    return metrics


def _get_endpoint_features(
    project: str, endpoint_id: str, features: Optional[str]
) -> List[Features]:
    if not features:
        url = f"projects/{project}/model-endpoints/{endpoint_id}/features"
        raise MLRunNotFoundError(f"Endpoint features not found' - {url}")

    parsed_features: List[dict] = json.loads(features) if features is not None else []
    feature_objects = [Features(**feature) for feature in parsed_features]
    return feature_objects


def _decode_labels(labels: Dict[str, Any]) -> List[str]:
    if not labels:
        return []
    return [f"{lbl.lstrip('_')}=={val}" for lbl, val in labels.items()]


def _build_kv_cursor_filter_expression(
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

            if "=" in label:
                lbl, value = list(map(lambda x: x.strip(), label.split("=")))
                filter_expression.append(f"{lbl}=='{value}'")
            else:
                filter_expression.append(f"exists({label})")

    return " AND ".join(filter_expression)


def get_endpoint_kv_record_by_id(
    access_key: str,
    project: str,
    endpoint_id: str,
    attribute_names: Optional[List[str]] = None,
) -> Dict[str, Any]:

    client = get_v3io_client(endpoint=config.v3io_api)

    endpoint = client.kv.get(
        container=config.model_endpoint_monitoring.container,
        table_path=f"{project}/{ENDPOINTS_TABLE_PATH}",
        key=endpoint_id,
        access_key=access_key,
        attribute_names=attribute_names or "*",
        raise_for_status=RaiseForStatus.never,
    ).output.item
    return endpoint


def _verify_endpoint(project, endpoint_id):
    endpoint_id_project, _ = endpoint_id.split(".")
    if endpoint_id_project != project:
        raise MLRunConflictError(
            f"project: {project} and endpoint_id: {endpoint_id} missmatch."
        )


def _get_access_key(_request: Request):
    access_key = _request.headers.get("X-V3io-Session-Key")
    if not access_key:
        raise MLRunBadRequestError(
            "Request header missing 'X-V3io-Session-Key' parameter."
        )
    return access_key
