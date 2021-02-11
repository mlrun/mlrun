from http import HTTPStatus
from typing import List, Dict, Any, Callable

from fastapi import APIRouter, Response, Request

from mlrun.api.api.endpoints.model_endpoints import get_access_key, list_endpoints
from mlrun.api.schemas import ModelEndpointStateList, GrafanaTable, GrafanaColumn
from mlrun.config import config
from mlrun.utils.v3io_clients import get_frames_client

router = APIRouter()


def _parse_query_parameters(request_body: Dict[str, Any]) -> Dict[str, str]:
    """
    Looks for target field Grafana's SimpleJson query json body, parses semi-colon separated (;), key-value
    queries.
    """
    targets = request_body.get("targets", [])
    target_obj = targets[0] if targets else {}
    target_query = target_obj.get("target") if target_obj else ""

    if not target_query:
        raise Exception(f"target missing in request body:\n {request_body}")

    parameters = {}
    for query in target_query.split(";"):
        query_parts = query.split("=")
        if len(query_parts) < 2:
            raise Exception(
                f"Query must contain both query key and query value. Expected query_key=query_value, found {query} instead."
            )

    return parameters


def _validate_query_parameters(query_parameters: Dict[str, str]):
    """Validates the parameters sent via Grafana's SimpleJson query"""
    if "target_endpoint" not in query_parameters:
        raise Exception(
            f"Expected 'target_endpoint' field in query, found {query_parameters} instead"
        )
    if query_parameters["target_endpoint"] not in DISPATCH:
        raise Exception(f"{query_parameters['target_endpoint']} unsupported.")


@router.get("/projects/grafana-proxy/model-endpoints", status_code=HTTPStatus.OK.value)
def grafana_proxy_model_endpoints_check_connection(request: Request):
    """
    Root of grafana proxy for the model-endpoints API, used for validating the model-endpoints data source
    connectivity.
    """
    get_access_key(request)
    return Response(status_code=HTTPStatus.OK.value)


@router.post(
    "/projects/grafana-proxy/model-endpoints/query", response_model=List[GrafanaTable]
)
async def grafana_proxy_model_endpoints(request: Request) -> List[GrafanaTable]:
    """

    """
    body = await request.json()
    query_parameters = _parse_query_parameters(body)
    _validate_query_parameters(query_parameters)

    # At this point everything is validated and we can access everything that is needed without performing all previous
    # checks
    target_type = query_parameters["target_endpoint"]
    result = DISPATCH[target_type](request, body, query_parameters)
    return result


def grafana_endpoint_features(
    request: Request, body: Dict[str, Any], query_parameters: Dict[str, str]
):
    endpoint_id = query_parameters.get("endpoint_id")
    project = query_parameters.get("project")

    start = body.get("rangeRaw", {}).get("start", "now-1h")
    end = body.get("rangeRaw", {}).get("end", "now")

    frames_client = get_frames_client(
        token=get_access_key(request),
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


def grafana_list_endpoints(
    request: Request, body: Dict[str, Any], query_parameters: Dict[str, str]
) -> List[GrafanaTable]:
    project = query_parameters.get("project")
    model = query_parameters.get("model", None)
    function = query_parameters.get("function", None)
    tag = query_parameters.get("tag", None)
    labels = query_parameters.get("labels", "")
    labels = labels.split(",") if labels else []

    metrics = query_parameters.get("metrics", "")
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


DISPATCH: Dict[
    str, Callable[[Request, Dict[str, Any], Dict[str, str]], List[GrafanaTable]]
] = {
    "list_endpoints": grafana_list_endpoints,
    "endpoint_features": grafana_endpoint_features,
}
