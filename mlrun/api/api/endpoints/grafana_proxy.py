from http import HTTPStatus
from typing import List, Dict, Any, Callable

from fastapi import APIRouter, Response, Request

from mlrun.api.crud.model_endpoints import ModelEndpoints, get_access_key
from mlrun.api.schemas import (
    GrafanaTable,
    GrafanaColumn,
    ModelEndpointState,
)
from mlrun.config import config
from mlrun.errors import MLRunBadRequestError
from mlrun.utils.logger import create_logger
from mlrun.utils.v3io_clients import get_frames_client

router = APIRouter()


logger = create_logger(level="debug", name="grafana_proxy")


@router.get("/grafana-proxy/model-endpoints", status_code=HTTPStatus.OK.value)
def grafana_proxy_model_endpoints_check_connection(request: Request):
    """
    Root of grafana proxy for the model-endpoints API, used for validating the model-endpoints data source
    connectivity.
    """
    get_access_key(request)
    return Response(status_code=HTTPStatus.OK.value)


@router.post("/grafana-proxy/model-endpoints/query", response_model=List[GrafanaTable])
async def grafana_proxy_model_endpoints_query(request: Request) -> List[GrafanaTable]:
    """
    Query route for model-endpoints grafana proxy API, used for creating an interface between grafana queries and
    model-endpoints logic.

    This implementation requires passing `target_function` query parameter in order to dispatch different
    model-endpoint monitoring functions.
    """
    access_key = get_access_key(request)
    body = await request.json()
    query_parameters = _parse_query_parameters(body)
    _validate_query_parameters(query_parameters)

    # At this point everything is validated and we can access everything that is needed without performing all previous
    # checks again.
    target_endpoint = query_parameters["target_endpoint"]
    result = NAME_TO_FUNCTION_DICTIONARY[target_endpoint](
        body, query_parameters, access_key
    )
    return result


def grafana_list_endpoints(
    body: Dict[str, Any], query_parameters: Dict[str, str], access_key: str
) -> List[GrafanaTable]:
    project = query_parameters.get("project")

    # Filters
    model = query_parameters.get("model", None)
    function = query_parameters.get("function", None)
    tag = query_parameters.get("tag", None)
    labels = query_parameters.get("labels", "")
    labels = labels.split(",") if labels else []

    # Metrics to include
    metrics = query_parameters.get("metrics", "")
    metrics = metrics.split(",") if metrics else []

    # Time range for metrics
    start = body.get("rangeRaw", {}).get("start", "now-1h")
    end = body.get("rangeRaw", {}).get("end", "now")

    endpoint_list: List[ModelEndpointState] = ModelEndpoints.list_endpoints(
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
    for endpoint_state in endpoint_list:
        if endpoint_state.metrics:
            for key in endpoint_state.metrics.keys():
                if key not in found_metrics:
                    found_metrics.add(key)
                    metric_columns.append(GrafanaColumn(text=key, type="number"))

    columns = columns + metric_columns

    rows = []
    for endpoint_state in endpoint_list:
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


def grafana_endpoint_features(
    body: Dict[str, Any], query_parameters: Dict[str, str], access_key: str
):
    endpoint_id = query_parameters.get("endpoint_id")
    project = query_parameters.get("project")

    start = body.get("rangeRaw", {}).get("start", "now-1h")
    end = body.get("rangeRaw", {}).get("end", "now")

    frames_client = get_frames_client(
        token=access_key,
        address=config.v3io_framesd,
        container=config.model_endpoint_monitoring.container,
    )

    results = frames_client.read(
        "tsdb",
        f"{project}/endpoint-features",
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


def _parse_query_parameters(request_body: Dict[str, Any]) -> Dict[str, str]:
    """
    This function searches for the `target` field in Grafana's `SimpleJson` json. Once located, the target string is
    parsed by splitting on semi-colons (;). Each part in the resulting list is then split by an equal sign (=) to be
    read as key-value pairs.
    """

    # Try to get the `target`
    targets = request_body.get("targets", [])

    if len(targets) > 1:
        logger.warn(
            f"The 'targets' list contains more then one element ({len(targets)}), all targets except the first one are "
            f"ignored."
        )

    target_obj = targets[0] if targets else {}
    target_query = target_obj.get("target") if target_obj else ""

    if not target_query:
        raise MLRunBadRequestError(f"Target missing in request body:\n {request_body}")

    parameters = {}
    for query in target_query.split(";"):
        query_parts = query.split("=")
        if len(query_parts) < 2:
            raise MLRunBadRequestError(
                f"Query must contain both query key and query value. Expected query_key=query_value, "
                f"found {query} instead."
            )
        parameters[query_parts[0]] = query_parts[1]

    return parameters


def _validate_query_parameters(query_parameters: Dict[str, str]):
    """Validates the parameters sent via Grafana's SimpleJson query"""
    if "target_endpoint" not in query_parameters:
        raise MLRunBadRequestError(
            f"Expected 'target_endpoint' field in query, found {query_parameters} instead"
        )
    if query_parameters["target_endpoint"] not in NAME_TO_FUNCTION_DICTIONARY:
        raise MLRunBadRequestError(
            f"{query_parameters['target_endpoint']} unsupported in query parameters: {query_parameters}. "
            f"Currently supports: {','.join(NAME_TO_FUNCTION_DICTIONARY.keys())}"
        )


NAME_TO_FUNCTION_DICTIONARY: Dict[
    str, Callable[[Dict[str, Any], Dict[str, str], str], List[GrafanaTable]]
] = {
    "list_endpoints": grafana_list_endpoints,
    "endpoint_features": grafana_endpoint_features,
}
