import json
from http import HTTPStatus
from typing import List, Dict, Any, Callable, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Response, Request

from mlrun.api.crud.model_endpoints import (
    ModelEndpoints,
    get_access_key,
    deserialize_endpoint_from_kv,
    ENDPOINT_EVENTS_TABLE_PATH,
    deserialize_endpoint_state_from_kv,
)
from mlrun.api.schemas import (
    GrafanaTable,
    GrafanaColumn,
    GrafanaNumberColumn,
    GrafanaStringColumn,
    GrafanaTimeSeries,
    GrafanaTimeSeriesTarget,
    GrafanaDataPoint,
    Format,
)
from mlrun.db import get_run_db
from mlrun.errors import MLRunBadRequestError
from mlrun.utils import logger, config
from mlrun.utils.v3io_clients import get_frames_client

router = APIRouter()


@router.get("/grafana-proxy/model-endpoints", status_code=HTTPStatus.OK.value)
def grafana_proxy_model_endpoints_check_connection(request: Request):
    """
    Root of grafana proxy for the model-endpoints API, used for validating the model-endpoints data source
    connectivity.
    """
    logger.debug("Querying grafana-proxy / health check")

    get_access_key(request)
    return Response(status_code=HTTPStatus.OK.value)


@router.post("/grafana-proxy/model-endpoints/query")
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

    logger.debug("Querying grafana-proxy / query", **query_parameters)

    _validate_query_parameters(query_parameters)
    query_parameters = _drop_grafana_escape_chars(query_parameters)

    # At this point everything is validated and we can access everything that is needed without performing all previous
    # checks again.
    target_endpoint = query_parameters["target_endpoint"]
    result = NAME_TO_FUNCTION_DICTIONARY[target_endpoint](
        body, query_parameters, access_key
    )
    return result


@router.post("/grafana-proxy/model-endpoints/search")
async def grafana_proxy_model_endpoints_query(request: Request) -> List[GrafanaTable]:
    """
    Query route for model-endpoints grafana proxy API, used for creating an interface between grafana queries and
    model-endpoints logic.

    This implementation requires passing `target_function` query parameter in order to dispatch different
    model-endpoint monitoring functions.
    """
    access_key = get_access_key(request)
    body = await request.json()
    query_parameters = _parse_search_parameters(body)

    logger.debug("Querying grafana-proxy / search", **query_parameters)

    _validate_query_parameters(query_parameters)
    query_parameters = _drop_grafana_escape_chars(query_parameters)

    # At this point everything is validated and we can access everything that is needed without performing all previous
    # checks again.
    target_endpoint = query_parameters["target_endpoint"]
    result = NAME_TO_FUNCTION_DICTIONARY[target_endpoint](
        body, query_parameters, access_key
    )
    return result


def grafana_list_projects(
    body: Dict[str, Any], query_parameters: Dict[str, str], access_key: str
) -> List[GrafanaTable]:
    mldb = get_run_db(config.dbpath)
    table = GrafanaTable(columns=[GrafanaStringColumn(text="project_name")])
    try:
        for project in mldb.list_projects(format_=Format.name_only):
            table.add_row(project)
    except Exception as e:
        logger.error(str(e))
    return [table]


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
        if endpoint_state.metrics is not None:
            for key in endpoint_state.metrics.keys():
                if key not in found_metrics:
                    found_metrics.add(key)
                    metric_columns.append(GrafanaColumn(text=key, type="number"))

    columns = columns + metric_columns
    table = GrafanaTable(columns=columns)

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

        if endpoint_state.metrics is not None and metric_columns:
            for metric_column in metric_columns:
                row.append(endpoint_state.metrics[metric_column.text])

        table.add_row(*row)

    return [table]


def grafana_individual_feature_analysis(
    body: Dict[str, Any], query_parameters: Dict[str, str], access_key: str
):
    endpoint_id = query_parameters.get("endpoint_id")
    project = query_parameters.get("project")

    endpoint_state = deserialize_endpoint_state_from_kv(
        access_key=access_key, project=project, endpoint_id=endpoint_id
    )

    # Load JSON data from KV, make sure not to fail if a field is missing
    feature_stats = endpoint_state.endpoint.metadata.feature_stats or {}
    current_stats = endpoint_state.current_stats or {}
    drift_measures = endpoint_state.drift_measures or {}

    table = GrafanaTable(
        columns=[
            GrafanaColumn(text="feature_name", type="string"),
            GrafanaColumn(text="actual_min", type="number"),
            GrafanaColumn(text="actual_mean", type="number"),
            GrafanaColumn(text="actual_max", type="number"),
            GrafanaColumn(text="expected_min", type="number"),
            GrafanaColumn(text="expected_mean", type="number"),
            GrafanaColumn(text="expected_max", type="number"),
            GrafanaColumn(text="tvd", type="number"),
            GrafanaColumn(text="hellinger", type="number"),
            GrafanaColumn(text="kld", type="number"),
        ]
    )

    for feature, base_stat in feature_stats.items():
        current_stat = current_stats.get(feature, {})
        drift_measure = drift_measures.get(feature, {})

        table.add_row(
            feature,
            current_stat.get("min"),
            current_stat.get("mean"),
            current_stat.get("max"),
            base_stat.get("min"),
            base_stat.get("mean"),
            base_stat.get("max"),
            drift_measure.get("tvd"),
            drift_measure.get("hellinger"),
            drift_measure.get("kld"),
        )

    return [table]


def grafana_overall_feature_analysis(
    body: Dict[str, Any], query_parameters: Dict[str, str], access_key: str
):
    endpoint_id = query_parameters.get("endpoint_id")
    project = query_parameters.get("project")

    endpoint_state = deserialize_endpoint_state_from_kv(
        access_key=access_key, project=project, endpoint_id=endpoint_id
    )

    table = GrafanaTable(
        columns=[
            GrafanaNumberColumn(text="tvd_sum"),
            GrafanaNumberColumn(text="tvd_mean"),
            GrafanaNumberColumn(text="hellinger_sum"),
            GrafanaNumberColumn(text="hellinger_mean"),
            GrafanaNumberColumn(text="kld_sum"),
            GrafanaNumberColumn(text="kld_mean"),
        ]
    )

    if endpoint_state.drift_measures:
        table.add_row(
            endpoint_state.drift_measures.get("tvd_sum"),
            endpoint_state.drift_measures.get("tvd_mean"),
            endpoint_state.drift_measures.get("hellinger_sum"),
            endpoint_state.drift_measures.get("hellinger_mean"),
            endpoint_state.drift_measures.get("kld_sum"),
            endpoint_state.drift_measures.get("kld_mean"),
        )

    return [table]


def grafana_incoming_features(
    body: Dict[str, Any], query_parameters: Dict[str, str], access_key: str
):
    endpoint_id = query_parameters.get("endpoint_id")
    project = query_parameters.get("project")
    start = body.get("rangeRaw", {}).get("from", "now-1h")
    end = body.get("rangeRaw", {}).get("to", "now")

    endpoint = deserialize_endpoint_from_kv(
        access_key=access_key, project=project, endpoint_id=endpoint_id
    )

    time_series = GrafanaTimeSeries()

    feature_names = endpoint.metadata.feature_names

    if not feature_names:
        logger.warn(
            "'feature_names' is either missing or not initialized in endpoint record",
            endpoint_id=endpoint.id,
        )
        return time_series.target_data_points

    client = get_frames_client(
        token=access_key,
        address=config.v3io_framesd,
        container=config.model_endpoint_monitoring.container,
    )

    data: pd.DataFrame = client.read(
        backend="tsdb",
        table=f"{project}/{ENDPOINT_EVENTS_TABLE_PATH}",
        columns=feature_names,
        filter=f"endpoint_id=='{endpoint_id}'",
        start=start,
        end=end,
    )

    data.drop(["endpoint_id"], axis=1, inplace=True, errors="ignore")
    data.index = data.index.astype(np.int64) // 10 ** 6

    for feature, indexed_values in data.to_dict().items():
        target = GrafanaTimeSeriesTarget(target=feature)
        for index, value in indexed_values.items():
            data_point = GrafanaDataPoint(value=float(value), timestamp=index)
            target.add_data_point(data_point)
        time_series.target_data_points.append(target)

    return time_series.target_data_points


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
    for query in filter(lambda q: q, target_query.split(";")):
        query_parts = query.split("=")
        if len(query_parts) < 2:
            raise MLRunBadRequestError(
                f"Query must contain both query key and query value. Expected query_key=query_value, "
                f"found {query} instead."
            )
        parameters[query_parts[0]] = query_parts[1]

    return parameters


def _parse_search_parameters(request_body: Dict[str, Any]) -> Dict[str, str]:
    """
    This function searches for the `target` field in Grafana's `SimpleJson` json. Once located, the target string is
    parsed by splitting on semi-colons (;). Each part in the resulting list is then split by an equal sign (=) to be
    read as key-value pairs.
    """

    # Try to get the `target`
    target = request_body.get("targets")

    if not target:
        raise MLRunBadRequestError(f"Target missing in request body:\n {request_body}")

    parameters = {}
    for query in filter(lambda q: q, target.split(";")):
        query_parts = query.split("=")
        if len(query_parts) < 2:
            raise MLRunBadRequestError(
                f"Query must contain both query key and query value. Expected query_key=query_value, "
                f"found {query} instead."
            )
        parameters[query_parts[0]] = query_parts[1]

    return parameters


def _drop_grafana_escape_chars(query_parameters: Dict[str, str]):
    query_parameters = dict(query_parameters)
    endpoint_id = query_parameters.get("endpoint_id")
    if endpoint_id is not None:
        query_parameters["endpoint_id"] = endpoint_id.replace("\\", "")
    return query_parameters


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


def _json_loads_or_default(string: Optional[str], default: Any):
    if string is None:
        return default
    obj = json.loads(string)
    if not obj:
        return default
    return obj


NAME_TO_FUNCTION_DICTIONARY: Dict[
    str, Callable[[Dict[str, Any], Dict[str, str], str], List[GrafanaTable]]
] = {
    "list_projects": grafana_list_projects,
    "list_endpoints": grafana_list_endpoints,
    "individual_feature_analysis": grafana_individual_feature_analysis,
    "overall_feature_analysis": grafana_overall_feature_analysis,
    "incoming_features": grafana_incoming_features,
}
