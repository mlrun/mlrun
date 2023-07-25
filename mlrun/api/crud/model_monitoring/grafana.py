# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.api.utils.auth.verifier
import mlrun.common.schemas
from mlrun.api.utils.singletons.project_member import get_project_member
from mlrun.common.model_monitoring.helpers import parse_model_endpoint_store_prefix
from mlrun.errors import MLRunBadRequestError
from mlrun.utils import config, logger
from mlrun.utils.v3io_clients import get_frames_client


def grafana_list_projects(
    db_session: Session,
    auth_info: mlrun.common.schemas.AuthInfo,
    query_parameters: Dict[str, str],
) -> List[str]:
    """
    List available project names. Will be used as a filter in each grafana dashboard.

    :param db_session:        A session that manages the current dialog with the database.
    :param auth_info:         The auth info of the request.
    :param query_parameters:  Dictionary of query parameters attached to the request. Note that this parameter is
                              required by the API even though it is not being used in this function.

    :return: List of available project names.
    """

    projects_output = get_project_member().list_projects(
        db_session,
        format_=mlrun.common.schemas.ProjectsFormat.name_only,
        leader_session=auth_info.session,
    )
    return projects_output.projects


# The following functions were not removed due to backward compatibility that is related to iguazio version <= 3.5.2


async def grafana_list_endpoints(
    body: Dict[str, Any],
    query_parameters: Dict[str, str],
    auth_info: mlrun.common.schemas.AuthInfo,
) -> List[mlrun.common.schemas.model_monitoring.grafana.GrafanaTable]:
    project = query_parameters.get("project")

    # Filters
    model = query_parameters.get("model", None)
    function = query_parameters.get("function", None)
    labels = query_parameters.get("labels", "")
    labels = labels.split(",") if labels else []

    # Metrics to include
    metrics = query_parameters.get("metrics", "")
    metrics = metrics.split(",") if metrics else []

    # Time range for metrics
    start = body.get("rangeRaw", {}).get("start", "now-1h")
    end = body.get("rangeRaw", {}).get("end", "now")

    # Endpoint type filter - will be used to filter the router models
    filter_router = query_parameters.get("filter_router", None)

    if project:
        await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
            project,
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    endpoint_list = await run_in_threadpool(
        mlrun.api.crud.ModelEndpoints().list_model_endpoints,
        auth_info=auth_info,
        project=project,
        model=model,
        function=function,
        labels=labels,
        metrics=metrics,
        start=start,
        end=end,
    )
    allowed_endpoints = await mlrun.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.model_endpoint,
        endpoint_list.endpoints,
        lambda _endpoint: (
            _endpoint.metadata.project,
            _endpoint.metadata.uid,
        ),
        auth_info,
    )
    endpoint_list.endpoints = allowed_endpoints

    columns = [
        mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
            text="endpoint_id", type="string"
        ),
        mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
            text="endpoint_function", type="string"
        ),
        mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
            text="endpoint_model", type="string"
        ),
        mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
            text="endpoint_model_class", type="string"
        ),
        mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
            text="first_request", type="time"
        ),
        mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
            text="last_request", type="time"
        ),
        mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
            text="accuracy", type="number"
        ),
        mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
            text="error_count", type="number"
        ),
        mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
            text="drift_status", type="number"
        ),
        mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
            text="predictions_per_second", type="number"
        ),
        mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
            text="latency_avg_1h", type="number"
        ),
    ]

    table = mlrun.common.schemas.model_monitoring.grafana.GrafanaTable(columns=columns)
    for endpoint in endpoint_list.endpoints:
        if (
            filter_router
            and endpoint.status.endpoint_type
            == mlrun.common.schemas.model_monitoring.EndpointType.ROUTER
        ):
            continue
        row = [
            endpoint.metadata.uid,
            endpoint.spec.function_uri,
            endpoint.spec.model,
            endpoint.spec.model_class,
            endpoint.status.first_request,
            endpoint.status.last_request,
            "N/A",  # Leaving here for backwards compatibility
            endpoint.status.error_count,
            endpoint.status.drift_status,
        ]

        if (
            endpoint.status.metrics
            and mlrun.common.schemas.model_monitoring.EventKeyMetrics.GENERIC
            in endpoint.status.metrics
        ):
            row.extend(
                [
                    endpoint.status.metrics[
                        mlrun.common.schemas.model_monitoring.EventKeyMetrics.GENERIC
                    ][
                        mlrun.common.schemas.model_monitoring.EventLiveStats.PREDICTIONS_PER_SECOND
                    ],
                    endpoint.status.metrics[
                        mlrun.common.schemas.model_monitoring.EventKeyMetrics.GENERIC
                    ][
                        mlrun.common.schemas.model_monitoring.EventLiveStats.LATENCY_AVG_1H
                    ],
                ]
            )

        table.add_row(*row)

    return [table]


async def grafana_individual_feature_analysis(
    body: Dict[str, Any],
    query_parameters: Dict[str, str],
    auth_info: mlrun.common.schemas.AuthInfo,
):
    endpoint_id = query_parameters.get("endpoint_id")
    project = query_parameters.get("project")
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.model_endpoint,
        project,
        endpoint_id,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )

    endpoint = await run_in_threadpool(
        mlrun.api.crud.ModelEndpoints().get_model_endpoint,
        auth_info=auth_info,
        project=project,
        endpoint_id=endpoint_id,
        feature_analysis=True,
    )

    # Load JSON data from KV, make sure not to fail if a field is missing
    feature_stats = endpoint.status.feature_stats or {}
    current_stats = endpoint.status.current_stats or {}
    drift_measures = endpoint.status.drift_measures or {}

    table = mlrun.common.schemas.model_monitoring.grafana.GrafanaTable(
        columns=[
            mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
                text="feature_name", type="string"
            ),
            mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
                text="actual_min", type="number"
            ),
            mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
                text="actual_mean", type="number"
            ),
            mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
                text="actual_max", type="number"
            ),
            mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
                text="expected_min", type="number"
            ),
            mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
                text="expected_mean", type="number"
            ),
            mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
                text="expected_max", type="number"
            ),
            mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
                text="tvd", type="number"
            ),
            mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
                text="hellinger", type="number"
            ),
            mlrun.common.schemas.model_monitoring.grafana.GrafanaColumn(
                text="kld", type="number"
            ),
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


async def grafana_overall_feature_analysis(
    body: Dict[str, Any],
    query_parameters: Dict[str, str],
    auth_info: mlrun.common.schemas.AuthInfo,
):
    endpoint_id = query_parameters.get("endpoint_id")
    project = query_parameters.get("project")
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.model_endpoint,
        project,
        endpoint_id,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    endpoint = await run_in_threadpool(
        mlrun.api.crud.ModelEndpoints().get_model_endpoint,
        auth_info=auth_info,
        project=project,
        endpoint_id=endpoint_id,
        feature_analysis=True,
    )

    table = mlrun.common.schemas.model_monitoring.grafana.GrafanaTable(
        columns=[
            mlrun.common.schemas.model_monitoring.grafana.GrafanaNumberColumn(
                text="tvd_sum"
            ),
            mlrun.common.schemas.model_monitoring.grafana.GrafanaNumberColumn(
                text="tvd_mean"
            ),
            mlrun.common.schemas.model_monitoring.grafana.GrafanaNumberColumn(
                text="hellinger_sum"
            ),
            mlrun.common.schemas.model_monitoring.grafana.GrafanaNumberColumn(
                text="hellinger_mean"
            ),
            mlrun.common.schemas.model_monitoring.grafana.GrafanaNumberColumn(
                text="kld_sum"
            ),
            mlrun.common.schemas.model_monitoring.grafana.GrafanaNumberColumn(
                text="kld_mean"
            ),
        ]
    )

    if endpoint.status.drift_measures:
        table.add_row(
            endpoint.status.drift_measures.get("tvd_sum"),
            endpoint.status.drift_measures.get("tvd_mean"),
            endpoint.status.drift_measures.get("hellinger_sum"),
            endpoint.status.drift_measures.get("hellinger_mean"),
            endpoint.status.drift_measures.get("kld_sum"),
            endpoint.status.drift_measures.get("kld_mean"),
        )

    return [table]


async def grafana_incoming_features(
    body: Dict[str, Any],
    query_parameters: Dict[str, str],
    auth_info: mlrun.common.schemas.AuthInfo,
):
    endpoint_id = query_parameters.get("endpoint_id")
    project = query_parameters.get("project")
    start = body.get("rangeRaw", {}).get("from", "now-1h")
    end = body.get("rangeRaw", {}).get("to", "now")

    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.model_endpoint,
        project,
        endpoint_id,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )

    endpoint = await run_in_threadpool(
        mlrun.api.crud.ModelEndpoints().get_model_endpoint,
        auth_info=auth_info,
        project=project,
        endpoint_id=endpoint_id,
    )

    time_series = []

    feature_names = endpoint.spec.feature_names

    if not feature_names:
        logger.warn(
            "'feature_names' is either missing or not initialized in endpoint record",
            endpoint_id=endpoint.metadata.uid,
        )
        return time_series

    path = config.model_endpoint_monitoring.store_prefixes.default.format(
        project=project, kind=mlrun.common.schemas.ModelMonitoringStoreKinds.EVENTS
    )
    _, container, path = parse_model_endpoint_store_prefix(path)

    client = get_frames_client(
        token=auth_info.data_session,
        address=config.v3io_framesd,
        container=container,
    )

    data: pd.DataFrame = await run_in_threadpool(
        client.read,
        backend="tsdb",
        table=path,
        columns=feature_names,
        filter=f"endpoint_id=='{endpoint_id}'",
        start=start,
        end=end,
    )

    data.drop(["endpoint_id"], axis=1, inplace=True, errors="ignore")
    data.index = data.index.astype(np.int64) // 10**6

    for feature, indexed_values in data.to_dict().items():
        target = mlrun.common.schemas.model_monitoring.grafana.GrafanaTimeSeriesTarget(
            target=feature
        )
        for index, value in indexed_values.items():
            data_point = mlrun.common.schemas.model_monitoring.grafana.GrafanaDataPoint(
                value=float(value), timestamp=index
            )
            target.add_data_point(data_point)
        time_series.append(target)

    return time_series


def parse_query_parameters(request_body: Dict[str, Any]) -> Dict[str, str]:
    """
    This function searches for the target field in Grafana's SimpleJson json. Once located, the target string is
    parsed by splitting on semi-colons (;). Each part in the resulting list is then split by an equal sign (=) to be
    read as key-value pairs.
    """

    # Try to get the target
    targets = request_body.get("targets", [])

    if len(targets) > 1:
        logger.warn(
            f"The 'targets' list contains more than one element ({len(targets)}), all targets except the first one are "
            f"ignored."
        )

    target_obj = targets[0] if targets else {}
    target_query = target_obj.get("target") if target_obj else ""

    if not target_query:
        raise MLRunBadRequestError(f"Target missing in request body:\n {request_body}")

    parameters = _parse_parameters(target_query)

    return parameters


def parse_search_parameters(request_body: Dict[str, Any]) -> Dict[str, str]:
    """
    This function searches for the target field in Grafana's SimpleJson json. Once located, the target string is
    parsed by splitting on semi-colons (;). Each part in the resulting list is then split by an equal sign (=) to be
    read as key-value pairs.
    """

    # Try to get the target
    target = request_body.get("target")

    if not target:
        raise MLRunBadRequestError(f"Target missing in request body:\n {request_body}")

    parameters = _parse_parameters(target)

    return parameters


def _parse_parameters(target_query):
    parameters = {}
    for query in filter(lambda q: q, target_query.split(";")):
        query_parts = query.split("=")
        if len(query_parts) < 2:
            raise MLRunBadRequestError(
                f"Query must contain both query key and query value. Expected query_key=query_value, found {query} "
                f"instead."
            )
        parameters[query_parts[0]] = query_parts[1]
    return parameters


def drop_grafana_escape_chars(query_parameters: Dict[str, str]):
    query_parameters = dict(query_parameters)
    endpoint_id = query_parameters.get("endpoint_id")
    if endpoint_id is not None:
        query_parameters["endpoint_id"] = endpoint_id.replace("\\", "")
    return query_parameters


def validate_query_parameters(
    query_parameters: Dict[str, str], supported_endpoints: Optional[Set[str]] = None
):
    """Validates the parameters sent via Grafana's SimpleJson query"""
    if "target_endpoint" not in query_parameters:
        raise MLRunBadRequestError(
            f"Expected 'target_endpoint' field in query, found {query_parameters} instead"
        )

    if (
        supported_endpoints is not None
        and query_parameters["target_endpoint"] not in supported_endpoints
    ):
        raise MLRunBadRequestError(
            f"{query_parameters['target_endpoint']} unsupported in query parameters: {query_parameters}. "
            f"Currently supports: {','.join(supported_endpoints)}"
        )
