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

from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.common.schemas.model_monitoring.grafana as grafana_schemas
from mlrun.common.model_monitoring.helpers import parse_model_endpoint_store_prefix
from mlrun.errors import MLRunBadRequestError
from mlrun.utils import config, logger
from mlrun.utils.v3io_clients import get_frames_client

import framework.utils.auth.verifier
import services.api.crud
from framework.utils.singletons.project_member import get_project_member


def grafana_list_projects(
    query_parameters: dict[str, str],
    auth_info: mlrun.common.schemas.AuthInfo,
    db_session: Session,
) -> list[str]:
    """
    List available project names. Will be used as a filter in each grafana dashboard.

    :param query_parameters:  Dictionary of query parameters attached to the request. Note that this parameter is
                              required by the API even though it is not being used in this function.
    :param auth_info:         The auth info of the request.
    :param db_session:        A session that manages the current dialog with the database.

    :return: List of available project names.
    """

    projects_output = get_project_member().list_projects(
        db_session,
        format_=mlrun.common.formatters.ProjectFormat.name_only,
        leader_session=auth_info.session,
    )
    return projects_output.projects


async def grafana_list_endpoints_uids(
    query_parameters: dict[str, str],
    auth_info: mlrun.common.schemas.AuthInfo,
    db_session: Session,
) -> list[str]:
    """
    List available model endpoint uids. Will be used as a filter in each model endpoint grafana dashboard.

    :param query_parameters:  Dictionary of query parameters attached to the request. Note that this parameter is
                              required by the API even though it is not being used in this function.
    :param auth_info:         The auth info of the request.
    :param db_session:        A session that manages the current dialog with the database.

    :return: List model endpoints uids.
    """

    project = query_parameters.get("project")
    if project:
        await framework.utils.auth.verifier.AuthVerifier().query_project_permissions(
            project,
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    endpoint_list = await run_in_threadpool(
        services.api.crud.ModelEndpoints().list_model_endpoints,
        db_session=db_session,
        project=project,
    )

    return [model_endpoint.metadata.uid for model_endpoint in endpoint_list.endpoints]


async def grafana_list_metrics(
    query_parameters: dict[str, str],
    auth_info: mlrun.common.schemas.AuthInfo,
    db_session: Session,
) -> list[str]:
    """
    List available metrics and results. Will be used as a filter in the application dashboard.

    :param query_parameters:  Dictionary of query parameters attached to the request.
    :param auth_info:         The auth info of the request.
    :param db_session:        A session that manages the current dialog with the database. Note that this parameter is
                              required by the API even though it is not being used in this function.

    :return: List of available metrics and results.
    """

    project = query_parameters.get("project")

    endpoint_id = query_parameters.get("endpoint_id")

    await (
        framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.model_endpoint,
            project,
            endpoint_id,
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    )

    metrics = []

    task_results = await services.api.api.endpoints.model_endpoints._collect_get_metrics_tasks_results(
        endpoint_ids=[endpoint_id], project=project, application_result_types="all"
    )
    for task_result in task_results:
        metrics.extend(task_result)

    return [metric.name for metric in metrics]


async def grafana_list_endpoints(
    body: dict[str, Any],
    query_parameters: dict[str, str],
    auth_info: mlrun.common.schemas.AuthInfo,
    db_session: Session,
) -> list[grafana_schemas.GrafanaTable]:
    project = query_parameters.get("project")
    if project:
        await framework.utils.auth.verifier.AuthVerifier().query_project_permissions(
            project,
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )

    # Filters
    model = query_parameters.get("model", None)
    function = query_parameters.get("function", None)

    uids = (
        query_parameters.get("uids", "").split(",")
        if query_parameters.get("uids")
        else None
    )

    labels = query_parameters.get("labels", "")
    labels = labels.split(",") if labels else []

    # Endpoint type filter - will be used to filter the router models
    filter_router = query_parameters.get("filter_router", None)

    endpoint_list = await run_in_threadpool(
        services.api.crud.ModelEndpoints().list_model_endpoints,
        db_session=db_session,
        project=project,
        model_name=model,
        function_name=function,
        labels=labels,
        uids=uids,
        tsdb_metrics=True,
    )

    allowed_endpoints = await framework.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.model_endpoint,
        endpoint_list.endpoints,
        lambda _endpoint: (
            _endpoint.metadata.project,
            _endpoint.metadata.uid,
        ),
        auth_info,
    )
    endpoint_list.endpoints = allowed_endpoints

    table = grafana_schemas.GrafanaModelEndpointsTable()
    for endpoint in endpoint_list.endpoints:
        if (
            filter_router
            and endpoint.status.endpoint_type
            == mlrun.common.schemas.model_monitoring.EndpointType.ROUTER
        ):
            continue
        row = [
            endpoint.metadata.uid,
            endpoint.metadata.name,
            endpoint.spec.function_name,
            endpoint.spec.model_name,
            endpoint.spec.model_class,
            endpoint.status.error_count,
            endpoint.status.result_status,
            endpoint.status.sampling_percentage,
        ]

        table.add_row(*row)

    return [table]


async def grafana_individual_feature_analysis(
    body: dict[str, Any],
    query_parameters: dict[str, str],
    auth_info: mlrun.common.schemas.AuthInfo,
):
    endpoint_id = query_parameters.get("endpoint_id")
    project = query_parameters.get("project")
    await (
        framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.model_endpoint,
            project,
            endpoint_id,
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    )

    endpoint = await run_in_threadpool(
        services.api.crud.ModelEndpoints().get_model_endpoint,
        project=project,
        endpoint_id=endpoint_id,
        feature_analysis=True,
    )

    # Load JSON data from KV, make sure not to fail if a field is missing
    feature_stats = endpoint.status.feature_stats or {}
    current_stats = endpoint.status.current_stats or {}
    drift_measures = endpoint.status.drift_measures or {}

    table = grafana_schemas.GrafanaTable(
        columns=[
            grafana_schemas.GrafanaColumn(text="feature_name", type="string"),
            grafana_schemas.GrafanaColumn(text="actual_min", type="number"),
            grafana_schemas.GrafanaColumn(text="actual_mean", type="number"),
            grafana_schemas.GrafanaColumn(text="actual_max", type="number"),
            grafana_schemas.GrafanaColumn(text="expected_min", type="number"),
            grafana_schemas.GrafanaColumn(text="expected_mean", type="number"),
            grafana_schemas.GrafanaColumn(text="expected_max", type="number"),
            grafana_schemas.GrafanaColumn(text="tvd", type="number"),
            grafana_schemas.GrafanaColumn(text="hellinger", type="number"),
            grafana_schemas.GrafanaColumn(text="kld", type="number"),
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
    body: dict[str, Any],
    query_parameters: dict[str, str],
    auth_info: mlrun.common.schemas.AuthInfo,
):
    endpoint_id = query_parameters.get("endpoint_id")
    project = query_parameters.get("project")
    await (
        framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.model_endpoint,
            project,
            endpoint_id,
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    )
    endpoint = await run_in_threadpool(
        services.api.crud.ModelEndpoints().get_model_endpoint,
        project=project,
        endpoint_id=endpoint_id,
        feature_analysis=True,
    )

    table = grafana_schemas.GrafanaTable(
        columns=[
            grafana_schemas.GrafanaNumberColumn(text="tvd_sum"),
            grafana_schemas.GrafanaNumberColumn(text="tvd_mean"),
            grafana_schemas.GrafanaNumberColumn(text="hellinger_sum"),
            grafana_schemas.GrafanaNumberColumn(text="hellinger_mean"),
            grafana_schemas.GrafanaNumberColumn(text="kld_sum"),
            grafana_schemas.GrafanaNumberColumn(text="kld_mean"),
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
    body: dict[str, Any],
    query_parameters: dict[str, str],
    auth_info: mlrun.common.schemas.AuthInfo,
):
    endpoint_id = query_parameters.get("endpoint_id")
    project = query_parameters.get("project")
    start = body.get("rangeRaw", {}).get("from", "now-1h")
    end = body.get("rangeRaw", {}).get("to", "now")

    await (
        framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            mlrun.common.schemas.AuthorizationResourceTypes.model_endpoint,
            project,
            endpoint_id,
            mlrun.common.schemas.AuthorizationAction.read,
            auth_info,
        )
    )

    endpoint = await run_in_threadpool(
        services.api.crud.ModelEndpoints().get_model_endpoint,
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
        target = grafana_schemas.GrafanaTimeSeriesTarget(target=feature)
        for index, value in indexed_values.items():
            data_point = grafana_schemas.GrafanaDataPoint(
                value=float(value), timestamp=index
            )
            target.add_data_point(data_point)
        time_series.append(target)

    return time_series


def parse_query_parameters(request_body: dict[str, Any]) -> dict[str, str]:
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


def parse_search_parameters(request_body: dict[str, Any]) -> dict[str, str]:
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


def drop_grafana_escape_chars(query_parameters: dict[str, str]):
    query_parameters = dict(query_parameters)
    endpoint_id = query_parameters.get("endpoint_id")
    if endpoint_id is not None:
        query_parameters["endpoint_id"] = endpoint_id.replace("\\", "")
    return query_parameters


def validate_query_parameters(
    query_parameters: dict[str, str], supported_endpoints: Optional[set[str]] = None
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
