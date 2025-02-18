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

from sqlalchemy.orm import Session

import mlrun.common.formatters
import mlrun.common.schemas
import mlrun.common.schemas.model_monitoring.grafana as grafana_schemas
from mlrun.errors import MLRunBadRequestError
from mlrun.utils import logger

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
    endpoint_list = await services.api.crud.ModelEndpoints().list_model_endpoints(
        db_session=db_session,
        project=project,
        latest_only=True,
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

    endpoint_list = await services.api.crud.ModelEndpoints().list_model_endpoints(
        db_session=db_session,
        project=project,
        model_name=model,
        function_name=function,
        labels=labels,
        uids=uids,
        tsdb_metrics=True,
        latest_only=True,
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
