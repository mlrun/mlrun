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

import asyncio
import typing
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import datetime, timedelta
from http import HTTPStatus
from typing import Annotated, Literal, Optional, Union

from fastapi import APIRouter, Depends, Path, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

import mlrun.common.schemas as schemas
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.common.schemas.model_monitoring.model_endpoints as mm_endpoints
import mlrun.model_monitoring
import mlrun.utils.helpers
from mlrun import MLRunInvalidArgumentError
from mlrun.utils import logger

import framework.api.deps
import framework.utils.auth.verifier
import services.api.crud
from framework.api import deps

router = APIRouter(prefix="/projects/{project}/model-endpoints")

ProjectAnnotation = Annotated[str, Path(pattern=mm_constants.PROJECT_PATTERN)]
EndpointIDAnnotation = Annotated[
    str, Path(pattern=mm_constants.MODEL_ENDPOINT_ID_PATTERN)
]


@router.post(
    "",
    status_code=HTTPStatus.CREATED.value,
    response_model=schemas.ModelEndpoint,
)
async def create_model_endpoint(
    model_endpoint: schemas.ModelEndpoint,
    project: ProjectAnnotation,
    creation_strategy: mm_constants.ModelEndpointCreationStrategy,
    auth_info: schemas.AuthInfo = Depends(framework.api.deps.authenticate_request),
    db_session: Session = Depends(framework.api.deps.get_db_session),
) -> schemas.ModelEndpoint:
    """
    Create a new model endpoint record in the DB.
    :param model_endpoint:  The model endpoint object.
    :param project:         The name of the project.
    :param creation_strategy: model endpoint creation strategy :
                            * overwrite - Create a new model endpoint and delete the last old one if it exists.
                            * inplace - Use the existing model endpoint if it already exists.
                            * archive - Preserve the old model endpoint and create a new one,
                            tagging it as the latest.
    :param auth_info:       The auth info of the request.
    :param db_session:      A session that manages the current dialog with the database.

    :return: A Model endpoint object without operative data.
    """
    logger.info(
        "Creating Model Endpoint record",
        model_endpoint_metadata=model_endpoint.metadata,
        creation_strategy=creation_strategy,
    )
    if project != model_endpoint.metadata.project:
        raise MLRunInvalidArgumentError(
            f"Project name in the URL '{project}' does not match the project name in the model endpoint metadata "
            f"'{model_endpoint.metadata.project}'. User is not allowed to create model endpoint in a different project."
        )
    await (
        framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            resource_type=schemas.AuthorizationResourceTypes.model_endpoint,
            project_name=model_endpoint.metadata.project,
            resource_name=model_endpoint.metadata.name,
            action=schemas.AuthorizationAction.store,
            auth_info=auth_info,
        )
    )

    if not model_endpoint.metadata.project or not model_endpoint.metadata.name:
        raise MLRunInvalidArgumentError("Model endpoint must have project and name")

    return await run_in_threadpool(
        services.api.crud.ModelEndpoints().create_model_endpoint,
        db_session=db_session,
        model_endpoint=model_endpoint,
        creation_strategy=creation_strategy,
    )


@router.patch(
    "",
    response_model=schemas.ModelEndpoint,
)
async def patch_model_endpoint(
    project: ProjectAnnotation,
    model_endpoint: schemas.ModelEndpoint,
    attributes_keys: list[str] = Query([], alias="attribute-key"),
    auth_info: schemas.AuthInfo = Depends(framework.api.deps.authenticate_request),
    db_session: Session = Depends(framework.api.deps.get_db_session),
) -> schemas.ModelEndpoint:
    """
    Patch the model endpoint record in the DB.
    :param project:         The name of the project.
    :param model_endpoint:  The model endpoint object.
    :param attributes_keys: The keys of the attributes to patch.
    :param auth_info:       The auth info of the request.
    :param db_session:      A session that manages the current dialog with the database.

    :return:                The patched model endpoint object.
    """

    logger.info(
        "Patching Model Endpoint record",
        model_endpoint=model_endpoint,
        attributes_keys=attributes_keys,
    )
    if project != model_endpoint.metadata.project:
        raise MLRunInvalidArgumentError(
            f"Project name in the URL '{project}' does not match the project name in the model endpoint metadata "
            f"'{model_endpoint.metadata.project}'. User is not allowed to patch model endpoint in a different project."
        )

    await (
        framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            resource_type=schemas.AuthorizationResourceTypes.model_endpoint,
            project_name=project,
            resource_name=model_endpoint.metadata.name,
            action=schemas.AuthorizationAction.update,
            auth_info=auth_info,
        )
    )
    attributes = {key: model_endpoint.get(key) for key in attributes_keys}

    return await run_in_threadpool(
        services.api.crud.ModelEndpoints().patch_model_endpoint,
        name=model_endpoint.metadata.name,
        project=project,
        function_name=model_endpoint.spec.function_name,
        function_tag=model_endpoint.spec.function_tag,
        endpoint_id=model_endpoint.metadata.uid,
        attributes=attributes,
        db_session=db_session,
    )


@router.delete(
    "/{name}",
    status_code=HTTPStatus.NO_CONTENT.value,
)
async def delete_model_endpoint(
    project: ProjectAnnotation,
    name: str,
    function_name: Optional[str] = None,
    function_tag: Optional[str] = None,
    endpoint_id: typing.Optional[EndpointIDAnnotation] = "*",
    auth_info: schemas.AuthInfo = Depends(framework.api.deps.authenticate_request),
    db_session: Session = Depends(framework.api.deps.get_db_session),
) -> None:
    """
    Delete a model endpoint record from the DB.
    :param project:         The name of the project.
    :param name:            The model endpoint name.
    :param function_name:   The name of the function.
    :param function_tag:    The tag of the function.
    :param endpoint_id:     The unique id of the model endpoint.
    :param auth_info:       The auth info of the request.
    :param db_session:      A session that manages the current dialog with the database.
    """

    await (
        framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            resource_type=schemas.AuthorizationResourceTypes.model_endpoint,
            project_name=project,
            resource_name=name,
            action=schemas.AuthorizationAction.delete,
            auth_info=auth_info,
        )
    )

    await run_in_threadpool(
        services.api.crud.ModelEndpoints().delete_model_endpoint,
        project=project,
        name=name,
        function_name=function_name,
        function_tag=function_tag,
        db_session=db_session,
        endpoint_id=endpoint_id,
    )


@router.get(
    "",
    status_code=HTTPStatus.OK.value,
    response_model=schemas.ModelEndpointList,
)
async def list_model_endpoints(
    project: ProjectAnnotation,
    name: Optional[str] = None,
    model_name: Optional[str] = None,
    model_tag: Optional[str] = None,
    function_name: Optional[str] = None,
    function_tag: Optional[str] = None,
    labels: list[str] = Query([], alias="label"),
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    top_level: bool = Query(False, alias="top-level"),
    tsdb_metrics: bool = True,
    uids: list[str] = Query(None, alias="uid"),
    latest_only: bool = False,
    auth_info: schemas.AuthInfo = Depends(framework.api.deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
) -> schemas.ModelEndpointList:
    """
    List model endpoints.

    :param project:         The name of the project.
    :param name:            The model endpoint name.
    :param model_name:      The model name.
    :param function_name:   The function name.
    :param function_tag:    The function tag.
    :param labels:          The labels of the model endpoint.
    :param start:           The start time to filter by.Corresponding to the `created` field.
    :param end:             The end time to filter by. Corresponding to the `created` field.
    :param tsdb_metrics:    Whether to include metrics from the time series DB.
    :param top_level:       Whether to return only top level model endpoints.
    :param uids:            A list of unique ids to filter by.
    :param latest_only:     Whether to return only the latest model endpoint for each name.
    :param auth_info:       The auth info of the request.
    :param db_session:      A session that manages the current dialog with the database.
    :return:                A list of model endpoints.
    """

    await framework.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project_name=project,
        action=schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )

    endpoints = await run_in_threadpool(
        services.api.crud.ModelEndpoints().list_model_endpoints,
        project=project,
        name=name,
        model_name=model_name,
        model_tag=model_tag,
        function_name=function_name,
        function_tag=function_tag,
        labels=labels,
        start=start,
        end=end,
        top_level=top_level,
        tsdb_metrics=tsdb_metrics,
        uids=uids,
        latest_only=latest_only,
        db_session=db_session,
    )
    allowed_endpoints = await framework.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
        schemas.AuthorizationResourceTypes.model_endpoint,
        endpoints.endpoints,
        lambda _endpoint: (
            _endpoint.metadata.project,
            _endpoint.metadata.uid,
        ),
        auth_info,
    )

    endpoints.endpoints = allowed_endpoints

    return endpoints


async def _verify_model_endpoint_read_permission(
    *, project: str, name_or_uid: str, auth_info: schemas.AuthInfo
) -> None:
    await (
        framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            schemas.AuthorizationResourceTypes.model_endpoint,
            project_name=project,
            resource_name=name_or_uid,
            action=schemas.AuthorizationAction.read,
            auth_info=auth_info,
        )
    )


@router.get(
    "/{name}",
    status_code=HTTPStatus.OK.value,
    response_model=schemas.ModelEndpoint,
)
async def get_model_endpoint(
    name: str,
    project: ProjectAnnotation,
    function_name: Optional[str] = None,
    function_tag: Optional[str] = None,
    endpoint_id: Optional[EndpointIDAnnotation] = None,
    tsdb_metrics: bool = True,
    feature_analysis: bool = False,
    auth_info: schemas.AuthInfo = Depends(framework.api.deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
) -> schemas.ModelEndpoint:
    """
    Get a model endpoint record from the DB.

    :param name:                The model endpoint name.
    :param project:             The name of the project.
    :param function_name:       The name of the function.
    :param function_tag:        The tag of the function.
    :param endpoint_id:         The unique id of the model endpoint.
    :param tsdb_metrics:        Whether to include metrics from the time series DB.
    :param feature_analysis:    Whether to include feature analysis.
    :param auth_info:           The auth info of the request.
    :param db_session:          A session that manages the current dialog with the database.
    :return:                    The model endpoint object.
    """
    await _verify_model_endpoint_read_permission(
        project=project, name_or_uid=name, auth_info=auth_info
    )

    return await run_in_threadpool(
        services.api.crud.ModelEndpoints().get_model_endpoint,
        name=name,
        project=project,
        function_name=function_name,
        function_tag=function_tag,
        endpoint_id=endpoint_id,
        feature_analysis=feature_analysis,
        tsdb_metrics=tsdb_metrics,
        db_session=db_session,
    )


@router.get(
    "/{endpoint_id}/metrics",
    response_model=list[mm_endpoints.ModelEndpointMonitoringMetric],
)
async def get_model_endpoint_monitoring_metrics(
    project: ProjectAnnotation,
    endpoint_id: EndpointIDAnnotation,
    auth_info: schemas.AuthInfo = Depends(framework.api.deps.authenticate_request),
    type: Literal["results", "metrics", "all"] = "all",
) -> list[mm_endpoints.ModelEndpointMonitoringMetric]:
    """
    :param project:     The name of the project.
    :param endpoint_id: The unique id of the model endpoint.
    :param auth_info:   The auth info of the request.
    :param type:        The type of the metrics to return. "all" means "results"
                        and "metrics".

    :returns:           A list of the application metrics or/and results for this model endpoint.
    """
    await _verify_model_endpoint_read_permission(
        project=project, name_or_uid=endpoint_id, auth_info=auth_info
    )
    metrics: list[mm_endpoints.ModelEndpointMonitoringMetric] = []
    tasks: list[asyncio.Task] = []
    if type == "results" or type == "all":
        tasks.append(
            asyncio.create_task(
                run_in_threadpool(
                    services.api.crud.ModelEndpoints.get_model_endpoints_metrics,
                    endpoint_id=endpoint_id,
                    type=mm_constants.ModelEndpointMonitoringMetricType.RESULT,
                    project=project,
                )
            )
        )
    if type == "metrics" or type == "all":
        tasks.append(
            asyncio.create_task(
                run_in_threadpool(
                    services.api.crud.ModelEndpoints.get_model_endpoints_metrics,
                    endpoint_id=endpoint_id,
                    type=mm_constants.ModelEndpointMonitoringMetricType.METRIC,
                    project=project,
                )
            )
        )
        metrics.append(mlrun.model_monitoring.helpers.get_invocations_metric(project))
    await asyncio.wait(tasks)
    for task in tasks:
        metrics.extend(task.result())
    return metrics


@dataclass
class _MetricsValuesParams:
    project: str
    endpoint_id: str
    metrics: list[mm_endpoints.ModelEndpointMonitoringMetric]
    results: list[mm_endpoints.ModelEndpointMonitoringMetric]
    start: datetime
    end: datetime


async def _get_metrics_values_params(
    project: ProjectAnnotation,
    endpoint_id: EndpointIDAnnotation,
    name: Annotated[
        list[str],
        Query(pattern=mm_constants.FQN_PATTERN),
    ],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    auth_info: schemas.AuthInfo = Depends(framework.api.deps.authenticate_request),
) -> _MetricsValuesParams:
    """
    Verify authorization, validate parameters and initialize the parameters.

    :param project:            The name of the project.
    :param endpoint_id:        The unique id of the model endpoint.
    :param name:               The full names of the requested results. At least one is required.
    :param start:              Start and end times are optional, and must be timezone aware.
    :param end:                See the `start` parameter.
    :param auth_info:          The auth info of the request.

    :return: _MetricsValuesParams object with the validated data.
    """
    await _verify_model_endpoint_read_permission(
        project=project, name_or_uid=endpoint_id, auth_info=auth_info
    )
    if start is None and end is None:
        end = mlrun.utils.helpers.datetime_now()
        start = end - timedelta(days=1)
    elif start is not None and end is not None:
        if start.tzinfo is None or end.tzinfo is None:
            raise mlrun.errors.MLRunInvalidArgumentTypeError(
                "Custom start and end times must contain the timezone."
            )
        if start > end:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "The start time must precede the end time."
            )
    else:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Provided only one of start time, end time. Please provide both or neither."
        )

    metrics = []
    results = []
    for fqn in name:
        metric = mm_endpoints._parse_metric_fqn_to_monitoring_metric(fqn)
        if metric.project != project:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Metric '{fqn}' does not belong to the project '{project}' given "
                f"in the API path, but to the project '{metric.project}'."
            )
        if metric.type == mm_constants.ModelEndpointMonitoringMetricType.METRIC:
            metrics.append(metric)
        else:
            results.append(metric)

    return _MetricsValuesParams(
        project=project,
        endpoint_id=endpoint_id,
        metrics=metrics,
        results=results,
        start=start,
        end=end,
    )


async def _wrap_coroutine_in_list(x):
    return [await x]


@router.get(
    "/{endpoint_id}/metrics-values",
    response_model=list[
        Union[
            mm_endpoints.ModelEndpointMonitoringMetricValues,
            mm_endpoints.ModelEndpointMonitoringResultValues,
            mm_endpoints.ModelEndpointMonitoringMetricNoData,
        ]
    ],
)
async def get_model_endpoint_monitoring_metrics_values(
    params: Annotated[_MetricsValuesParams, Depends(_get_metrics_values_params)],
) -> list[
    Union[
        mm_endpoints.ModelEndpointMonitoringMetricValues,
        mm_endpoints.ModelEndpointMonitoringResultValues,
        mm_endpoints.ModelEndpointMonitoringMetricNoData,
    ]
]:
    """
    :param params: A combined object with all the request parameters.

    :returns:      A list of the results values for this model endpoint.
    """
    coroutines: list[Coroutine] = []

    invocations_full_name = mlrun.model_monitoring.helpers.get_invocations_fqn(
        params.project
    )
    try:
        tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
            project=params.project,
            secret_provider=services.api.crud.secrets.get_project_secret_provider(
                project=params.project
            ),
        )
    except mlrun.errors.MLRunInvalidMMStoreTypeError as e:
        logger.debug(
            "Failed to retrieve model endpoint metrics-values because tsdb connection is not defined."
            " Returning an empty list of metric-values",
            error=mlrun.errors.err_to_str(e),
        )
        return []

    for metrics, type in [(params.results, "results"), (params.metrics, "metrics")]:
        if metrics:
            metrics_without_invocations = list(
                filter(
                    lambda metric: metric.full_name != invocations_full_name, metrics
                )
            )
            if len(metrics_without_invocations) != len(metrics):
                coroutines.append(
                    _wrap_coroutine_in_list(
                        run_in_threadpool(
                            tsdb_connector.read_predictions,
                            endpoint_id=params.endpoint_id,
                            start=params.start,
                            end=params.end,
                            aggregation_window=mm_constants.PredictionsQueryConstants.DEFAULT_AGGREGATION_GRANULARITY,
                            agg_funcs=["count"],
                        )
                    )
                )
            if metrics_without_invocations:
                coroutines.append(
                    run_in_threadpool(
                        tsdb_connector.read_metrics_data,
                        endpoint_id=params.endpoint_id,
                        start=params.start,
                        end=params.end,
                        metrics=metrics_without_invocations,
                        type=type,
                    )
                )

    metrics_values = []
    for result in await asyncio.gather(*coroutines):
        metrics_values.extend(result)
    return metrics_values
