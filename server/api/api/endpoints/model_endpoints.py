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
import json
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
import server.api.api.deps
import server.api.crud
import server.api.utils.auth.verifier
from mlrun.errors import MLRunConflictError
from mlrun.utils import logger

router = APIRouter(prefix="/projects/{project}/model-endpoints")

ProjectAnnotation = Annotated[str, Path(pattern=mm_constants.PROJECT_PATTERN)]
EndpointIDAnnotation = Annotated[
    str, Path(pattern=mm_constants.MODEL_ENDPOINT_ID_PATTERN)
]


@router.post(
    "/{endpoint_id}",
    response_model=schemas.ModelEndpoint,
)
async def create_model_endpoint(
    project: ProjectAnnotation,
    endpoint_id: EndpointIDAnnotation,
    model_endpoint: schemas.ModelEndpoint,
    auth_info: schemas.AuthInfo = Depends(server.api.api.deps.authenticate_request),
    db_session: Session = Depends(server.api.api.deps.get_db_session),
) -> schemas.ModelEndpoint:
    """
    Create a DB record of a given `ModelEndpoint` object.

    :param project:         The name of the project.
    :param endpoint_id:     The unique id of the model endpoint.
    :param model_endpoint:  Model endpoint object to record in DB.
    :param auth_info:       The auth info of the request.
    :param db_session:      A session that manages the current dialog with the database. When creating a new model
                            endpoint id record, we need to use the db session for getting information from an existing
                            model artifact and also for storing the new model monitoring feature set.

    :return: A Model endpoint object.
    """

    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=schemas.AuthorizationResourceTypes.model_endpoint,
        project_name=project,
        resource_name=endpoint_id,
        action=schemas.AuthorizationAction.store,
        auth_info=auth_info,
    )

    if project != model_endpoint.metadata.project:
        raise MLRunConflictError(
            f"Can't store endpoint of project {model_endpoint.metadata.project} into project {project}"
        )
    if endpoint_id != model_endpoint.metadata.uid:
        raise MLRunConflictError(
            f"Mismatch between endpoint_id {endpoint_id} and ModelEndpoint.metadata.uid {model_endpoint.metadata.uid}."
            f"\nMake sure the supplied function_uri, and model are configured as intended"
        )

    return await run_in_threadpool(
        server.api.crud.ModelEndpoints().create_model_endpoint,
        db_session=db_session,
        model_endpoint=model_endpoint,
    )


@router.patch(
    "/{endpoint_id}",
    response_model=schemas.ModelEndpoint,
)
async def patch_model_endpoint(
    project: ProjectAnnotation,
    endpoint_id: EndpointIDAnnotation,
    attributes: str = None,
    auth_info: schemas.AuthInfo = Depends(server.api.api.deps.authenticate_request),
) -> schemas.ModelEndpoint:
    """
    Update a DB record of a given `ModelEndpoint` object.

    :param project:       The name of the project.
    :param endpoint_id:   The unique id of the model endpoint.
    :param attributes:    Attributes that will be updated. The input is provided in a json structure that will be
                          converted into a dictionary before applying the patch process. Note that the keys of
                          the dictionary should exist in the DB target.

                          example::

                          attributes = {"drift_status": "POSSIBLE_DRIFT", "state": "new_state"}

    :param auth_info:     The auth info of the request.

    :return: A Model endpoint object.
    """

    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=schemas.AuthorizationResourceTypes.model_endpoint,
        project_name=project,
        resource_name=endpoint_id,
        action=schemas.AuthorizationAction.update,
        auth_info=auth_info,
    )

    if not attributes:
        raise mlrun.errors.MLRunNotFoundError(
            f"No attributes provided for patching the model endpoint {endpoint_id}",
        )
    return await run_in_threadpool(
        server.api.crud.ModelEndpoints().patch_model_endpoint,
        project=project,
        endpoint_id=endpoint_id,
        attributes=json.loads(attributes),
    )


@router.delete(
    "/{endpoint_id}",
    status_code=HTTPStatus.NO_CONTENT.value,
)
async def delete_model_endpoint(
    project: ProjectAnnotation,
    endpoint_id: EndpointIDAnnotation,
    auth_info: schemas.AuthInfo = Depends(server.api.api.deps.authenticate_request),
):
    """
    Clears endpoint record from the DB based on endpoint_id.

    :param project:       The name of the project.
    :param endpoint_id:   The unique id of the model endpoint.
    :param auth_info:     The auth info of the request.

    """

    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=schemas.AuthorizationResourceTypes.model_endpoint,
        project_name=project,
        resource_name=endpoint_id,
        action=schemas.AuthorizationAction.delete,
        auth_info=auth_info,
    )

    await run_in_threadpool(
        server.api.crud.ModelEndpoints().delete_model_endpoint,
        project=project,
        endpoint_id=endpoint_id,
    )


@router.get(
    "",
    response_model=schemas.ModelEndpointList,
)
async def list_model_endpoints(
    project: ProjectAnnotation,
    model: Optional[str] = Query(None),
    function: Optional[str] = Query(None),
    labels: list[str] = Query([], alias="label"),
    start: str = Query(default="now-1h"),
    end: str = Query(default="now"),
    metrics: list[str] = Query([], alias="metric"),
    top_level: bool = Query(False, alias="top-level"),
    uids: list[str] = Query(None, alias="uid"),
    auth_info: schemas.AuthInfo = Depends(server.api.api.deps.authenticate_request),
) -> schemas.ModelEndpointList:
    """
    Returns a list of endpoints of type 'ModelEndpoint', supports filtering by model, function, tag,
    labels or top level. By default, when no filters are applied, all available endpoints for the given project will be
    listed.

    If uids are passed: will return `ModelEndpointList` of endpoints with uid in uids
    Labels can be used to filter on the existence of a label:
    api/projects/{project}/model-endpoints/?label=mylabel

    Or on the value of a given label:
    api/projects/{project}/model-endpoints/?label=mylabel=1

    Multiple labels can be queried in a single request by either using "&" separator:
    api/projects/{project}/model-endpoints/?label=mylabel=1&label=myotherlabel=2

    Or by using a "," (comma) separator:
    api/projects/{project}/model-endpoints/?label=mylabel=1,myotherlabel=2
    Top level: if true will return only routers and endpoint that are NOT children of any router

    :param auth_info: The auth info of the request.
    :param project:   The name of the project.
    :param model:     The name of the model to filter by.
    :param function:  The name of the function to filter by.
    :param labels:    A list of labels to filter by. Label filters work by either filtering a specific value of a label
                      (i.e. list("key=value")) or by looking for the existence of a given key (i.e. "key").
    :param metrics:   A list of real-time metrics to return for each endpoint. There are pre-defined real-time metrics
                      for model endpoints such as predictions_per_second and latency_avg_5m but also custom metrics
                      defined by the user. Please note that these metrics are stored in the time series DB and the
                      results will be appeared under model_endpoint.spec.metrics of each endpoint.
    :param start:     The start time of the metrics. Can be represented by a string containing an RFC 3339 time, a
                      Unix timestamp in milliseconds, a relative time (`'now'` or `'now-[0-9]+[mhd]'`, where
                      `m` = minutes, `h` = hours, `'d'` = days, and `'s'` = seconds), or 0 for the earliest time.
    :param end:       The end time of the metrics. Can be represented by a string containing an RFC 3339 time, a
                      Unix timestamp in milliseconds, a relative time (`'now'` or `'now-[0-9]+[mhd]'`, where
                      `m` = minutes, `h` = hours, `'d'` = days, and `'s'` = seconds), or 0 for the earliest time.
    :param top_level: If True will return only routers and endpoint that are NOT children of any router.
    :param uids:      Will return `ModelEndpointList` of endpoints with uid in uids.

    :return: An object of `ModelEndpointList` which is literally a list of model endpoints along with some metadata. To
             get a standard list of model endpoints use ModelEndpointList.endpoints.
    """

    await server.api.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project_name=project,
        action=schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )

    endpoints = await run_in_threadpool(
        server.api.crud.ModelEndpoints().list_model_endpoints,
        project=project,
        model=model,
        function=function,
        labels=labels,
        metrics=metrics,
        start=start,
        end=end,
        top_level=top_level,
        uids=uids,
    )
    allowed_endpoints = await server.api.utils.auth.verifier.AuthVerifier().filter_project_resources_by_permissions(
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
    *, project: str, endpoint_id: str, auth_info: schemas.AuthInfo
) -> None:
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        schemas.AuthorizationResourceTypes.model_endpoint,
        project_name=project,
        resource_name=endpoint_id,
        action=schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )


@router.get(
    "/{endpoint_id}",
    response_model=schemas.ModelEndpoint,
)
async def get_model_endpoint(
    project: ProjectAnnotation,
    endpoint_id: EndpointIDAnnotation,
    start: str = Query(default="now-1h"),
    end: str = Query(default="now"),
    metrics: list[str] = Query([], alias="metric"),
    feature_analysis: bool = Query(default=False),
    auth_info: schemas.AuthInfo = Depends(server.api.api.deps.authenticate_request),
) -> schemas.ModelEndpoint:
    """Get a single model endpoint object. You can apply different time series metrics that will be added to the
       result.


    :param project:                    The name of the project
    :param endpoint_id:                The unique id of the model endpoint.
    :param start:                      The start time of the metrics. Can be represented by a string containing an RFC
                                       3339 time, a  Unix timestamp in milliseconds, a relative time (`'now'` or
                                       `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, `'d'` = days, and `'s'`
                                       = seconds), or 0 for the earliest time.
    :param end:                        The end time of the metrics. Can be represented by a string containing an RFC
                                       3339 time, a  Unix timestamp in milliseconds, a relative time (`'now'` or
                                       `'now-[0-9]+[mhd]'`, where `m` = minutes, `h` = hours, `'d'` = days, and `'s'`
                                       = seconds), or 0 for the earliest time.
    :param metrics:                    A list of real-time metrics to return for the model endpoint. There are
                                       pre-defined real-time metrics for model endpoints such as predictions_per_second
                                       and latency_avg_5m but also custom metrics defined by the user. Please note that
                                       these metrics are stored in the time series DB and the results will be
                                       appeared under model_endpoint.spec.metrics.
    :param feature_analysis:           When True, the base feature statistics and current feature statistics will
                                       be added to the output of the resulting object.
    :param auth_info:                  The auth info of the request

    :return:  A `ModelEndpoint` object.
    """
    await _verify_model_endpoint_read_permission(
        project=project, endpoint_id=endpoint_id, auth_info=auth_info
    )

    return await run_in_threadpool(
        server.api.crud.ModelEndpoints().get_model_endpoint,
        project=project,
        endpoint_id=endpoint_id,
        metrics=metrics,
        start=start,
        end=end,
        feature_analysis=feature_analysis,
    )


@router.get(
    "/{endpoint_id}/metrics",
    response_model=list[mm_endpoints.ModelEndpointMonitoringMetric],
)
async def get_model_endpoint_monitoring_metrics(
    project: ProjectAnnotation,
    endpoint_id: EndpointIDAnnotation,
    auth_info: schemas.AuthInfo = Depends(server.api.api.deps.authenticate_request),
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
        project=project, endpoint_id=endpoint_id, auth_info=auth_info
    )
    try:
        get_model_endpoint_metrics = (
            server.api.crud.model_monitoring.helpers.get_store_object(
                project=project
            ).get_model_endpoint_metrics
        )
    except mlrun.errors.MLRunInvalidMMStoreTypeError as e:
        logger.debug(
            "Failed to list model endpoint metrics because store connection is not defined."
            " Returning an empty list of metrics",
            error=mlrun.errors.err_to_str(e),
        )
        return []
    metrics: list[mm_endpoints.ModelEndpointMonitoringMetric] = []
    tasks: list[asyncio.Task] = []
    if type == "results" or type == "all":
        tasks.append(
            asyncio.create_task(
                run_in_threadpool(
                    get_model_endpoint_metrics,
                    endpoint_id=endpoint_id,
                    type=mm_constants.ModelEndpointMonitoringMetricType.RESULT,
                )
            )
        )
    if type == "metrics" or type == "all":
        tasks.append(
            asyncio.create_task(
                run_in_threadpool(
                    get_model_endpoint_metrics,
                    endpoint_id=endpoint_id,
                    type=mm_constants.ModelEndpointMonitoringMetricType.METRIC,
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
    auth_info: schemas.AuthInfo = Depends(server.api.api.deps.authenticate_request),
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
        project=project, endpoint_id=endpoint_id, auth_info=auth_info
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
            secret_provider=server.api.crud.secrets.get_project_secret_provider(
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
