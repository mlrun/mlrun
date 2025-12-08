# Copyright 2025 Iguazio
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
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, Optional, Union

from fastapi import APIRouter, Depends, Query
from fastapi.concurrency import run_in_threadpool

import mlrun.common.schemas as schemas
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.common.schemas.model_monitoring.model_endpoints as mm_endpoints
import mlrun.errors
import mlrun.model_monitoring
from mlrun.model_monitoring.helpers import validate_time_range
from mlrun.utils import logger

import framework.api.deps
import services.api.common.constants as api_constants
import services.api.crud
from .model_endpoints import _verify_model_endpoint_read_permission

# Hard limit for results per metric in v2 API
_MAX_RESULTS_PER_METRIC = 500

router = APIRouter()

ProjectAnnotation = api_constants.ProjectAnnotation
EndpointIDAnnotation = api_constants.EndpointIDAnnotation


@dataclass
class _MetricsValuesParamsV2:
    """Parameters for v2 metrics-values endpoint with aggregation support."""

    project: str
    endpoint_id: str
    metrics: list[mm_endpoints.ModelEndpointMonitoringMetric]
    results: list[mm_endpoints.ModelEndpointMonitoringMetric]
    start: datetime
    end: datetime
    agg_period_requested: Optional[str]  # User's raw input: None, "raw", "1h", etc.
    agg_functions_requested: Optional[list[str]]  # User's raw input


async def _get_metrics_values_params_v2(
    project: ProjectAnnotation,
    endpoint_id: EndpointIDAnnotation,
    name: Annotated[
        list[str],
        Query(pattern=mm_constants.FQN_PATTERN),
    ],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    agg_period: Annotated[
        Optional[Literal["1h", "6h", "12h", "24h", "raw"]],
        Query(alias="agg-period"),
    ] = None,
    agg_functions: Annotated[
        Optional[list[Literal["avg", "min", "max", "count"]]],
        Query(alias="agg-function"),
    ] = None,
    auth_info: schemas.AuthInfo = Depends(framework.api.deps.authenticate_request),
) -> _MetricsValuesParamsV2:
    """
    Verify authorization, validate parameters and initialize the parameters for v2 API.

    :param project:       The name of the project.
    :param endpoint_id:   The unique id of the model endpoint.
    :param name:          The full names of the requested results. At least one is required.
    :param start:         Start time (optional, timezone aware).
    :param end:           End time (optional, timezone aware).
    :param agg_period:    Aggregation period:
                          "raw" explicitly requests non-aggregated data.
                          None triggers auto-selection based on configured intervals.
    :param agg_functions: Aggregation functions:
                          Defaults to ["avg"] when agg_period is specified.
    :param auth_info:     The auth info of the request.

    :return: _MetricsValuesParamsV2 object with the validated data.
    """
    await _verify_model_endpoint_read_permission(
        project=project, name_or_uid=endpoint_id, auth_info=auth_info
    )
    start, end = validate_time_range(start, end)

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

    return _MetricsValuesParamsV2(
        project=project,
        endpoint_id=endpoint_id,
        metrics=metrics,
        results=results,
        start=start,
        end=end,
        agg_period_requested=agg_period,
        agg_functions_requested=list(agg_functions) if agg_functions else None,
    )


async def _wrap_coroutine_in_list(x):
    return [await x]


@router.get(
    "/projects/{project}/model-endpoints/{endpoint_id}/metrics-values",
    response_model=list[
        Union[
            mm_endpoints.ModelEndpointMonitoringMetricValuesV2,
            mm_endpoints.ModelEndpointMonitoringResultValuesV2,
            mm_endpoints.ModelEndpointMonitoringMetricNoData,
        ]
    ],
)
async def get_model_endpoint_monitoring_metrics_values_v2(
    params: Annotated[_MetricsValuesParamsV2, Depends(_get_metrics_values_params_v2)],
) -> list[
    Union[
        mm_endpoints.ModelEndpointMonitoringMetricValuesV2,
        mm_endpoints.ModelEndpointMonitoringResultValuesV2,
        mm_endpoints.ModelEndpointMonitoringMetricNoData,
    ]
]:
    """
    Get model endpoint monitoring metrics/results values with v2 response format.

    The v2 API adds aggregation support:
    - `agg-period`: Aggregation period
    - `agg-function`: Aggregation functions

    When `agg-period` is omitted, auto-selection uses PreAggregateManager to choose
    the best available interval based on the time range and configured intervals.

    Response includes `aggregation_config` with aggregation metadata.

    :param params: A combined object with all the request parameters.
    :returns:      A list of the results values for this model endpoint with v2 schema.
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
    except mlrun.errors.MLRunNotFoundError as e:
        logger.debug(
            "Failed to retrieve model endpoint metrics-values because the TSDB datastore profile was not found. "
            "Returning an empty list of metric-values",
            error=mlrun.errors.err_to_str(e),
        )
        return []

    # Pass user's requested values directly to query methods
    # Query methods will auto-select period if None, and handle default functions
    agg_period = params.agg_period_requested
    agg_functions = params.agg_functions_requested

    for metrics, type in [(params.results, "results"), (params.metrics, "metrics")]:
        if metrics:
            metrics_without_invocations = list(
                filter(
                    lambda metric: metric.full_name != invocations_full_name, metrics
                )
            )
            if len(metrics_without_invocations) != len(metrics):
                # Handle invocations separately using read_predictions
                # Use the same aggregation settings as other metrics
                coroutines.append(
                    _wrap_coroutine_in_list(
                        run_in_threadpool(
                            tsdb_connector.read_predictions,
                            endpoint_id=params.endpoint_id,
                            start=params.start,
                            end=params.end,
                            aggregation_window=agg_period,
                            agg_funcs=agg_functions or None,
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
                        agg_period=agg_period,
                        agg_functions=agg_functions,
                    )
                )

    metrics_values = []
    for result in await asyncio.gather(*coroutines):
        metrics_values.extend(result)

    # Validate result count - hard limit of 500 results per metric
    for metric_value in metrics_values:
        if (
            hasattr(metric_value, "values")
            and len(metric_value.values) > _MAX_RESULTS_PER_METRIC
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Metric '{metric_value.full_name}' returned {len(metric_value.values)} results, "
                f"which exceeds the maximum allowed ({_MAX_RESULTS_PER_METRIC}). "
                f"Please use a larger aggregation period or narrow the time range."
            )

    return metrics_values
