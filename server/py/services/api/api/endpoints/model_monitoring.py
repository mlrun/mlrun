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

import http
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated, Literal, Optional

import fastapi
import semver
from fastapi import APIRouter, Depends, Header, Query
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

import mlrun.common.schemas
import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.common.schemas.model_monitoring.model_endpoints as mm_endpoints
import mlrun.model_monitoring.helpers
from mlrun.utils import logger

import framework.api.utils
import framework.utils.auth.verifier
import services.api.api.endpoints.model_endpoints
import services.api.common.constants as api_constants
from framework.api import deps
from framework.constants import MINIMUM_CLIENT_VERSION_FOR_MM
from services.api.api.endpoints.nuclio import process_model_monitoring_secret
from services.api.crud.model_monitoring.deployment import MonitoringDeployment

ProjectAnnotation = api_constants.ProjectAnnotation
EndpointIDAnnotation = api_constants.EndpointIDAnnotation

router = APIRouter(prefix="/projects/{project}/model-monitoring")


@dataclass
class _CommonParams:
    """Common parameters for model monitoring endpoints"""

    project: str
    auth_info: mlrun.common.schemas.AuthInfo
    db_session: Session
    model_monitoring_access_key: Optional[str] = None
    auth_token_name: Optional[str] = None

    def __post_init__(self) -> None:
        if mlrun.mlconf.is_using_v3io():
            # Get V3IO Access Key
            self.model_monitoring_access_key = process_model_monitoring_secret(
                self.db_session,
                self.project,
                mm_constants.ProjectSecretKeys.ACCESS_KEY,
            )

    def get_monitoring_deployment(self) -> MonitoringDeployment:
        """Get the MonitoringDeployment instance for the current project"""
        return MonitoringDeployment(
            project=self.project,
            auth_info=self.auth_info,
            db_session=self.db_session,
            model_monitoring_access_key=self.model_monitoring_access_key,
            auth_token_name=self.auth_token_name,
        )


async def _verify_authorization(
    project: str,
    auth_info: mlrun.common.schemas.AuthInfo,
    client_version: str,
    action: str = mlrun.common.schemas.AuthorizationAction.store,
) -> None:
    """Verify project authorization"""
    if (
        client_version
        and semver.Version.parse(client_version)
        < semver.Version.parse(MINIMUM_CLIENT_VERSION_FOR_MM)
        and "unstable" not in client_version
    ):
        framework.api.utils.log_and_raise(
            http.HTTPStatus.BAD_REQUEST.value,
            reason=f"Model monitoring is supported from client version {MINIMUM_CLIENT_VERSION_FOR_MM}. "
            f"Please upgrade your client accordingly.",
        )
    await (
        framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
            resource_type=mlrun.common.schemas.AuthorizationResourceTypes.function,
            project_name=project,
            resource_name=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            action=action,
            auth_info=auth_info,
        )
    )


async def _common_parameters(
    project: ProjectAnnotation,
    auth_info: Annotated[
        mlrun.common.schemas.AuthInfo, Depends(deps.authenticate_request)
    ],
    db_session: Annotated[Session, Depends(deps.get_db_session)],
    client_version: Optional[str] = Header(
        None, alias=mlrun.common.schemas.HeaderNames.client_version
    ),
    auth_token_name: Optional[str] = Query(
        None, description="Auth token name (set by mlrun.RuntimeConfigurationContext)"
    ),
) -> _CommonParams:
    """
    Verify authorization and return common parameters.

    :param project:         Project name.
    :param auth_info:       The auth info of the request.
    :param db_session:      A session that manages the current dialog with the database.
    :param client_version:  The client version.
    :param auth_token_name: The auth token name (set by mlrun.RuntimeConfigurationContext).
    :returns:          A `_CommonParameters` object that contains the input data.
    """
    await _verify_authorization(
        project=project, auth_info=auth_info, client_version=client_version
    )
    return _CommonParams(
        project=project,
        auth_info=auth_info,
        db_session=db_session,
        auth_token_name=auth_token_name,
    )


@router.put("/")
def enable_model_monitoring(
    commons: Annotated[_CommonParams, Depends(_common_parameters)],
    base_period: int = 10,
    image: str = "mlrun/mlrun",
    deploy_histogram_data_drift_app: bool = True,
    fetch_credentials_from_sys_config: bool = Query(
        False,
        deprecated=True,
        description=(
            "`fetch_credentials_from_sys_config` is deprecated as of 1.10.0 and will be removed in 1.12.0."
        ),
    ),
    lag_threshold: int | None = Query(
        None, description="Lag threshold in minutes for writer lag detection."
    ),
    lag_event_cooldown: int | None = Query(
        None,
        description="Cooldown in minutes between consecutive lag events per worker.",
    ),
):
    """
    Deploy model monitoring application controller, writer and stream functions.
    While the main goal of the controller function is to handle the monitoring processing and triggering
    applications, the goal of the model monitoring writer function is to write all the monitoring
    application results to the databases.
    And the stream function goal is to monitor the log of the data stream. It is triggered when a new log entry
    is detected. It processes the new events into statistics that are then written to statistics databases.

    :param commons:                           The common parameters of the request.
    :param base_period:                       The time period in minutes in which the model monitoring controller
                                              function triggers. By default, the base period is 10 minutes.
    :param image:                             The image of the model monitoring controller, writer & monitoring
                                              stream functions, which are real time nuclio functions.
                                              By default, the image is mlrun/mlrun.
    :param deploy_histogram_data_drift_app:   If true, deploy the default histogram-based data drift application.
    :param fetch_credentials_from_sys_config: Deprecated. If true, fetch the credentials from the system configuration.
    :param lag_threshold:                     Lag threshold in minutes for writer lag detection.
    :param lag_event_cooldown:                Cooldown in minutes between consecutive lag events per worker.

    """
    commons.get_monitoring_deployment().deploy_monitoring_functions(
        image=image,
        base_period=base_period,
        deploy_histogram_data_drift_app=deploy_histogram_data_drift_app,
        fetch_credentials_from_sys_config=fetch_credentials_from_sys_config,
        lag_threshold=lag_threshold,
        lag_event_cooldown=lag_event_cooldown,
    )


@router.patch("/controller")
def update_model_monitoring_controller(
    commons: Annotated[_CommonParams, Depends(_common_parameters)],
    base_period: int = 10,
    image: str = "mlrun/mlrun",
):
    """
    Redeploy model monitoring application controller function.
    The main goal of the controller function is to handle the monitoring processing and triggering applications.

    :param commons:     The common parameters of the request.
    :param base_period: The time period in minutes in which the model monitoring controller function
                        triggers. By default, the base period is 10 minutes.
    :param image:       The default image of the model monitoring controller job. Note that the writer
                        function, which is a real time nuclio functino, will be deployed with the same
                        image. By default, the image is mlrun/mlrun.
    """
    try:
        # validate that the model monitoring stream has not yet been deployed
        mlrun.runtimes.nuclio.function.get_nuclio_deploy_status(
            name=mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            project=commons.project,
            tag="",
            auth_info=commons.auth_info,
        )

    except mlrun.errors.MLRunNotFoundError:
        raise mlrun.errors.MLRunNotFoundError(
            f"{mm_constants.MonitoringFunctionNames.APPLICATION_CONTROLLER} does not exist. "
            f"Run `project.enable_model_monitoring()` first."
        )

    return commons.get_monitoring_deployment().deploy_model_monitoring_controller(
        controller_image=image,
        base_period=base_period,
        overwrite=True,
    )


@router.delete(
    "/",
    responses={
        http.HTTPStatus.ACCEPTED.value: {
            "model": mlrun.common.schemas.BackgroundTaskList
        },
    },
)
async def disable_model_monitoring(
    commons: Annotated[_CommonParams, Depends(_common_parameters)],
    background_tasks: fastapi.BackgroundTasks,
    response: fastapi.Response,
    delete_resources: bool = True,
    delete_stream_function: bool = False,
    delete_histogram_data_drift_app: bool = True,
    delete_user_applications: bool = False,
    user_application_list: Optional[list[str]] = None,
):
    """
    Disable model monitoring application controller, writer, stream, histogram data drift application
    and the user's applications functions, according to the given params.

    :param commons:                             The common parameters of the request.
    :param background_tasks:                    Background tasks.
    :param response:                            The response.
    :param delete_resources:                    If True, it would delete the model monitoring controller & writer
                                                functions. Default True
    :param delete_stream_function:              If True, it would delete model monitoring stream function,
                                                need to use wisely because if you're deleting this function
                                                this can cause data loss in case you will want to
                                                enable the model monitoring capability to the project.
                                                Default False.
    :param delete_histogram_data_drift_app:     If True, it would delete the default histogram-based data drift
                                                application. Default False.
    :param delete_user_applications:            If True, it would delete the user's model monitoring
                                                application according to user_application_list, Default False.
    :param user_application_list:               List of the user's model monitoring application to disable.
                                                Default all the applications.
                                                Note: you have to set delete_user_applications to True
                                                in order to delete the desired application.

    """
    tasks = await commons.get_monitoring_deployment().disable_model_monitoring(
        delete_resources=delete_resources,
        delete_stream_function=delete_stream_function,
        delete_histogram_data_drift_app=delete_histogram_data_drift_app,
        delete_user_applications=delete_user_applications,
        user_application_list=user_application_list,
        background_tasks=background_tasks,
    )
    response.status_code = http.HTTPStatus.ACCEPTED.value
    return tasks


@router.delete(
    "/functions",
    responses={
        http.HTTPStatus.ACCEPTED.value: {
            "model": mlrun.common.schemas.BackgroundTaskList
        },
    },
)
async def delete_model_monitoring_function(
    commons: Annotated[_CommonParams, Depends(_common_parameters)],
    background_tasks: fastapi.BackgroundTasks,
    response: fastapi.Response,
    functions: list[str] = Query([], alias="function"),
):
    """
    Delete model monitoring functions.

    :param commons:                             The common parameters of the request.
    :param background_tasks:                    Background tasks.
    :param response:                            The response.
    :param functions:                           List of the user's model monitoring application to delete.
    """
    tasks = await commons.get_monitoring_deployment().disable_model_monitoring(
        delete_resources=False,
        delete_stream_function=False,
        delete_histogram_data_drift_app=False,
        delete_user_applications=True,
        user_application_list=functions,
        background_tasks=background_tasks,
    )
    response.status_code = http.HTTPStatus.ACCEPTED.value
    return tasks


@router.put("/credentials")
def set_model_monitoring_credentials(
    commons: Annotated[_CommonParams, Depends(_common_parameters)],
    tsdb_profile_name: str,
    stream_profile_name: str,
    replace_creds: bool = False,
) -> None:
    """
    Set the credentials that will be used by the project's model monitoring
    infrastructure functions. Important to note that you have to set the credentials before deploying any
    model monitoring or serving function.
    :param commons:                   The common parameters of the request.
    :param tsdb_profile_name:         TSDB datastore profile name.
    :param stream_profile_name:       Stream datastore profile name.
                                      The profile can be V3IO or KafkaSource.
    :param replace_creds:             If True, it will force the credentials update. By default, False.
    """
    commons.get_monitoring_deployment().set_credentials(
        tsdb_profile_name=tsdb_profile_name,
        stream_profile_name=stream_profile_name,
        replace_creds=replace_creds,
    )


@dataclass
class _FunctionSummariesParams:
    project: str
    auth_info: mlrun.common.schemas.AuthInfo
    db_session: Session
    start: datetime
    end: datetime


async def _common_function_parameters(
    project: api_constants.ProjectAnnotation,
    auth_info: Annotated[
        mlrun.common.schemas.AuthInfo, Depends(deps.authenticate_request)
    ],
    db_session: Annotated[Session, Depends(deps.get_db_session)],
    client_version: Optional[str] = Header(
        None, alias=mlrun.common.schemas.HeaderNames.client_version
    ),
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> _FunctionSummariesParams:
    """
    Verify authorization and return common parameters.

    :param project:         Project name.
    :param auth_info:       The auth info of the request.
    :param db_session:      A session that manages the current dialog with the database.
    :returns:          A `_FunctionSummariesParams` object that contains the input data.
    """

    await _verify_authorization(
        project=project,
        auth_info=auth_info,
        client_version=client_version,
        action=mlrun.common.schemas.AuthorizationAction.read,
    )
    if (start and start.tzinfo is None) or (end and end.tzinfo is None):
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Custom start and end times must contain the timezone."
        )
    if start is None and end is None:
        end = mlrun.utils.helpers.datetime_now()
        start = end - timedelta(days=1)
    elif start is not None and end is not None:
        if start > end:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "The start time must be before the end time. Note that if end time is not provided, "
                "the current time is used by default."
            )
    return _FunctionSummariesParams(
        project=project,
        auth_info=auth_info,
        db_session=db_session,
        start=start,
        end=end,
    )


@router.get("/function-summaries")
async def get_model_monitoring_function_summaries(
    commons: Annotated[_FunctionSummariesParams, Depends(_common_function_parameters)],
    names: Optional[list[str]] = Query(None, alias="name"),
    labels: list[str] = Query([], alias="label"),
    include_stats: bool = Query(True, alias="include-stats"),
    include_infra: bool = Query(True, alias="include-infra"),
) -> list[mlrun.common.schemas.model_monitoring.FunctionSummary]:
    """Get monitoring function summaries for the specified project.

    :param commons:       The common parameters of the request.
    :param names:         List of function names to filter by (optional).
    :param labels:        Labels to filter by (optional).
    :param include_stats: Whether to include statistics in the response (default is True).
    :param include_infra: whether to include model monitoring infrastructure functions (default is True).

    :return: A list of FunctionSummary objects containing information about the monitoring functions.
    """
    return await MonitoringDeployment(
        project=commons.project,
        auth_info=commons.auth_info,
        db_session=commons.db_session,
    ).function_summaries(
        start=commons.start,
        end=commons.end,
        names=names,
        labels=labels,
        include_stats=include_stats,
        include_infra=include_infra,
    )


@router.get(
    "/function-summaries/{function_name}",
    response_model=mlrun.common.schemas.model_monitoring.FunctionSummary,
)
async def get_model_monitoring_function_summary(
    commons: Annotated[_FunctionSummariesParams, Depends(_common_function_parameters)],
    function_name: str,
    include_latest_metrics: bool = Query(True, alias="include-latest-metrics"),
) -> mlrun.common.schemas.model_monitoring.FunctionSummary:
    """Get monitoring function summary for the specified project and function name.
    :param commons:                The common parameters of the request.
    :param function_name:          The name of the function to retrieve the summary for.
    :param include_latest_metrics: Whether to include the latest metrics in the response (default is True).

    :return: A FunctionSummary object containing information about the monitoring function.
    """

    return await MonitoringDeployment(
        project=commons.project,
        auth_info=commons.auth_info,
        db_session=commons.db_session,
    ).function_summary(
        start=commons.start,
        end=commons.end,
        name=function_name,
        include_latest_metrics=include_latest_metrics,
    )


@router.get(
    "/metrics",
    response_model=dict[str, list[mm_endpoints.ModelEndpointMonitoringMetric]],
)
async def get_model_endpoints_metrics_values(
    commons: Annotated[_CommonParams, Depends(_common_parameters)],
    type: Literal["results", "metrics", "all"] = "all",
    endpoint_ids: list[api_constants.EndpointIDAnnotation] = Query(
        [], alias="endpoint-id"
    ),
    events_format: mm_constants.GetEventsFormat = Query(None, alias="events-format"),
) -> dict[str, list[mm_endpoints.ModelEndpointMonitoringMetric]]:
    """
    :param commons:          The common parameters of the request.
    :param type:          The type of the metrics to return. "all" means "results"
                          and "metrics".
    :param endpoint_ids:  The unique id of the model endpoint. Can be a single id or a list of ids.
    :param events_format: response format:
                          separation: {"mep_id1":[...], "mep_id2":[...]}
                          intersection {"intersect_metrics":[], "intersect_results":[]}
    :returns:             A dictionary of application metrics and/or results for the model endpoints,
                          formatted by events_format.
    """
    return await services.api.api.endpoints.model_endpoints.get_metrics_by_multiple_endpoints(
        project=commons.project,
        auth_info=commons.auth_info,
        db_session=commons.db_session,
        type=type,
        endpoint_ids=endpoint_ids,
        events_format=events_format,
    )


@router.delete("/metrics", status_code=http.HTTPStatus.NO_CONTENT)
async def delete_model_endpoints_metrics_values(
    commons: Annotated[_CommonParams, Depends(_common_parameters)],
    application_name: Annotated[
        str,
        Query(pattern=mm_constants.APP_NAME_REGEX.pattern, alias="application-name"),
    ],
    endpoint_id: Annotated[
        Optional[list[str]],
        Query(
            pattern=mm_constants.MODEL_ENDPOINT_ID_PATTERN,
            alias="endpoint-id",
            description=(
                "The unique id of the model endpoint. If none is provided, the metrics "
                "values will be deleted from all project's model endpoints."
            ),
        ),
    ] = None,
) -> None:
    """
    Delete model endpoints metrics values.

    :param commons:          The common parameters of the request.
    :param application_name: The name of the application.
    :param endpoint_id:      The unique IDs of the model endpoint to delete metrics values from. If none is
                             provided, the metrics values will be deleted from all project's model endpoints.
    """
    await framework.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=mlrun.common.schemas.AuthorizationResourceTypes.model_monitoring,
        project_name=commons.project,
        resource_name=application_name,
        action=mlrun.common.schemas.AuthorizationAction.delete,
        auth_info=commons.auth_info,
    )
    # call delete_application_records of the tsdb connector
    await run_in_threadpool(
        commons.get_monitoring_deployment().delete_application_records,
        application_name=application_name,
        endpoint_ids=endpoint_id,
    )


@router.get(
    "/drift-over-time",
    status_code=http.HTTPStatus.OK.value,
    response_model=mlrun.common.schemas.ModelEndpointDriftValues,
)
async def get_model_endpoint_drift_over_time(
    project: ProjectAnnotation,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(
        framework.api.deps.authenticate_request
    ),
) -> mlrun.common.schemas.ModelEndpointDriftValues:
    """
    Get drift counts over time for the project.

    :param project:     The name of the project.
    :param start:       Start time of the range to retrieve drift counts from.
    :param end:         End time of the range to retrieve drift counts from.
    :param auth_info:   The auth info of the request.

    :return: A ModelEndpointDriftValues object containing the drift counts over time.
    """
    start, end = mlrun.model_monitoring.helpers.validate_time_range(start, end)
    await framework.utils.auth.verifier.AuthVerifier().query_project_permissions(
        project_name=project,
        action=mlrun.common.schemas.AuthorizationAction.read,
        auth_info=auth_info,
    )
    try:
        tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
            project=project,
            secret_provider=services.api.crud.secrets.get_project_secret_provider(
                project=project
            ),
        )
    except mlrun.errors.MLRunNotFoundError as e:
        logger.debug(
            "Failed to retrieve model endpoint metrics-values because the TSDB datastore profile was not found. "
            "Returning an empty list of metric-values",
            error=mlrun.errors.err_to_str(e),
        )
        return mlrun.common.schemas.ModelEndpointDriftValues(values=[])
    return await run_in_threadpool(tsdb_connector.get_drift_data, start, end)
