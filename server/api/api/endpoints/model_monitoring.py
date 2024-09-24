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
from typing import Annotated, Optional

import fastapi
import semver
from fastapi import APIRouter, Depends, Header, Query
from sqlalchemy.orm import Session

import mlrun.common.schemas
import server.api.api.utils
import server.api.utils.auth.verifier
import server.api.utils.clients.chief
from server.api import MINIMUM_CLIENT_VERSION_FOR_MM
from server.api.api import deps
from server.api.api.endpoints.nuclio import process_model_monitoring_secret
from server.api.crud.model_monitoring.deployment import MonitoringDeployment

router = APIRouter(prefix="/projects/{project}/model-monitoring")


@dataclass
class _CommonParams:
    """Common parameters for model monitoring endpoints"""

    project: str
    auth_info: mlrun.common.schemas.AuthInfo
    db_session: Session
    model_monitoring_access_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not mlrun.mlconf.is_ce_mode():
            # Get V3IO Access Key
            self.model_monitoring_access_key = process_model_monitoring_secret(
                self.db_session,
                self.project,
                mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ACCESS_KEY,
            )


async def _verify_authorization(
    project: str, auth_info: mlrun.common.schemas.AuthInfo, client_version: str
) -> None:
    """Verify project authorization"""
    if (
        semver.Version.parse(client_version)
        < semver.Version.parse(MINIMUM_CLIENT_VERSION_FOR_MM)
        and "unstable" not in client_version
    ):
        server.api.api.utils.log_and_raise(
            http.HTTPStatus.BAD_REQUEST.value,
            reason=f"Model monitoring is supported from client version {MINIMUM_CLIENT_VERSION_FOR_MM}. "
            f"Please upgrade your client accordingly.",
        )
    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=mlrun.common.schemas.AuthorizationResourceTypes.function,
        project_name=project,
        resource_name=mlrun.common.schemas.model_monitoring.MonitoringFunctionNames.APPLICATION_CONTROLLER,
        action=mlrun.common.schemas.AuthorizationAction.store,
        auth_info=auth_info,
    )


async def _common_parameters(
    project: str,
    auth_info: Annotated[
        mlrun.common.schemas.AuthInfo, Depends(deps.authenticate_request)
    ],
    db_session: Annotated[Session, Depends(deps.get_db_session)],
    client_version: Optional[str] = Header(
        None, alias=mlrun.common.schemas.HeaderNames.client_version
    ),
) -> _CommonParams:
    """
    Verify authorization and return common parameters.

    :param project:         Project name.
    :param auth_info:       The auth info of the request.
    :param db_session:      A session that manages the current dialog with the database.
    :param client_version:  The client version.
    :returns:          A `_CommonParameters` object that contains the input data.
    """
    await _verify_authorization(
        project=project, auth_info=auth_info, client_version=client_version
    )
    return _CommonParams(
        project=project,
        auth_info=auth_info,
        db_session=db_session,
    )


@router.post("/enable-model-monitoring")
async def enable_model_monitoring(
    commons: Annotated[_CommonParams, Depends(_common_parameters)],
    base_period: int = 10,
    image: str = "mlrun/mlrun",
    deploy_histogram_data_drift_app: bool = True,
    rebuild_images: bool = False,
    fetch_credentials_from_sys_config: bool = False,
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
    :param rebuild_images:                    If true, force rebuild of model monitoring infrastructure images
                                              (controller, writer & stream).
    :param fetch_credentials_from_sys_config: If true, fetch the credentials from the system configuration.

    """
    MonitoringDeployment(
        project=commons.project,
        auth_info=commons.auth_info,
        db_session=commons.db_session,
        model_monitoring_access_key=commons.model_monitoring_access_key,
    ).deploy_monitoring_functions(
        image=image,
        base_period=base_period,
        deploy_histogram_data_drift_app=deploy_histogram_data_drift_app,
        rebuild_images=rebuild_images,
        fetch_credentials_from_sys_config=fetch_credentials_from_sys_config,
    )


@router.patch("/model-monitoring-controller")
async def update_model_monitoring_controller(
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
            name=mlrun.common.schemas.model_monitoring.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            project=commons.project,
            tag="",
            auth_info=commons.auth_info,
        )

    except mlrun.errors.MLRunNotFoundError:
        raise mlrun.errors.MLRunNotFoundError(
            f"{mlrun.common.schemas.model_monitoring.MonitoringFunctionNames.APPLICATION_CONTROLLER} does not exist. "
            f"Run `project.enable_model_monitoring()` first."
        )

    return MonitoringDeployment(
        project=commons.project,
        auth_info=commons.auth_info,
        db_session=commons.db_session,
        model_monitoring_access_key=commons.model_monitoring_access_key,
    ).deploy_model_monitoring_controller(
        controller_image=image,
        base_period=base_period,
        overwrite=True,
    )


@router.post("/deploy-histogram-data-drift-app")
def deploy_histogram_data_drift_app(
    commons: Annotated[_CommonParams, Depends(_common_parameters)],
    image: str = "mlrun/mlrun",
) -> None:
    """
    Deploy the histogram data drift app on the go.

    :param commons: The common parameters of the request.
    :param image:   The image of the application, defaults to "mlrun/mlrun".
    """
    MonitoringDeployment(
        project=commons.project,
        auth_info=commons.auth_info,
        db_session=commons.db_session,
        model_monitoring_access_key=commons.model_monitoring_access_key,
    ).deploy_histogram_data_drift_app(image=image)


@router.delete(
    "/disable-model-monitoring",
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
    user_application_list: list[str] = None,
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
    tasks = await MonitoringDeployment(
        project=commons.project,
        auth_info=commons.auth_info,
        db_session=commons.db_session,
        model_monitoring_access_key=commons.model_monitoring_access_key,
    ).disable_model_monitoring(
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
    tasks = await MonitoringDeployment(
        project=commons.project,
        auth_info=commons.auth_info,
        db_session=commons.db_session,
        model_monitoring_access_key=commons.model_monitoring_access_key,
    ).disable_model_monitoring(
        delete_resources=False,
        delete_stream_function=False,
        delete_histogram_data_drift_app=False,
        delete_user_applications=True,
        user_application_list=functions,
        background_tasks=background_tasks,
    )
    response.status_code = http.HTTPStatus.ACCEPTED.value
    return tasks


@router.post("/set-model-monitoring-credentials")
def set_model_monitoring_credentials(
    commons: Annotated[_CommonParams, Depends(_common_parameters)],
    access_key: Optional[str] = None,
    endpoint_store_connection: Optional[str] = None,
    stream_path: Optional[str] = None,
    tsdb_connection: Optional[str] = None,
    replace_creds: bool = False,
) -> None:
    """
    Set the credentials that will be used by the project's model monitoring
    infrastructure functions. Important to note that you have to set the credentials before deploying any
    model monitoring or serving function.
    :param commons:                   The common parameters of the request.
    :param access_key:                Model Monitoring access key for managing user permissions.
    :param endpoint_store_connection: Endpoint store connection string. By default, None.
                                      Options:
                                      1. None, will be set from the system configuration.
                                      2. v3io - for v3io endpoint store,
                                         pass `v3io` and the system will generate the exact path.
                                      3. MySQL/SQLite - for SQL endpoint store, please provide full
                                         connection string, for example
                                         mysql+pymysql://<username>:<password>@<host>:<port>/<db_name>
    :param stream_path:               Path to the model monitoring stream. By default, None.
                                      Options:
                                      1. None, will be set from the system configuration.
                                      2. v3io - for v3io stream,
                                         pass `v3io` and the system will generate the exact path.
                                      3. Kafka - for Kafka stream, please provide full connection string without
                                         custom topic, for example kafka://<some_kafka_broker>:<port>.
    :param tsdb_connection:           Connection string to the time series database. By default, None.
                                      Options:
                                      1. None, will be set from the system configuration.
                                      2. v3io - for v3io stream,
                                         pass `v3io` and the system will generate the exact path.
                                      3. TDEngine - for TDEngine tsdb, please provide full websocket connection URL,
                                         for example taosws://<username>:<password>@<host>:<port>.
    :param replace_creds:             If True, it will force the credentials update. By default, False.
    """
    MonitoringDeployment(
        project=commons.project,
        auth_info=commons.auth_info,
        db_session=commons.db_session,
        model_monitoring_access_key=commons.model_monitoring_access_key,
    ).set_credentials(
        access_key=access_key,
        endpoint_store_connection=endpoint_store_connection,
        stream_path=stream_path,
        tsdb_connection=tsdb_connection,
        replace_creds=replace_creds,
    )
