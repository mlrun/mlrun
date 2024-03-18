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

import fastapi
from sqlalchemy.orm import Session

import mlrun.common.schemas
import server.api.utils.auth.verifier
import server.api.utils.clients.chief
from server.api.api import deps
from server.api.api.endpoints.functions import process_model_monitoring_secret
from server.api.crud.model_monitoring.deployment import MonitoringDeployment

router = fastapi.APIRouter(prefix="/projects/{project}/model-monitoring")


@router.post("/enable-model-monitoring")
async def enable_model_monitoring(
    project: str,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(deps.get_db_session),
    base_period: int = 10,
    image: str = "mlrun/mlrun",
):
    """
    Deploy model monitoring application controller, writer and stream functions.
    While the main goal of the controller function is to handle the monitoring processing and triggering
    applications, the goal of the model monitoring writer function is to write all the monitoring
    application results to the databases.
    And the stream function goal is to monitor the log of the data stream. It is triggered when a new log entry
    is detected. It processes the new events into statistics that are then written to statistics databases.


    :param project:                  Project name.
    :param auth_info:                The auth info of the request.
    :param db_session:               A session that manages the current dialog with the database.
    :param base_period:              The time period in minutes in which the model monitoring controller function
                                     triggers. By default, the base period is 10 minutes.
    :param image:                    The image of the model monitoring controller, writer & monitoring
                                     stream functions, which are real time nuclio functions.
                                     By default, the image is mlrun/mlrun.
    """

    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=mlrun.common.schemas.AuthorizationResourceTypes.function,
        project_name=project,
        resource_name=mlrun.common.schemas.model_monitoring.MonitoringFunctionNames.APPLICATION_CONTROLLER,
        action=mlrun.common.schemas.AuthorizationAction.store,
        auth_info=auth_info,
    )

    model_monitoring_access_key = None
    if not mlrun.mlconf.is_ce_mode():
        # Generate V3IO Access Key
        model_monitoring_access_key = process_model_monitoring_secret(
            db_session,
            project,
            mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ACCESS_KEY,
        )

    return MonitoringDeployment().deploy_monitoring_functions(
        project=project,
        model_monitoring_access_key=model_monitoring_access_key,
        db_session=db_session,
        auth_info=auth_info,
        image=image,
        base_period=base_period,
    )


@router.post("/model-monitoring-controller")
async def update_model_monitoring_controller(
    project: str,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(deps.get_db_session),
    base_period: int = 10,
    image: str = "mlrun/mlrun",
):
    """
    Redeploy model monitoring application controller function.
    The main goal of the controller function is to handle the monitoring processing and triggering applications.

    :param project:                  Project name.
    :param auth_info:                The auth info of the request.
    :param db_session:               A session that manages the current dialog with the database.
    :param base_period:              The time period in minutes in which the model monitoring controller function
                                     triggers. By default, the base period is 10 minutes.
    :param image:                    The default image of the model monitoring controller job. Note that the writer
                                     function, which is a real time nuclio functino, will be deployed with the same
                                     image. By default, the image is mlrun/mlrun.
    """

    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=mlrun.common.schemas.AuthorizationResourceTypes.function,
        project_name=project,
        resource_name=mlrun.common.schemas.model_monitoring.MonitoringFunctionNames.APPLICATION_CONTROLLER,
        action=mlrun.common.schemas.AuthorizationAction.store,
        auth_info=auth_info,
    )

    model_monitoring_access_key = None
    if not mlrun.mlconf.is_ce_mode():
        # Generate V3IO Access Key
        model_monitoring_access_key = process_model_monitoring_secret(
            db_session,
            project,
            mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ACCESS_KEY,
        )
    try:
        # validate that the model monitoring stream has not yet been deployed
        mlrun.runtimes.nuclio.function.get_nuclio_deploy_status(
            name=mlrun.common.schemas.model_monitoring.MonitoringFunctionNames.APPLICATION_CONTROLLER,
            project=project,
            tag="",
            auth_info=auth_info,
        )

    except mlrun.errors.MLRunNotFoundError:
        raise mlrun.errors.MLRunNotFoundError(
            f"{mlrun.common.schemas.model_monitoring.MonitoringFunctionNames.APPLICATION_CONTROLLER} does not exist. "
            f"Run `project.enable_model_monitoring()` first."
        )

    return MonitoringDeployment().deploy_model_monitoring_controller(
        project=project,
        model_monitoring_access_key=model_monitoring_access_key,
        db_session=db_session,
        auth_info=auth_info,
        controller_image=image,
        base_period=base_period,
        overwrite=True,
    )
