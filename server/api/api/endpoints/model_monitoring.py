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
from mlrun.utils import logger
from server.api.api import deps
from server.api.api.endpoints.functions import process_model_monitoring_secret
from server.api.crud.model_monitoring.deployment import MonitoringDeployment

router = fastapi.APIRouter(prefix="/projects/{project}/model_monitoring")


@router.post("/enable-model-monitoring")
async def enable_model_monitoring(
    project: str,
    request: fastapi.Request,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(deps.get_db_session),
    image: str = "mlrun/mlrun",
    base_period: int = 10,
    overwrite: bool = False,
):
    """
    Submit model monitoring application controller job along with deploying the model monitoring writer function.
    While the main goal of the controller job is to handle the monitoring processing and triggering applications,
    the goal of the model monitoring writer function is to write all the monitoring application results to the
    databases. Note that the default scheduling policy of the controller job is to run every 10 min.

    :param project:                  Project name.
    :param request:                  fastapi request for the HTTP connection.
    :param auth_info:                The auth info of the request.
    :param db_session:               A session that manages the current dialog with the database.
    :param default_controller_image: The default image of the model monitoring controller job. Note that the writer
                                     function, which is a real time nuclio functino, will be deployed with the same
                                     image. By default, the image is mlrun/mlrun.
    :param base_period:              Minutes to determine the frequency in which the model monitoring controller job
                                     is running. By default, the base period is 5 minutes.
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

    MonitoringDeployment().deploy_monitoring_functions(
        project=project,
        model_monitoring_access_key=model_monitoring_access_key,
        db_session=db_session,
        auth_info=auth_info,
        image=image,
        base_period=base_period,
        overwrite=overwrite,
    )


@router.post("/model-monitoring-controller")
async def deploy_model_monitoring_controller(
    project: str,
    request: fastapi.Request,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(deps.get_db_session),
    default_controller_image: str = "mlrun/mlrun",
    base_period: int = 10,
):
    """
    Submit model monitoring application controller job along with deploying the model monitoring writer function.
    While the main goal of the controller job is to handle the monitoring processing and triggering applications,
    the goal of the model monitoring writer function is to write all the monitoring application results to the
    databases. Note that the default scheduling policy of the controller job is to run every 10 min.

    :param project:                  Project name.
    :param request:                  fastapi request for the HTTP connection.
    :param auth_info:                The auth info of the request.
    :param db_session:               A session that manages the current dialog with the database.
    :param default_controller_image: The default image of the model monitoring controller job. Note that the writer
                                     function, which is a real time nuclio functino, will be deployed with the same
                                     image. By default, the image is mlrun/mlrun.
    :param base_period:              Minutes to determine the frequency in which the model monitoring controller job
                                     is running. By default, the base period is 5 minutes.
    """

    await server.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=mlrun.common.schemas.AuthorizationResourceTypes.function,
        project_name=project,
        resource_name=mlrun.common.schemas.model_monitoring.MonitoringFunctionNames.APPLICATION_CONTROLLER,
        action=mlrun.common.schemas.AuthorizationAction.store,
        auth_info=auth_info,
    )

    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting to deploy model monitoring controller, re-routing to chief",
            function_name="model-monitoring-controller",
            project=project,
        )
        chief_client = server.api.utils.clients.chief.Client()
        params = {
            "default_controller_image": default_controller_image,
            "base_period": base_period,
        }
        return await chief_client.create_model_monitoring_controller(
            project=project, request=request, json=params
        )

    model_monitoring_access_key = None
    if not mlrun.mlconf.is_ce_mode():
        # Generate V3IO Access Key
        model_monitoring_access_key = process_model_monitoring_secret(
            db_session,
            project,
            mlrun.common.schemas.model_monitoring.ProjectSecretKeys.ACCESS_KEY,
        )

    MonitoringDeployment().deploy_model_monitoring_controller(
        project=project,
        model_monitoring_access_key=model_monitoring_access_key,
        db_session=db_session,
        auth_info=auth_info,
        controller_image=default_controller_image,
        base_period=base_period,
    )
