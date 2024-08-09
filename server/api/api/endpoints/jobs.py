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
import typing

import fastapi
from fastapi import Header
from sqlalchemy.orm import Session

import mlrun.common.schemas
import server.api.utils.auth.verifier
from server.api.api import deps
from server.api.api.endpoints.nuclio import process_model_monitoring_secret
from server.api.crud.model_monitoring.deployment import MonitoringDeployment

router = fastapi.APIRouter(prefix="/projects/{project}/jobs")


# TODO: remove /projects/{project}/jobs/model-monitoring-controller in 1.9.0
@router.post(
    "/model-monitoring-controller",
    deprecated=True,
    description="/projects/{project}/jobs/model-monitoring-controller "
    "is deprecated in 1.7.0 and will be removed in 1.9.0, "
    "use /projects/{project}/model-monitoring/enable-model-monitoring instead",
)
async def create_model_monitoring_controller(
    project: str,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(deps.get_db_session),
    default_controller_image: str = "mlrun/mlrun",
    base_period: int = 10,
    client_version: typing.Optional[str] = Header(
        None, alias=mlrun.common.schemas.HeaderNames.client_version
    ),
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
    :param default_controller_image: The default image of the model monitoring controller job. Note that the writer
                                     function, which is a real time nuclio functino, will be deployed with the same
                                     image. By default, the image is mlrun/mlrun.
    :param base_period:              Minutes to determine the frequency in which the model monitoring controller job
                                     is running. By default, the base period is 5 minutes.
    :param client_version:           The client version that sent the request.
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

    MonitoringDeployment(
        project=project,
        auth_info=auth_info,
        db_session=db_session,
        model_monitoring_access_key=model_monitoring_access_key,
    ).deploy_monitoring_functions(
        image=default_controller_image,
        base_period=base_period,
        deploy_histogram_data_drift_app=False,  # mlrun client < 1.7.0
        client_version=client_version,
    )

    return {
        "func": "Submitted the model-monitoring controller, writer and stream deployment"
    }
