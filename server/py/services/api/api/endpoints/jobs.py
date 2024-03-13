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
from typing import Any

import fastapi
from sqlalchemy.orm import Session

import mlrun.common.schemas
import server.py.services.api.utils.auth.verifier
import server.py.services.api.utils.clients.chief
from mlrun.model_monitoring import TrackingPolicy
from mlrun.utils import logger
from server.py.services.api.api import deps
from server.py.services.api.api.endpoints.functions import (
    process_model_monitoring_secret,
)
from server.py.services.api.crud.model_monitoring.deployment import MonitoringDeployment

router = fastapi.APIRouter(prefix="/projects/{project}/jobs")


# TODO : delete when batch is deprecated.
@router.post("/batch-monitoring")
async def deploy_monitoring_batch_job(
    project: str,
    request: fastapi.Request,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        deps.authenticate_request
    ),
    db_session: Session = fastapi.Depends(deps.get_db_session),
    default_batch_image: str = "mlrun/mlrun",
    with_schedule: bool = False,
) -> dict[str, Any]:
    """
    Submit model monitoring batch job. By default, this API submit only the batch job as ML function without scheduling.
    To submit a scheduled job as well, please set with_schedule = True.

    :param project:             Project name.
    :param request:             fastapi request for the HTTP connection.
    :param auth_info:           The auth info of the request.
    :param db_session:          a session that manages the current dialog with the database.
    :param default_batch_image: The default image of the model monitoring batch job. By default, the image
                                is mlrun/mlrun.
    :param with_schedule:       If true, submit the model monitoring scheduled job as well.

    :return: model monitoring batch job as a dictionary.
    """

    await server.py.services.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        resource_type=mlrun.common.schemas.AuthorizationResourceTypes.function,
        project_name=project,
        resource_name=mlrun.common.schemas.model_monitoring.MonitoringFunctionNames.BATCH,
        action=mlrun.common.schemas.AuthorizationAction.store,
        auth_info=auth_info,
    )

    if with_schedule and (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting to deploy model monitoring batch job, re-routing to chief",
            function_name="model-monitoring-batch",
            project=project,
        )
        chief_client = server.py.services.api.utils.clients.chief.Client()
        params = {
            "default_batch_image": default_batch_image,
            "with_schedule": with_schedule,
        }
        return await chief_client.deploy_monitoring_batch_job(
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

    tracking_policy = TrackingPolicy(default_batch_image=default_batch_image)
    batch_function = MonitoringDeployment().deploy_model_monitoring_batch_processing(
        project=project,
        model_monitoring_access_key=model_monitoring_access_key,
        db_session=db_session,
        auth_info=auth_info,
        tracking_policy=tracking_policy,
        with_schedule=with_schedule,
    )

    if isinstance(batch_function, mlrun.runtimes.kubejob.KubejobRuntime):
        batch_function = batch_function.to_dict()

    return {
        "func": batch_function,
    }
