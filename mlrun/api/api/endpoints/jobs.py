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
from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

import mlrun.common.schemas
from mlrun.api.api import deps
from mlrun.api.api.endpoints.functions import process_model_monitoring_secret
from mlrun.api.crud.model_monitoring.deployment import MonitoringDeployment
from mlrun.model_monitoring import TrackingPolicy

router = APIRouter(prefix="/projects/{project}/jobs")


@router.post("/batch-monitoring")
def deploy_monitoring_batch_job(
    project: str,
    auth_info: mlrun.common.schemas.AuthInfo = Depends(deps.authenticate_request),
    db_session: Session = Depends(deps.get_db_session),
    default_batch_image: str = "mlrun/mlrun",
    with_schedule: bool = False,
) -> Dict[str, Any]:
    """
    Submit model monitoring batch job. By default, this API submit only the batch job as ML function without scheduling.
    To submit a scheduled job as well, please set with_schedule = True.

    :param project:             Project name.
    :param auth_info:           The auth info of the request.
    :param db_session:          a session that manages the current dialog with the database.
    :param default_batch_image: The default image of the model monitoring batch job. By default, the image
                                is mlrun/mlrun.
    :param with_schedule:       If true, submit the model monitoring scheduled job as well.

    :return: model monitoring batch job as a dictionary.
    """

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
