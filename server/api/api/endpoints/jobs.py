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
from http import HTTPStatus

import fastapi
from fastapi import Header
from sqlalchemy.orm import Session

import mlrun.common.schemas
import server.api.api.utils
import server.api.utils.auth.verifier
from server.api import MINIMUM_CLIENT_VERSION_FOR_MM
from server.api.api import deps

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
    Deprecated.

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
    server.api.api.utils.log_and_raise(
        HTTPStatus.BAD_REQUEST.value,
        reason=f"Model monitoring is supported from client version {MINIMUM_CLIENT_VERSION_FOR_MM}. "
        f"Please upgrade your client accordingly.",
    )
