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
import http

from fastapi import APIRouter

import mlrun.common.schemas
from mlrun.config import config as mlconfig

router = APIRouter()


@router.get(
    "/healthz",
    status_code=http.HTTPStatus.OK.value,
)
def health():

    # offline is the initial state
    # waiting for chief is set for workers waiting for chief to be ready and then clusterize against it
    if mlconfig.httpdb.state in [
        mlrun.common.schemas.APIStates.offline,
        mlrun.common.schemas.APIStates.waiting_for_chief,
    ]:
        raise mlrun.errors.MLRunServiceUnavailableError()

    return {
        # for old `align_mlrun.sh` scripts expecting `version` in the response
        # TODO: remove on mlrun >= 1.6.0
        "version": mlconfig.version,
        "status": "ok",
    }
