# Copyright 2018 Iguazio
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

import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.clients.chief
from mlrun.utils import logger

router = fastapi.APIRouter()


@router.get("/clusterization-spec", response_model=mlrun.api.schemas.ClusterizationSpec)
def clusterization_spec():
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.api.schemas.ClusterizationRole.chief
    ):
        logger.info(
            "Requesting clusterization spec from worker, re-routing to chief",
        )
        chief_client = mlrun.api.utils.clients.chief.Client()
        return chief_client.get_clusterization_spec()

    return mlrun.api.crud.ClusterizationSpec().get_clusterization_spec()
