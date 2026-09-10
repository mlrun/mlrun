# Copyright 2026 Iguazio
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

import fastapi

import mlrun.common.schemas

import framework.utils.clients.chief
import framework.utils.singletons.project_member

router = fastapi.APIRouter()


@router.get("/followers-sync-status")
async def followers_sync_status():
    """
    Lets a worker ask the chief whether the chief's own followers-sync trigger (Follower
    Contract HLD checklist item 10) has reached a terminal state yet, instead of each replica
    independently triggering its own sweep against the leader.
    """
    if (
        mlrun.mlconf.httpdb.clusterization.role
        != mlrun.common.schemas.ClusterizationRole.chief
    ):
        chief_client = framework.utils.clients.chief.Client()
        return {"ready": await chief_client.is_followers_sync_ready()}

    return {
        "ready": framework.utils.singletons.project_member.get_project_member().is_followers_sync_ready()
    }
