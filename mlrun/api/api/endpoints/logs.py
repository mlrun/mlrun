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
import fastapi.concurrency
import sqlalchemy.orm

import mlrun.api.api.deps
import mlrun.api.crud
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier

router = fastapi.APIRouter()


@router.post("/log/{project}/{uid}")
async def store_log(
    request: fastapi.Request,
    project: str,
    uid: str,
    append: bool = True,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions,
        mlrun.api.schemas.AuthorizationResourceTypes.log,
        project,
        uid,
        mlrun.api.schemas.AuthorizationAction.store,
        auth_info,
    )
    body = await request.body()
    await fastapi.concurrency.run_in_threadpool(
        mlrun.api.crud.Logs().store_log,
        body,
        project,
        uid,
        append,
    )
    return {}


@router.get("/log/{project}/{uid}")
def get_log(
    project: str,
    uid: str,
    size: int = -1,
    offset: int = 0,
    auth_info: mlrun.api.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.api.schemas.AuthorizationResourceTypes.log,
        project,
        uid,
        mlrun.api.schemas.AuthorizationAction.read,
        auth_info,
    )
    run_state, log = mlrun.api.crud.Logs().get_logs(
        db_session, project, uid, size, offset
    )
    headers = {
        "x-mlrun-run-state": run_state,
        # pod_status was changed x-mlrun-run-state in 0.5.3, keeping it here for backwards compatibility (so <0.5.3
        # clients will work with the API)
        # TODO: remove this in 0.7.0
        "pod_status": run_state,
    }
    return fastapi.Response(content=log, media_type="text/plain", headers=headers)
