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
import sqlalchemy.orm
from fastapi.concurrency import run_in_threadpool

import mlrun.api.api.deps
import mlrun.api.crud
import mlrun.api.utils.auth.verifier
import mlrun.common.schemas

router = fastapi.APIRouter()


# TODO: remove /log/{project}/{uid} in 1.7.0
@router.post(
    "/log/{project}/{uid}",
    deprecated=True,
    description="/log/{project}/{uid} is deprecated in 1.5.0 and will be removed in 1.7.0, "
    "use /projects/{project}/logs/{uid} instead",
)
@router.post("/projects/{project}/logs/{uid}")
async def store_log(
    request: fastapi.Request,
    project: str,
    uid: str,
    append: bool = True,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.log,
        project,
        uid,
        mlrun.common.schemas.AuthorizationAction.store,
        auth_info,
    )
    body = await request.body()
    await run_in_threadpool(
        mlrun.api.crud.Logs().store_log,
        body,
        project,
        uid,
        append,
    )
    return {}


# TODO: remove /log/{project}/{uid} in 1.7.0
@router.get(
    "/log/{project}/{uid}",
    deprecated=True,
    description="/log/{project}/{uid} is deprecated in 1.5.0 and will be removed in 1.7.0, "
    "use /projects/{project}/logs/{uid} instead",
)
@router.get("/projects/{project}/logs/{uid}")
async def get_log(
    project: str,
    uid: str,
    size: int = -1,
    offset: int = 0,
    auth_info: mlrun.common.schemas.AuthInfo = fastapi.Depends(
        mlrun.api.api.deps.authenticate_request
    ),
    db_session: sqlalchemy.orm.Session = fastapi.Depends(
        mlrun.api.api.deps.get_db_session
    ),
):
    await mlrun.api.utils.auth.verifier.AuthVerifier().query_project_resource_permissions(
        mlrun.common.schemas.AuthorizationResourceTypes.log,
        project,
        uid,
        mlrun.common.schemas.AuthorizationAction.read,
        auth_info,
    )
    run_state, log_stream = await mlrun.api.crud.Logs().get_logs(
        db_session, project, uid, size, offset
    )
    headers = {
        "x-mlrun-run-state": run_state,
    }
    return fastapi.responses.StreamingResponse(
        log_stream,
        media_type="text/plain",
        headers=headers,
    )
