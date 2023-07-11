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

import uvicorn.protocols.utils
from fastapi import Request
from sqlalchemy.orm import Session

import mlrun
import mlrun.api.db.session
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.clients.iguazio
import mlrun.common.schemas


def get_db_session() -> typing.Generator[Session, None, None]:
    db_session = None
    try:
        db_session = mlrun.api.db.session.create_session()
        yield db_session
    finally:
        if db_session:
            mlrun.api.db.session.close_session(db_session)


async def authenticate_request(request: Request) -> mlrun.common.schemas.AuthInfo:
    return await mlrun.api.utils.auth.verifier.AuthVerifier().authenticate_request(
        request
    )


def verify_api_state(request: Request):
    path_with_query_string = uvicorn.protocols.utils.get_path_with_query_string(
        request.scope
    )
    path = path_with_query_string.split("?")[0]
    if mlrun.mlconf.httpdb.state == mlrun.common.schemas.APIStates.offline:
        enabled_endpoints = [
            # we want to stay healthy
            "healthz",
            # we want the workers to be able to pull chief state even if the state is offline
            "clusterization-spec",
        ]
        if not any(enabled_endpoint in path for enabled_endpoint in enabled_endpoints):
            raise mlrun.errors.MLRunPreconditionFailedError("API is in offline state")
    if mlrun.mlconf.httpdb.state in [
        mlrun.common.schemas.APIStates.waiting_for_migrations,
        mlrun.common.schemas.APIStates.migrations_in_progress,
        mlrun.common.schemas.APIStates.migrations_failed,
        mlrun.common.schemas.APIStates.waiting_for_chief,
    ]:
        enabled_endpoints = [
            "healthz",
            "background-tasks",
            "client-spec",
            "migrations",
            "clusterization-spec",
            "memory-reports",
        ]
        if not any(enabled_endpoint in path for enabled_endpoint in enabled_endpoints):
            message = mlrun.common.schemas.APIStates.description(
                mlrun.mlconf.httpdb.state
            )
            raise mlrun.errors.MLRunPreconditionFailedError(message)


def expose_internal_endpoints(request: Request):
    if not mlrun.mlconf.debug.expose_internal_api_endpoints:
        path_with_query_string = uvicorn.protocols.utils.get_path_with_query_string(
            request.scope
        )
        path = path_with_query_string.split("?")[0]
        if "/_internal" in path:
            raise mlrun.errors.MLRunPreconditionFailedError(
                "Internal endpoints are not exposed"
            )
