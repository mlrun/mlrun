import typing

import uvicorn.protocols.utils
from fastapi import Request
from sqlalchemy.orm import Session

import mlrun
import mlrun.api.db.session
import mlrun.api.schemas
import mlrun.api.utils.auth.verifier
import mlrun.api.utils.clients.iguazio


def get_db_session() -> typing.Generator[Session, None, None]:
    try:
        db_session = mlrun.api.db.session.create_session()
        yield db_session
    finally:
        mlrun.api.db.session.close_session(db_session)


def authenticate_request(request: Request) -> mlrun.api.schemas.AuthInfo:
    return mlrun.api.utils.auth.verifier.AuthVerifier().authenticate_request(request)


def verify_api_state(request: Request):
    path_with_query_string = uvicorn.protocols.utils.get_path_with_query_string(
        request.scope
    )
    if mlrun.mlconf.httpdb.state == mlrun.api.schemas.APIStates.offline:
        # we do want to stay healthy
        if "healthz" not in path_with_query_string:
            raise mlrun.errors.MLRunPreconditionFailedError("API is in offline state")
    if mlrun.mlconf.httpdb.state in [
        mlrun.api.schemas.APIStates.waiting_for_migrations,
        mlrun.api.schemas.APIStates.migration_in_progress,
    ]:
        enabled_endpoints = [
            "healthz",
            "background-tasks",
            "migrations",
        ]
        if not any(
            enabled_endpoint in path_with_query_string
            for enabled_endpoint in enabled_endpoints
        ):
            message = (
                "API is waiting for migration to be triggered. Send POST request to /api/migrations/start to tr"
                "igger it"
            )
            if (
                mlrun.mlconf.httpdb.state
                == mlrun.api.schemas.APIStates.migration_in_progress
            ):
                message = "Migration is in progress"
            raise mlrun.errors.MLRunPreconditionFailedError(message)
