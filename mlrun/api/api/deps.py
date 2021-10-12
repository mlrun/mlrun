import typing

from fastapi import Request
from sqlalchemy.orm import Session

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
