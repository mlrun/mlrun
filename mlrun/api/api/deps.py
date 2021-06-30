from base64 import b64decode
from http import HTTPStatus
from typing import Generator

from fastapi import Request
from sqlalchemy.orm import Session

import mlrun.api.utils.authorizers.authorizer
import mlrun.api.utils.authorizers.nop
import mlrun.api.utils.authorizers.opa
import mlrun.api.utils.clients.iguazio
from mlrun.api.api.utils import log_and_raise
from mlrun.api.db.session import close_session, create_session
from mlrun.config import config


def get_db_session() -> Generator[Session, None, None]:
    try:
        db_session = create_session()
        yield db_session
    finally:
        close_session(db_session)


class AuthVerifier:
    _basic_prefix = "Basic "
    _bearer_prefix = "Bearer "

    def __init__(self, request: Request):
        # Basic auth
        self.username = None
        self.password = None
        # Bearer auth
        self.token = None
        # Iguazio auth
        self.session = None
        self.data_session = None
        self.uid = None
        self.gids = None

        self._authenticate_request(request)
        self._authorize_request(request)

    def _authorize_request(self, request: Request):
        if config.httpdb.authorization.mode == "none":
            authorizer = mlrun.api.utils.authorizers.nop.Authorizer()
        elif config.httpdb.authorization.mode == "opa":
            authorizer = mlrun.api.utils.authorizers.opa.Authorizer()
        else:
            raise NotImplementedError(
                f"Configured authorization mode is not supported. mode={config.httpdb.authorization.mode}"
            )
        authorizer.authorize(request)

    def _authenticate_request(self, request: Request):
        header = request.headers.get("Authorization", "")
        if self._basic_auth_required():
            if not header.startswith(self._basic_prefix):
                log_and_raise(
                    HTTPStatus.UNAUTHORIZED.value, reason="Missing basic auth header"
                )
            username, password = self._parse_basic_auth(header)
            if (
                username != config.httpdb.authentication.basic.username
                or password != config.httpdb.authentication.basic.password
            ):
                log_and_raise(
                    HTTPStatus.UNAUTHORIZED.value,
                    reason="Username or password did not match",
                )
            self.username = username
            self.password = password
        elif self._bearer_auth_required():
            if not header.startswith(self._bearer_prefix):
                log_and_raise(
                    HTTPStatus.UNAUTHORIZED.value, reason="Missing bearer auth header"
                )
            token = header[len(self._bearer_prefix) :]
            if token != config.httpdb.authentication.bearer.token:
                log_and_raise(
                    HTTPStatus.UNAUTHORIZED.value, reason="Token did not match"
                )
            self.token = token
        elif self._iguazio_auth_required():
            iguazio_client = mlrun.api.utils.clients.iguazio.Client()
            (
                self.username,
                self.session,
                self.uid,
                self.gids,
                planes,
            ) = iguazio_client.verify_request_session(request)
            if "x-data-session-override" in request.headers:
                self.data_session = request.headers["x-data-session-override"]
            elif "data" in planes:
                self.data_session = self.session

    @staticmethod
    def _basic_auth_required():
        return config.httpdb.authentication.mode == "basic" and (
            config.httpdb.authentication.basic.username
            or config.httpdb.authentication.basic.password
        )

    @staticmethod
    def _bearer_auth_required():
        return (
            config.httpdb.authentication.mode == "bearer"
            and config.httpdb.authentication.bearer.token
        )

    @staticmethod
    def _iguazio_auth_required():
        return config.httpdb.authentication.mode == "iguazio"

    @staticmethod
    def _parse_basic_auth(header):
        """
        parse_basic_auth('Basic YnVnczpidW5ueQ==')
        ['bugs', 'bunny']
        """
        b64value = header[len(AuthVerifier._basic_prefix) :]
        value = b64decode(b64value).decode()
        return value.split(":", 1)
