from base64 import b64decode
from http import HTTPStatus
from typing import Generator

from fastapi import Request
from sqlalchemy.orm import Session

from mlrun.api.api.utils import log_and_raise
from mlrun.api.db.session import create_session, close_session
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
        self.username = None
        self.password = None
        self.token = None

        cfg = config.httpdb

        header = request.headers.get("Authorization", "")
        if self._basic_auth_required(cfg):
            if not header.startswith(self._basic_prefix):
                log_and_raise(
                    HTTPStatus.UNAUTHORIZED.value, reason="missing basic auth"
                )
            user, password = self._parse_basic_auth(header)
            if user != cfg.user or password != cfg.password:
                log_and_raise(HTTPStatus.UNAUTHORIZED.value, reason="bad basic auth")
            self.username = user
            self.password = password
        elif self._bearer_auth_required(cfg):
            if not header.startswith(self._bearer_prefix):
                log_and_raise(
                    HTTPStatus.UNAUTHORIZED.value, reason="missing bearer auth"
                )
            token = header[len(self._bearer_prefix) :]
            if token != cfg.token:
                log_and_raise(HTTPStatus.UNAUTHORIZED.value, reason="bad basic auth")
            self.token = token

    @staticmethod
    def _basic_auth_required(cfg):
        return cfg.user or cfg.password

    @staticmethod
    def _bearer_auth_required(cfg):
        return cfg.token

    @staticmethod
    def _parse_basic_auth(header):
        """
        parse_basic_auth('Basic YnVnczpidW5ueQ==')
        ['bugs', 'bunny']
        """
        b64value = header[len(AuthVerifier._basic_prefix) :]
        value = b64decode(b64value).decode()
        return value.split(":", 1)
