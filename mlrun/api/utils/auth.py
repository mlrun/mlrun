import base64
import http

import fastapi

import mlrun
import mlrun.api.api.utils
import mlrun.api.schemas
import mlrun.api.utils.clients.iguazio
import mlrun.utils.singleton


class AuthVerifier(metaclass=mlrun.utils.singleton.Singleton):
    _basic_prefix = "Basic "
    _bearer_prefix = "Bearer "

    def authenticate_request(
        self, request: fastapi.Request
    ) -> mlrun.api.schemas.AuthInfo:
        auth_info = mlrun.api.schemas.AuthInfo()
        header = request.headers.get("Authorization", "")
        if self._basic_auth_configured():
            if not header.startswith(self._basic_prefix):
                mlrun.api.api.utils.log_and_raise(
                    http.HTTPStatus.UNAUTHORIZED.value,
                    reason="Missing basic auth header",
                )
            username, password = self._parse_basic_auth(header)
            if (
                username != mlrun.mlconf.httpdb.authentication.basic.username
                or password != mlrun.mlconf.httpdb.authentication.basic.password
            ):
                mlrun.api.api.utils.log_and_raise(
                    http.HTTPStatus.UNAUTHORIZED.value,
                    reason="Username or password did not match",
                )
            auth_info.username = username
            auth_info.password = password
        elif self._bearer_auth_configured():
            if not header.startswith(self._bearer_prefix):
                mlrun.api.api.utils.log_and_raise(
                    http.HTTPStatus.UNAUTHORIZED.value,
                    reason="Missing bearer auth header",
                )
            token = header[len(self._bearer_prefix) :]
            if token != mlrun.mlconf.httpdb.authentication.bearer.token:
                mlrun.api.api.utils.log_and_raise(
                    http.HTTPStatus.UNAUTHORIZED.value, reason="Token did not match"
                )
            auth_info.token = token
        elif self._iguazio_auth_configured():
            iguazio_client = mlrun.api.utils.clients.iguazio.Client()
            auth_info = iguazio_client.verify_request_session(request)
            if "x-data-session-override" in request.headers:
                auth_info.data_session = request.headers["x-data-session-override"]

        # Fallback in case auth method didn't fill in the username already, and it is provided by the caller
        if not auth_info.username and "x-remote-user" in request.headers:
            auth_info.username = request.headers["x-remote-user"]

        projects_role_header = request.headers.get(
            mlrun.api.schemas.HeaderNames.projects_role
        )
        auth_info.projects_role = (
            mlrun.api.schemas.ProjectsRole(projects_role_header)
            if projects_role_header
            else None
        )
        # In Iguazio 3.0 we're running with auth mode none cause auth is done by the ingress, in that auth mode sessions
        # needed for data operations were passed through this header, keep reading it to be backwards compatible
        if not auth_info.data_session and "X-V3io-Session-Key" in request.headers:
            auth_info.data_session = request.headers["X-V3io-Session-Key"]
        return auth_info

    def generate_auth_info_from_session(
        self, session: str
    ) -> mlrun.api.schemas.AuthInfo:
        if not self._iguazio_auth_configured():
            raise NotImplementedError(
                "Session is currently supported only for iguazio authentication mode"
            )
        return mlrun.api.utils.clients.iguazio.Client().verify_session(session)

    @staticmethod
    def _basic_auth_configured():
        return mlrun.mlconf.httpdb.authentication.mode == "basic" and (
            mlrun.mlconf.httpdb.authentication.basic.username
            or mlrun.mlconf.httpdb.authentication.basic.password
        )

    @staticmethod
    def _bearer_auth_configured():
        return (
            mlrun.mlconf.httpdb.authentication.mode == "bearer"
            and mlrun.mlconf.httpdb.authentication.bearer.token
        )

    @staticmethod
    def _iguazio_auth_configured():
        return mlrun.mlconf.httpdb.authentication.mode == "iguazio"

    @staticmethod
    def _parse_basic_auth(header):
        """
        parse_basic_auth('Basic YnVnczpidW5ueQ==')
        ['bugs', 'bunny']
        """
        b64value = header[len(AuthVerifier._basic_prefix) :]
        value = base64.b64decode(b64value).decode()
        return value.split(":", 1)
