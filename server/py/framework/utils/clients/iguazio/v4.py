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

import http
import typing

import mlrun.common.schemas
import mlrun.errors
from mlrun.utils import get_in

from framework.utils.clients.iguazio.base import BaseAsyncClient, BaseClient


class Client(BaseClient):
    def _generate_auth_info_from_session_verification_response(
        self,
        response_headers: typing.Mapping[str, typing.Any],
        response_body: typing.Mapping[typing.Any, typing.Any],
    ) -> mlrun.common.schemas.AuthInfo:
        username, group_ids = self._resolve_params_from_response_body(response_body)
        auth_info = mlrun.common.schemas.AuthInfo(
            username=username,
            user_group_ids=group_ids,
        )
        return auth_info

    @property
    def _verify_session_http_method(self) -> str:
        return http.HTTPMethod.GET

    def _prepare_request_kwargs(
        self, session: typing.Optional[str], path: str, *, kwargs: dict
    ):
        headers = kwargs.setdefault("headers", {})

        # Accept an Authorization header or a session cookie named "_oauth2_proxy"
        authorization = headers.get("authorization") or headers.get("Authorization")
        cookie = headers.get("cookie", "")

        has_auth = bool(authorization) or "_oauth2_proxy=" in cookie

        if not has_auth:
            raise mlrun.errors.MLRunUnauthorizedError(
                "Request must include either an Authorization header or _oauth2_proxy cookie"
            )

        # Ensure headers are lowercase consistent
        if authorization:
            headers["authorization"] = authorization

    def _handle_error_response(
        self,
        method: str,
        path: str,
        response: typing.Any,
        response_body: dict,
        error_message: str,
        kwargs: dict,
    ) -> None:
        raise NotImplementedError()

    @staticmethod
    def _resolve_params_from_response_body(
        response_body: typing.Mapping[typing.Any, typing.Any],
    ) -> tuple[typing.Optional[str], typing.Optional[list[str]]]:
        username = get_in(response_body, "metadata.username", "")

        group_ids = []
        for relationship in response_body.get("relationships", []):
            if relationship.get(
                "@type"
            ) == "type.googleapis.com/group.Group" and get_in(
                relationship, "metadata.id"
            ):
                group_ids.append(relationship["metadata"]["id"])

        return username, group_ids


class AsyncClient(BaseAsyncClient, Client):
    pass
