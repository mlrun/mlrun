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

import typing

import mlrun.common.schemas
import mlrun.common.types
import mlrun.errors
from mlrun.utils import get_in

from framework.utils.clients.iguazio.base import BaseAsyncClient, BaseClient

_GROUP_TYPE_KEY = "@type"
_GROUP_TYPE_VALUE = "type.googleapis.com/group.Group"


class Client(BaseClient):
    def _generate_auth_info_from_session_verification_response(
        self,
        response_headers: typing.Mapping[str, typing.Any],
        response_body: typing.Mapping[typing.Any, typing.Any],
    ) -> mlrun.common.schemas.AuthInfo:
        username, group_ids = self._parse_auth_response_data(response_body)
        return mlrun.common.schemas.AuthInfo(
            username=username,
            user_group_ids=group_ids,
        )

    @property
    def _verify_session_http_method(self) -> str:
        return mlrun.common.types.HTTPMethod.GET

    def _prepare_request_kwargs(
        self, session: typing.Optional[str], path: str, *, kwargs: dict
    ):
        headers = kwargs.setdefault("headers", {})

        # Accept an Authorization header or a session cookie named "_oauth2_proxy"
        authorization = headers.get(mlrun.common.schemas.HeaderNames.authorization, "")
        cookie = headers.get(mlrun.common.schemas.HeaderNames.cookies, "")

        has_auth = (
            bool(authorization)
            or mlrun.common.schemas.CookieNames.oauth2_proxy in cookie
        )

        if not has_auth:
            raise mlrun.errors.MLRunUnauthorizedError(
                "Request must include either an Authorization header or _oauth2_proxy cookie"
            )

    # TODO: implement this method
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
    def _parse_auth_response_data(
        response_body: typing.Mapping[typing.Any, typing.Any],
    ) -> tuple[str, list[str]]:
        if not isinstance(response_body, dict):
            raise mlrun.errors.MLRunBadRequestError("Expected dict in response body")

        username = get_in(response_body, "metadata.username", "")
        if not username:
            raise mlrun.errors.MLRunUnauthorizedError(
                "Missing or empty username in authentication response"
            )

        group_ids = []
        for relationship in response_body.get("relationships", []):
            if relationship.get(_GROUP_TYPE_KEY) == _GROUP_TYPE_VALUE and get_in(
                relationship, "metadata.id"
            ):
                group_ids.append(relationship["metadata"]["id"])

        if not group_ids:
            raise mlrun.errors.MLRunUnauthorizedError(
                "Missing or empty group IDs in authentication response"
            )

        return username, group_ids


class AsyncClient(BaseAsyncClient, Client):
    """Asynchronous implementation of the Iguazio V4 client. Inherits logic from Client and BaseAsyncClient."""

    pass
