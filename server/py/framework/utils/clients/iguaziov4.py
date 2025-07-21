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


import fastapi

import mlrun.common.schemas
import mlrun.errors

from framework.utils.clients.base_client import BaseAsyncClient, BaseClient


class ClientV4(BaseClient):
    pass


class AsyncClient(BaseAsyncClient, ClientV4):
    async def verify_request_session(
        self, request: fastapi.Request
    ) -> mlrun.common.schemas.AuthInfo:
        """
        Verifies the session of the incoming request for IG4 environments:
        Requires either an Authorization header (JWT) or _oauth2_proxy cookie.
        """
        authorization = request.headers.get("authorization")
        oauth2_cookie = request.cookies.get("_oauth2_proxy")

        # Enforce presence of at least one valid auth credential
        if not authorization and not oauth2_cookie:
            raise mlrun.errors.MLRunUnauthorizedError(
                "Missing authentication credentials: expected either Authorization header (JWT) or _oauth2_proxy cookie"
            )

        user_info = await self._fetch_user_info(request)

        return self._parse_auth_info_from_user_info(
            user_info, session=authorization or oauth2_cookie
        )

    async def _fetch_user_info(self, request: fastapi.Request) -> dict:
        raise NotImplementedError()

    def _parse_auth_info_from_user_info(
        self, user_info: dict, session: str
    ) -> mlrun.common.schemas.AuthInfo:
        username = user_info.get("username")
        group_ids = user_info.get("groups", [])

        if not username:
            raise mlrun.errors.MLRunUnauthorizedError(
                "Received invalid user identity from Iguazio"
            )

        return mlrun.common.schemas.AuthInfo(
            username=username,
            session=session,
            user_group_ids=group_ids,
        )
