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

from framework.utils.clients.iguazio.base import BaseAsyncClient, BaseClient


class Client(BaseClient):
    def _generate_auth_info_from_session_verification_response(
        self,
        response_headers: typing.Mapping[str, typing.Any],
        response_body: typing.Mapping[typing.Any, typing.Any],
    ) -> mlrun.common.schemas.AuthInfo:
        raise NotImplementedError()

    @property
    def _verify_session_http_method(self) -> str:
        return mlrun.common.types.HTTPMethod.GET

    def _prepare_request_kwargs(
        self, session: typing.Optional[str], path: str, *, kwargs: dict
    ):
        raise NotImplementedError()

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


class AsyncClient(BaseAsyncClient, Client):
    pass
