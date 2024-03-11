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
from typing import Optional

import pydantic

import mlrun.common.types


class APIGatewayAuthenticationMode(mlrun.common.types.StrEnum):
    basic = "basicAuth"
    none = "none"

    @classmethod
    def from_str(cls, authentication_mode: str):
        if authentication_mode == "none":
            return cls.none
        elif authentication_mode == "basicAuth":
            return cls.basic
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Authentication mode `{authentication_mode}` is not supported",
            )


class _APIGatewayBaseModel(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.allow


class APIGatewayMetadata(_APIGatewayBaseModel):
    name: str
    namespace: Optional[str]
    labels: Optional[dict] = {}


class APIGatewayBasicAuth(_APIGatewayBaseModel):
    username: str
    password: str


class APIGatewayUpstream(_APIGatewayBaseModel):
    kind: Optional[str] = "nucliofunction"
    nucliofunction: dict[str, str]
    percentage: Optional[int] = 0


class APIGatewaySpec(_APIGatewayBaseModel):
    name: str
    description: Optional[str]
    path: Optional[str] = "/"
    authenticationMode: Optional[APIGatewayAuthenticationMode] = (
        APIGatewayAuthenticationMode.none
    )
    upstreams: list[APIGatewayUpstream]
    authentication: Optional[dict[str, Optional[APIGatewayBasicAuth]]]
    host: Optional[str]


class APIGatewayStatus(_APIGatewayBaseModel):
    name: Optional[str]
    state: Optional[str]


class APIGateway(_APIGatewayBaseModel):
    metadata: APIGatewayMetadata
    spec: APIGatewaySpec
    status: Optional[APIGatewayStatus]


class APIGatewaysOutput(_APIGatewayBaseModel):
    api_gateways: typing.Optional[dict[str, APIGateway]] = {}
