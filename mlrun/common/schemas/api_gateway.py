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
from typing import Dict, List, Optional

import pydantic

import mlrun.common.types


class APIGatewayAuthenticationMode(mlrun.common.types.StrEnum):
    @classmethod
    def from_str(cls, authentication_mode: str):
        if authentication_mode == "none":
            return cls.none
        elif authentication_mode == "basicAuth":
            return cls.basic
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Authentication mode `{authentication_mode}` is not supported",
                authentication_mode=authentication_mode,
            )

    basic = "basicAuth"
    none = "none"


class APIGatewayMetadata(pydantic.BaseModel):
    name: str
    namespace: Optional[str]
    labels: Optional[dict] = {}

    class Config:
        extra = pydantic.Extra.allow


class APIGatewayBasicAuth(pydantic.BaseModel):
    username: str
    password: str


class APIGatewayUpstream(pydantic.BaseModel):
    kind: Optional[str] = "nucliofunction"
    nucliofunction: Dict[str, str]
    percentage: Optional[int] = 0


class APIGatewaySpec(pydantic.BaseModel):
    name: str
    description: Optional[str]
    path: Optional[str] = "/"
    authenticationMode: Optional[
        APIGatewayAuthenticationMode
    ] = APIGatewayAuthenticationMode.none
    upstreams: List[APIGatewayUpstream]
    authentication: Optional[Dict[str, Optional[APIGatewayBasicAuth]]]
    host: Optional[str]

    class Config:
        extra = pydantic.Extra.allow


class APIGatewayStatus(pydantic.BaseModel):
    name: Optional[str]
    state: Optional[str]

    class Config:
        extra = pydantic.Extra.allow


class APIGateway(pydantic.BaseModel):
    metadata: APIGatewayMetadata
    spec: APIGatewaySpec
    status: Optional[APIGatewayStatus]

    class Config:
        extra = pydantic.Extra.allow


class APIGateways(pydantic.BaseModel):
    api_gateways: typing.Optional[Dict[str, APIGateway]] = {}
