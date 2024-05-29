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
from mlrun.common.constants import MLRUN_FUNCTIONS_ANNOTATION


class APIGatewayAuthenticationMode(mlrun.common.types.StrEnum):
    basic = "basicAuth"
    none = "none"
    access_key = "accessKey"

    @classmethod
    def from_str(cls, authentication_mode: str):
        if authentication_mode == "none":
            return cls.none
        elif authentication_mode == "basicAuth":
            return cls.basic
        elif authentication_mode == "accessKey":
            return cls.access_key
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Authentication mode `{authentication_mode}` is not supported",
            )


class APIGatewayState(mlrun.common.types.StrEnum):
    none = ""
    ready = "ready"
    error = "error"
    waiting_for_provisioning = "waitingForProvisioning"


class _APIGatewayBaseModel(pydantic.BaseModel):
    class Config:
        extra = pydantic.Extra.allow


class APIGatewayMetadata(_APIGatewayBaseModel):
    name: str
    namespace: Optional[str]
    labels: Optional[dict] = {}
    annotations: Optional[dict] = {}


class APIGatewayBasicAuth(_APIGatewayBaseModel):
    username: str
    password: str


class APIGatewayUpstream(_APIGatewayBaseModel):
    kind: Optional[str] = "nucliofunction"
    nucliofunction: dict[str, str]
    percentage: Optional[int] = 0
    port: Optional[int] = 0


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
    state: Optional[APIGatewayState]


class APIGateway(_APIGatewayBaseModel):
    metadata: APIGatewayMetadata
    spec: APIGatewaySpec
    status: Optional[APIGatewayStatus]

    def get_function_names(self):
        return [
            upstream.nucliofunction.get("name")
            for upstream in self.spec.upstreams
            if upstream.nucliofunction.get("name")
        ]

    def enrich_mlrun_function_names(self):
        upstream_with_nuclio_names = []
        mlrun_function_uris = []
        for upstream in self.spec.upstreams:
            uri = upstream.nucliofunction.get("name")
            project, function_name, tag, _ = (
                mlrun.common.helpers.parse_versioned_object_uri(uri)
            )
            upstream.nucliofunction["name"] = (
                mlrun.runtimes.nuclio.function.get_fullname(function_name, project, tag)
            )

            upstream_with_nuclio_names.append(upstream)
            mlrun_function_uris.append(uri)

        self.spec.upstreams = upstream_with_nuclio_names
        if len(mlrun_function_uris) == 1:
            self.metadata.annotations[MLRUN_FUNCTIONS_ANNOTATION] = mlrun_function_uris[
                0
            ]
        elif len(mlrun_function_uris) == 2:
            self.metadata.annotations[MLRUN_FUNCTIONS_ANNOTATION] = "&".join(
                mlrun_function_uris
            )
        return self

    def replace_nuclio_names_with_mlrun_uri(self):
        mlrun_functions = self.metadata.annotations.get(MLRUN_FUNCTIONS_ANNOTATION)
        if mlrun_functions:
            mlrun_function_uris = (
                mlrun_functions.split("&")
                if "&" in mlrun_functions
                else [mlrun_functions]
            )
            if len(mlrun_function_uris) != len(self.spec.upstreams):
                raise mlrun.errors.MLRunValueError(
                    "Error when translating nuclio names to mlrun names in api gateway:"
                    "  number of functions doesn't match the mlrun functions in annotation"
                )
            for i in range(len(mlrun_function_uris)):
                self.spec.upstreams[i].nucliofunction["name"] = mlrun_function_uris[i]
        return self


class APIGatewaysOutput(_APIGatewayBaseModel):
    api_gateways: typing.Optional[dict[str, APIGateway]] = {}
