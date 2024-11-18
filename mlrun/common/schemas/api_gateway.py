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

import pydantic.v1

import mlrun.common.constants as mlrun_constants
import mlrun.common.types
from mlrun.common.constants import MLRUN_FUNCTIONS_ANNOTATION
from mlrun.common.helpers import generate_api_gateway_name


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


class _APIGatewayBaseModel(pydantic.v1.BaseModel):
    class Config:
        extra = pydantic.v1.Extra.allow


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
    authenticationMode: Optional[APIGatewayAuthenticationMode] = (  # noqa: N815 - for compatibility with Nuclio https://github.com/nuclio/nuclio/blob/672b8e36f9edd6e42b4685ec1d27cabae3c5f045/pkg/platform/types.go#L476
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

    def get_invoke_url(self):
        if self.spec.host and self.spec.path:
            return f"{self.spec.host.rstrip('/')}/{self.spec.path.lstrip('/')}".rstrip(
                "/"
            )
        return self.spec.host.rstrip("/")

    def enrich_mlrun_names(self):
        self._enrich_api_gateway_mlrun_name()
        self._enrich_mlrun_function_names()
        return self

    def replace_nuclio_names_with_mlrun_names(self):
        self._replace_nuclio_api_gateway_name_with_mlrun_name()
        self._replace_nuclio_function_names_with_mlrun_names()
        return self

    def _replace_nuclio_function_names_with_mlrun_names(self):
        # replace function names from nuclio names to mlrun names
        # and adds mlrun function URI's to an api gateway annotations
        # so when we then get api gateway entity from nuclio, we are able to get mlrun function names
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

    def _replace_nuclio_api_gateway_name_with_mlrun_name(self):
        # replace api gateway name
        # in Nuclio, api gateways are named as `<project>-<mlrun-api-gateway-name>`
        # remove the project prefix from the name if it exists
        project_name = self.metadata.labels.get(
            mlrun_constants.MLRunInternalLabels.nuclio_project_name
        )
        if project_name and self.spec.name.startswith(f"{project_name}-"):
            self.spec.name = self.spec.name[len(project_name) + 1 :]
            self.metadata.name = self.spec.name
        return self

    def _enrich_mlrun_function_names(self):
        # enrich mlrun names with nuclio prefixes
        # and add mlrun function's URIs to Nuclio function annotations
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

    def _enrich_api_gateway_mlrun_name(self):
        # replace api gateway name
        # in Nuclio, api gateways are named as `<project>-<mlrun-api-gateway-name>`
        # add the project prefix to the name
        project_name = self.metadata.labels.get(
            mlrun_constants.MLRunInternalLabels.nuclio_project_name
        )
        if project_name:
            self.spec.name = generate_api_gateway_name(project_name, self.spec.name)
            self.metadata.name = self.spec.name
        return self


class APIGatewaysOutput(_APIGatewayBaseModel):
    api_gateways: typing.Optional[dict[str, APIGateway]] = {}
