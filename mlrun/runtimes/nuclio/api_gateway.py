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
import base64
import typing
from typing import Optional, Union
from urllib.parse import urljoin

import requests
from requests.auth import HTTPBasicAuth

import mlrun
import mlrun.common.schemas

from .function import RemoteRuntime, get_fullname
from .serving import ServingRuntime

NUCLIO_API_GATEWAY_AUTHENTICATION_MODE_BASIC_AUTH = "basicAuth"
NUCLIO_API_GATEWAY_AUTHENTICATION_MODE_NONE = "none"
PROJECT_NAME_LABEL = "nuclio.io/project-name"


class APIGatewayAuthenticator(typing.Protocol):
    @property
    def authentication_mode(self) -> str:
        return NUCLIO_API_GATEWAY_AUTHENTICATION_MODE_NONE

    @classmethod
    def from_scheme(cls, api_gateway_spec: mlrun.common.schemas.APIGatewaySpec):
        if (
            api_gateway_spec.authenticationMode
            == NUCLIO_API_GATEWAY_AUTHENTICATION_MODE_BASIC_AUTH
        ):
            if api_gateway_spec.authentication:
                return BasicAuth(
                    username=api_gateway_spec.authentication.get("username", ""),
                    password=api_gateway_spec.authentication.get("password", ""),
                )
            else:
                return BasicAuth()
        else:
            return NoneAuth()

    def to_scheme(
        self,
    ) -> Optional[dict[str, Optional[mlrun.common.schemas.APIGatewayBasicAuth]]]:
        return None


class NoneAuth(APIGatewayAuthenticator):
    pass


class BasicAuth(APIGatewayAuthenticator):
    def __init__(self, username=None, password=None):
        self._username = username
        self._password = password

    @property
    def authentication_mode(self) -> str:
        return NUCLIO_API_GATEWAY_AUTHENTICATION_MODE_BASIC_AUTH

    def to_scheme(
        self,
    ) -> Optional[dict[str, Optional[mlrun.common.schemas.APIGatewayBasicAuth]]]:
        return {
            "authentication": mlrun.common.schemas.APIGatewayBasicAuth(
                username=self._username, password=self._password
            )
        }


class APIGateway:
    def __init__(
        self,
        project,
        name: str,
        functions: Union[
            list[str],
            Union[
                list[
                    Union[
                        RemoteRuntime,
                        ServingRuntime,
                    ]
                ],
                Union[RemoteRuntime, ServingRuntime],
            ],
        ],
        description: str = "",
        path: str = "/",
        authentication: Optional[APIGatewayAuthenticator] = NoneAuth(),
        host: Optional[str] = None,
        canary: Optional[list[int]] = None,
    ):
        self.functions = None
        self._validate(
            project=project,
            functions=functions,
            name=name,
            canary=canary,
        )
        self.project = project
        self.name = name
        self.host = host

        self.path = path
        self.description = description
        self.canary = canary
        self.authentication = authentication

    def invoke(
        self,
        method="POST",
        headers: dict = {},
        auth: Optional[tuple[str, str]] = None,
        **kwargs,
    ):
        if not self.invoke_url:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Invocation url is not set. Set up gateway's `invoke_url` attribute."
            )
        if (
            self.authentication.authentication_mode
            == NUCLIO_API_GATEWAY_AUTHENTICATION_MODE_BASIC_AUTH
            and not auth
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "API Gateway invocation requires authentication. Please pass credentials"
            )
        return requests.request(
            method=method,
            url=self.invoke_url,
            headers=headers,
            **kwargs,
            auth=HTTPBasicAuth(*auth) if auth else None,
        )

    def with_basic_auth(self, username: str, password: str):
        self.authentication = BasicAuth(username=username, password=password)

    def with_canary(
        self,
        functions: Union[
            list[str],
            list[
                Union[
                    RemoteRuntime,
                    ServingRuntime,
                ]
            ],
        ],
        canary: list[int],
    ):
        if len(functions) != 2:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Gateway with canary can be created only with two functions, "
                f"the number of functions passed is {len(functions)}"
            )
        self.functions = self._validate_functions(self.project, functions)
        self.canary = self._validate_canary(canary)

    @classmethod
    def from_scheme(cls, api_gateway: mlrun.common.schemas.APIGateway):
        project = api_gateway.metadata.labels.get(PROJECT_NAME_LABEL)
        functions, canary = cls._resolve_canary(api_gateway.spec.upstreams)
        return cls(
            project=project,
            description=api_gateway.spec.description,
            name=api_gateway.spec.name,
            host=api_gateway.spec.host,
            path=api_gateway.spec.path,
            authentication=APIGatewayAuthenticator.from_scheme(api_gateway.spec),
            functions=functions,
            canary=canary,
        )

    def to_scheme(self) -> mlrun.common.schemas.APIGateway:
        upstreams = (
            [
                mlrun.common.schemas.APIGatewayUpstream(
                    nucliofunction={"name": function_name},
                    percentage=percentage,
                )
                for function_name, percentage in zip(self.functions, self.canary)
            ]
            if self.canary
            else [
                mlrun.common.schemas.APIGatewayUpstream(
                    nucliofunction={"name": function_name},
                )
                for function_name in self.functions
            ]
        )
        api_gateway = mlrun.common.schemas.APIGateway(
            metadata=mlrun.common.schemas.APIGatewayMetadata(name=self.name, labels={}),
            spec=mlrun.common.schemas.APIGatewaySpec(
                name=self.name,
                description=self.description,
                host=self.host,
                path=self.path,
                authenticationMode=mlrun.common.schemas.APIGatewayAuthenticationMode.from_str(
                    self.authentication.authentication_mode
                ),
                upstreams=upstreams,
            ),
        )
        api_gateway.spec.authentication = self.authentication.to_scheme()
        return api_gateway

    @property
    def invoke_url(
        self,
    ):
        return urljoin(self.host, self.path)

    def _validate(
        self,
        name: str,
        project: str,
        functions: Union[
            list[str],
            Union[
                list[
                    Union[
                        RemoteRuntime,
                        ServingRuntime,
                    ]
                ],
                Union[RemoteRuntime, ServingRuntime],
            ],
        ],
        canary: Optional[list[int]] = None,
    ):
        if not name:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "API Gateway name cannot be empty"
            )

        self.functions = self._validate_functions(project=project, functions=functions)

        # validating canary
        if canary:
            self._validate_canary(canary)

    def _validate_canary(self, canary: list[int]):
        if len(self.functions) != len(canary):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Function and canary lists lengths do not match"
            )
        for canary_percent in canary:
            if canary_percent < 0 or canary_percent > 100:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "The percentage value must be in the range from 0 to 100"
                )
        if sum(canary) != 100:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "The sum of canary function percents should be equal to 100"
            )
        return canary

    @staticmethod
    def _validate_functions(
        project: str,
        functions: Union[
            list[str],
            Union[
                list[
                    Union[
                        RemoteRuntime,
                        ServingRuntime,
                    ]
                ],
                Union[RemoteRuntime, ServingRuntime],
            ],
        ],
    ):
        if not isinstance(functions, list):
            functions = [functions]

        # validating functions
        if not 1 <= len(functions) <= 2:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Gateway can be created from one or two functions, "
                f"the number of functions passed is {len(functions)}"
            )

        function_names = []
        for func in functions:
            if isinstance(func, str):
                function_names.append(func)
                continue

            function_name = (
                func.metadata.name if hasattr(func, "metadata") else func.name
            )
            if func.kind not in mlrun.runtimes.RuntimeKinds.nuclio_runtimes():
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Input function {function_name} is not a Nuclio function"
                )
            if func.metadata.project != project:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"input function {function_name} "
                    f"does not belong to this project"
                )
            nuclio_name = get_fullname(function_name, project, func.metadata.tag)
            function_names.append(nuclio_name)
        return function_names

    @staticmethod
    def _generate_basic_auth(username: str, password: str):
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        return f"Basic {token}"

    @staticmethod
    def _resolve_canary(
        upstreams: list[mlrun.common.schemas.APIGatewayUpstream],
    ) -> tuple[Union[list[str], None], Union[list[int], None]]:
        if len(upstreams) == 1:
            return [upstreams[0].nucliofunction.get("name")], None
        elif len(upstreams) == 2:
            canary = [0, 0]
            functions = [
                upstreams[0].nucliofunction.get("name"),
                upstreams[1].nucliofunction.get("name"),
            ]
            percentage_1 = upstreams[0].percentage
            percentage_2 = upstreams[1].percentage

            if not percentage_1 and percentage_2:
                percentage_1 = 100 - percentage_2
            if not percentage_2 and percentage_1:
                percentage_2 = 100 - percentage_1
            if percentage_1 and percentage_2:
                canary = [percentage_1, percentage_2]
            return functions, canary
        else:
            # Nuclio only supports 1 or 2 upstream functions
            return None, None
