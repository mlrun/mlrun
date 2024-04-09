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

from ..utils import logger
from .function import RemoteRuntime, get_fullname, min_nuclio_versions
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
    """
    An API gateway authenticator with no authentication.
    """

    pass


class BasicAuth(APIGatewayAuthenticator):
    """
    An API gateway authenticator with basic authentication.

    :param username: (str) The username for basic authentication.
    :param password: (str) The password for basic authentication.
    """

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
            "basicAuth": mlrun.common.schemas.APIGatewayBasicAuth(
                username=self._username, password=self._password
            )
        }


class APIGateway:
    @min_nuclio_versions("1.13.1")
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
        """
        Initialize the APIGateway instance.

        :param project: The project name
        :param name: The name of the API gateway
        :param functions: The list of functions associated with the API gateway
            Can be a list of function names (["my-func1", "my-func2"])
            or a list or a single entity of
            :py:class:`~mlrun.runtimes.nuclio.function.RemoteRuntime` OR
            :py:class:`~mlrun.runtimes.nuclio.serving.ServingRuntime`

        :param description: Optional description of the API gateway
        :param path: Optional path of the API gateway, default value is "/"
        :param authentication: The authentication for the API gateway of type
                :py:class:`~mlrun.runtimes.nuclio.api_gateway.BasicAuth`
        :param host:  The host of the API gateway (optional). If not set, it will be automatically generated
        :param canary: The canary percents for the API gateway of type list[int]; for instance: [20,80]
        """
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
        self.state = ""

    def invoke(
        self,
        method="POST",
        headers: dict = {},
        auth: Optional[tuple[str, str]] = None,
        **kwargs,
    ):
        """
        Invoke the API gateway.

        :param method: (str, optional) The HTTP method for the invocation.
        :param headers: (dict, optional) The HTTP headers for the invocation.
        :param auth: (Optional[tuple[str, str]], optional) The authentication creds for the invocation if required.
        :param kwargs: (dict) Additional keyword arguments.

        :return: The response from the API gateway invocation.
        """
        if not self.invoke_url:
            # try to resolve invoke_url before fail
            self.sync()
            if not self.invoke_url:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Invocation url is not set. Set up gateway's `invoke_url` attribute."
                )
        if not self.is_ready():
            raise mlrun.errors.MLRunPreconditionFailedError(
                f"API gateway is not ready. " f"Current state: {self.state}"
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

    def wait_for_readiness(self, max_wait_time=90):
        """
        Wait for the API gateway to become ready within the maximum wait time.

        Parameters:
            max_wait_time: int - Maximum time to wait in seconds (default is 90 seconds).

        Returns:
            bool: True if the entity becomes ready within the maximum wait time, False otherwise
        """

        def _ensure_ready():
            if not self.is_ready():
                raise AssertionError(
                    f"Waiting for gateway readiness is taking more than {max_wait_time} seconds"
                )

        return mlrun.utils.helpers.retry_until_successful(
            3, max_wait_time, logger, False, _ensure_ready
        )

    def is_ready(self):
        if self.state is not mlrun.common.schemas.api_gateway.APIGatewayState.ready:
            # try to sync the state
            self.sync()
        return self.state == mlrun.common.schemas.api_gateway.APIGatewayState.ready

    def sync(self):
        """
        Synchronize the API gateway from the server.
        """
        synced_gateway = mlrun.get_run_db().get_api_gateway(self.name, self.project)
        synced_gateway = self.from_scheme(synced_gateway)

        self.host = synced_gateway.host
        self.path = synced_gateway.path
        self.authentication = synced_gateway.authentication
        self.functions = synced_gateway.functions
        self.canary = synced_gateway.canary
        self.description = synced_gateway.description
        self.state = synced_gateway.state

    def with_basic_auth(self, username: str, password: str):
        """
        Set basic authentication for the API gateway.

        :param username: (str) The username for basic authentication.
        :param password: (str) The password for basic authentication.
        """
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
        """
        Set canary function for the API gateway

        :param functions: The list of functions associated with the API gateway
            Can be a list of function names (["my-func1", "my-func2"])
            or a list of nuclio functions of types
            :py:class:`~mlrun.runtimes.nuclio.function.RemoteRuntime` OR
            :py:class:`~mlrun.runtimes.nuclio.serving.ServingRuntime`
        :param canary: The canary percents for the API gateway of type list[int]; for instance: [20,80]

        """
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
        state = (
            api_gateway.status.state
            if api_gateway.status
            else mlrun.common.schemas.APIGatewayState.none
        )
        api_gateway = cls(
            project=project,
            description=api_gateway.spec.description,
            name=api_gateway.spec.name,
            host=api_gateway.spec.host,
            path=api_gateway.spec.path,
            authentication=APIGatewayAuthenticator.from_scheme(api_gateway.spec),
            functions=functions,
            canary=canary,
        )
        api_gateway.state = state
        return api_gateway

    def to_scheme(self) -> mlrun.common.schemas.APIGateway:
        upstreams = (
            [
                mlrun.common.schemas.APIGatewayUpstream(
                    nucliofunction={"name": self.functions[0]},
                    percentage=self.canary[0],
                ),
                mlrun.common.schemas.APIGatewayUpstream(
                    # do not set percent for the second function,
                    # so we can define which function to display as a primary one in UI
                    nucliofunction={"name": self.functions[1]},
                ),
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
        """
        Get the invoke URL.

        :return: (str) The invoke URL.
        """
        host = self.host
        if not self.host.startswith("http"):
            host = f"https://{self.host}"
        return urljoin(host, self.path)

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
