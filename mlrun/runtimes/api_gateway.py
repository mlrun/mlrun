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
from typing import Optional, Union

import requests

import mlrun
import mlrun.common.schemas

from .function import RemoteRuntime
from .serving import ServingRuntime

BASIC_AUTH_NUCLIO_API_GATEWAY_AUTH_MODE = "basicAuth"
NO_AUTH_NUCLIO_API_GATEWAY_AUTH_MODE = "none"
PROJECT_NAME_LABEL = "nuclio.io/project-name"


class APIGateway:
    def __init__(
        self,
        project,
        name: str,
        path: str,
        description: str,
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
        authentication_mode: Optional[str] = NO_AUTH_NUCLIO_API_GATEWAY_AUTH_MODE,
        host: Optional[str] = None,
        canary: Optional[list[int]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.functions = self._validate_functions(functions=functions, project=project)
        self._validate(
            name=name,
            canary=canary,
            username=username,
            password=password,
        )
        self.project = project
        self.name = name
        self.host = host

        self.path = path
        self.description = description
        self.authentication_mode = (
            authentication_mode
            if authentication_mode
            else self._enrich_authentication_mode(username=username, password=password)
        )
        self.canary = canary
        self._invoke_url = None
        self.__username = username
        self.__password = password
        if host:
            self._invoke_url = self.generate_invoke_url()

    def invoke(self, headers: dict = {}, auth: Optional[tuple[str, str]] = None):
        if not self._invoke_url:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Invocation url is not set. Use set_invoke_url method to set it."
            )
        if (
            self.authentication_mode == BASIC_AUTH_NUCLIO_API_GATEWAY_AUTH_MODE
            and not auth
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "API Gateway invocation requires authentication. Please pass credentials"
            )
        if auth:
            headers["Authorization"] = self._generate_auth(*auth)
        return requests.post(self._invoke_url, headers=headers)

    @classmethod
    def from_dict(
        cls,
        dict_values,
    ):
        project = (
            dict_values.get("metadata", {}).get("labels", {}).get(PROJECT_NAME_LABEL)
        )
        spec = dict_values.get("spec", {})
        upstreams = spec.get("upstreams", [])
        functions, canary = cls._process_canary(upstreams)
        if not functions:
            return None
        return cls(
            project=project,
            name=spec.get("name"),
            host=spec.get("host"),
            path=spec.get("path", ""),
            authentication_mode=spec.get("authenticationMode"),
            description=spec.get("description", ""),
            functions=functions,
            canary=canary,
        )

    def to_scheme(self) -> mlrun.common.schemas.APIGateway:
        return mlrun.common.schemas.APIGateway(
            functions=self.functions,
            path=self.path,
            description=self.description,
            username=self.__username,
            password=self.__password,
            canary=self.canary,
        )

    def generate_invoke_url(
        self,
    ):
        return f"{self.host}{self.path}"

    def set_invoke_url(self, invoke_url: str):
        self._invoke_url = invoke_url

    def _validate(
        self,
        name: str,
        canary: Optional[list[int]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        if not name:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "API Gateway name cannot be empty"
            )

        # validating canary
        if canary:
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

        # validating auth
        if username and not password:
            raise mlrun.errors.MLRunInvalidArgumentError("Password is not specified")

        if password and not username:
            raise mlrun.errors.MLRunInvalidArgumentError("Username is not specified")

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
                f"but the number of functions is passed {len(functions)}"
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
            function_names.append(func.uri)
        return function_names

    @staticmethod
    def _enrich_authentication_mode(username, password):
        return (
            NO_AUTH_NUCLIO_API_GATEWAY_AUTH_MODE
            if username is not None and password is not None
            else BASIC_AUTH_NUCLIO_API_GATEWAY_AUTH_MODE
        )

    @staticmethod
    def _generate_auth(username: str, password: str):
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        return f"Basic {token}"

    @staticmethod
    def _process_canary(
        upstreams: list[dict],
    ) -> tuple[Union[list[str], None], Union[list[int], None]]:
        if len(upstreams) == 1:
            return [upstreams[0].get("nucliofunction", {}).get("name")], None
        elif len(upstreams) == 2:
            canary = [0, 0]
            functions = [
                upstreams[0].get("nucliofunction", {}).get("name"),
                upstreams[1].get("nucliofunction", {}).get("name"),
            ]
            percentage_1 = upstreams[0].get("percentage")
            percentage_2 = upstreams[1].get("percentage")

            if not percentage_1 and percentage_2:
                percentage_1 = 100 - percentage_2
            if not percentage_2 and percentage_1:
                percentage_2 = 100 - percentage_1
            if percentage_1 and percentage_2:
                canary = [percentage_1, percentage_2]
            return functions, canary
        else:
            # The upstream list length should be either 1 or 2; otherwise, we cannot parse the upstream value correctly
            return None, None
