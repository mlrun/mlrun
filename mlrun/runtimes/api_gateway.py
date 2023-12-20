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
import urllib.parse
from typing import Optional, Union

import requests

import mlrun
import mlrun.common.schemas

BASIC_AUTH_NUCLIO_API_GATEWAY_AUTH_MODE = "basicAuth"
NO_AUTH_NUCLIO_API_GATEWAY_AUTH_MODE = "none"


class APIGateway:
    def __init__(
        self,
        project,
        name: str,
        host: str,
        path: str,
        description: str,
        functions: list[str],
        canary: Optional[list[int]],
    ):
        self.project = project
        self.name = name
        self.host = host
        self.functions = functions
        self.path = path
        self.description = description
        self.canary = canary
        self._nuclio_dashboard_url = mlrun.mlconf.nuclio_dashboard_url
        self._invoke_url = self._generate_invoke_url() if not host else host

    def invoke(self, auth: Optional[tuple[str, str]]):
        headers = {} if not auth else {"Authorization": self._generate_auth(*auth)}
        return requests.post(self._invoke_url, headers=headers)

    def to_scheme(
        self, auth: Optional[tuple[str, str]] = None
    ) -> mlrun.common.schemas.APIGateway:
        api_gateway = mlrun.common.schemas.APIGateway(
            function=self.functions,
            host=self.host,
            path=self.path,
            description=self.description,
            canary=self.canary,
        )
        if auth:
            username, password = auth
            api_gateway.username = username
            api_gateway.password = password
        return api_gateway

    @classmethod
    def from_values(
        cls,
        project,
        name: str,
        host: str,
        path: str,
        description: str,
        functions: list[str],
        canary: Optional[list[int]],
    ) -> "APIGateway":
        if not name:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "API Gateway name cannot be empty"
            )
        if canary:
            if len(functions) != len(canary):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Lengths of function and canary lists do not match"
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
        return cls(
            project=project,
            name=name,
            host=host,
            path=path,
            description=description,
            functions=functions,
            canary=canary,
        )

    @classmethod
    def from_dict(
        cls,
        dict_values,
    ):
        project = (
            dict_values.get("metadata", {})
            .get("labels", {})
            .get("nuclio.io/project-name")
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
            description=spec.get("description", ""),
            functions=functions,
            canary=canary,
        )

    def _generate_invoke_url(self):
        nuclio_hostname = urllib.parse.urlparse(self._nuclio_dashboard_url).netloc

        # Remove the 'nuclio' prefix from the hostname
        # For example, from `nuclio.default-tenant.app.dev62.lab.iguazeng.com`,
        # it becomes `default-tenant.app.dev62.lab.iguazeng.com`
        common_hostname = nuclio_hostname[nuclio_hostname.find(".") + 1 :]

        # Generate a unique invoke URL which contains the API gateway name and project name
        return urllib.parse.urljoin(
            f"{self.name}-{self.project}.{common_hostname}", self.path
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
