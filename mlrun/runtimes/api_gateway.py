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
from typing import Union
import urllib.parse
import requests

import mlrun

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
        username: Union[None, str],
        password: Union[None, str],
        canary: Union[dict[str, int], None],
    ):
        self.project = project
        self.name = name
        self.host = host
        self.functions = functions
        self.path = path
        self.description = description
        self.canary = canary
        self._nuclio_dashboard_url = mlrun.mlconf.nuclio_dashboard_url
        self._auth = None
        self._invoke_url = self._generate_invoke_url() if not host else host
        self._generate_auth(username, password)

    def invoke(self):
        headers = {} if not self._auth else {"Authorization": self._auth}
        return requests.post(self._invoke_url, headers=headers)

    def _generate_auth(self, username: Union[None, str], password: Union[None, str]):
        if username and password:
            token = base64.b64encode(f"{username}:{password}".encode()).decode()
            self._auth = f"Basic {token}"

    def requires_auth(self):
        return self._auth is not None

    def _generate_invoke_url(self):
        nuclio_hostname = urllib.parse.urlparse(self._nuclio_dashboard_url).netloc
        # cut nuclio prefix
        common_hostname = nuclio_hostname[nuclio_hostname.find(".") + 1 :]
        return urllib.parse.urljoin(
            f"{self.name}-{self.project}.{common_hostname}", self.path
        )

    @classmethod
    def from_values(
        cls,
        project,
        name: str,
        host: str,
        path: str,
        description: str,
        functions: list[str],
        username: Union[None, str],
        password: Union[None, str],
        canary: Union[list[int], None],
    ) -> "APIGateway":
        if not name:
            raise ValueError("API Gateway name cannot be empty")

        if canary:
            if len(functions) != len(canary):
                raise ValueError("Lengths of function and canary lists do not match")
            if sum(canary) != 100:
                raise ValueError(
                    "The sum of canary function percents should be equal to 100"
                )
        return cls(
            project=project,
            name=name,
            host=host,
            path=path,
            description=description,
            functions=functions,
            username=username,
            password=password,
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
        authentication_mode = spec.get(
            "authenticationMode", NO_AUTH_NUCLIO_API_GATEWAY_AUTH_MODE
        )
        authentication = spec.get("authentication", {}).get(authentication_mode, {})
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
            username=authentication.get("username"),
            password=authentication.get("password"),
            canary=canary,
        )

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
            percentage_2 = upstreams[0].get("percentage")

            if not percentage_1 and percentage_2:
                percentage_1 = 100 - percentage_2
            if not percentage_2 and percentage_1:
                percentage_2 = 100 - percentage_1
            if percentage_1 and percentage_2:
                canary = [percentage_1, percentage_2]
            return functions, canary
        else:
            return None, None
