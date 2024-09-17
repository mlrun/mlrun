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
from nuclio.auth import AuthInfo as NuclioAuthInfo
from nuclio.auth import AuthKinds as NuclioAuthKinds

import mlrun
import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas as schemas
import mlrun.common.types
from mlrun.model import ModelObj
from mlrun.platforms.iguazio import min_iguazio_versions
from mlrun.utils import logger

from .function import min_nuclio_versions


class Authenticator(typing.Protocol):
    @property
    def authentication_mode(self) -> str:
        return schemas.APIGatewayAuthenticationMode.none.value

    @classmethod
    def from_scheme(cls, api_gateway_spec: schemas.APIGatewaySpec):
        if (
            api_gateway_spec.authenticationMode
            == schemas.APIGatewayAuthenticationMode.basic.value
        ):
            if api_gateway_spec.authentication:
                return BasicAuth(
                    username=api_gateway_spec.authentication.get("username", ""),
                    password=api_gateway_spec.authentication.get("password", ""),
                )
            else:
                return BasicAuth()
        elif (
            api_gateway_spec.authenticationMode
            == schemas.APIGatewayAuthenticationMode.access_key.value
        ):
            return AccessKeyAuth()
        else:
            return NoneAuth()

    def to_scheme(
        self,
    ) -> Optional[dict[str, Optional[schemas.APIGatewayBasicAuth]]]:
        return None


class APIGatewayAuthenticator(Authenticator, ModelObj):
    _dict_fields = ["authentication_mode"]


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
        return schemas.APIGatewayAuthenticationMode.basic.value

    def to_scheme(
        self,
    ) -> Optional[dict[str, Optional[schemas.APIGatewayBasicAuth]]]:
        return {
            "basicAuth": schemas.APIGatewayBasicAuth(
                username=self._username, password=self._password
            )
        }


class AccessKeyAuth(APIGatewayAuthenticator):
    """
    An API gateway authenticator with access key authentication.
    """

    @property
    def authentication_mode(self) -> str:
        return schemas.APIGatewayAuthenticationMode.access_key.value


class APIGatewayMetadata(ModelObj):
    _dict_fields = ["name", "namespace", "labels", "annotations", "creation_timestamp"]

    def __init__(
        self,
        name: str,
        namespace: str = None,
        labels: dict = None,
        annotations: dict = None,
        creation_timestamp: str = None,
    ):
        """
        :param name: The name of the API gateway
        :param namespace: The namespace of the API gateway
        :param labels: The labels of the API gateway
        :param annotations: The annotations of the API gateway
        :param creation_timestamp: The creation timestamp of the API gateway
        """
        self.name = name
        self.namespace = namespace
        self.labels = labels or {}
        self.annotations = annotations or {}
        self.creation_timestamp = creation_timestamp

        if not self.name:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "API Gateway name cannot be empty"
            )


class APIGatewaySpec(ModelObj):
    _dict_fields = [
        "functions",
        "project",
        "name",
        "description",
        "host",
        "path",
        "authentication",
        "canary",
    ]

    def __init__(
        self,
        functions: Union[
            list[
                Union[
                    str,
                    "mlrun.runtimes.nuclio.function.RemoteRuntime",
                    "mlrun.runtimes.nuclio.serving.ServingRuntime",
                    "mlrun.runtimes.nuclio.application.ApplicationRuntime",
                ]
            ],
            "mlrun.runtimes.nuclio.function.RemoteRuntime",
            "mlrun.runtimes.nuclio.serving.ServingRuntime",
            "mlrun.runtimes.nuclio.application.ApplicationRuntime",
        ],
        project: str = None,
        description: str = "",
        host: str = None,
        path: str = "/",
        authentication: Optional[APIGatewayAuthenticator] = NoneAuth(),
        canary: Optional[list[int]] = None,
        ports: Optional[list[int]] = None,
    ):
        """
        :param functions: The list of functions associated with the API gateway
            Can be a list of function names (["my-func1", "my-func2"])
            or a list or a single entity of
            :py:class:`~mlrun.runtimes.nuclio.function.RemoteRuntime` OR
            :py:class:`~mlrun.runtimes.nuclio.serving.ServingRuntime` OR
            :py:class:`~mlrun.runtimes.nuclio.application.ApplicationRuntime`
        :param project: The project name
        :param description: Optional description of the API gateway
        :param path: Optional path of the API gateway, default value is "/"
        :param authentication: The authentication for the API gateway of type
                :py:class:`~mlrun.runtimes.nuclio.api_gateway.BasicAuth`
        :param host:  The host of the API gateway (optional). If not set, it will be automatically generated
        :param canary: The canary percents for the API gateway of type list[int]; for instance: [20,80] (optional)
        :param ports: The ports of the API gateway, as a list of integers that correspond to the functions in the
            functions list. for instance: [8050] or [8050, 8081] (optional)
        """
        self.description = description
        self.host = host
        self.path = path
        self.authentication = authentication
        self.functions = functions
        self.canary = canary
        self.project = project
        self.ports = ports

        self.validate(project=project, functions=functions, canary=canary, ports=ports)

    def validate(
        self,
        project: str,
        functions: Union[
            list[
                Union[
                    str,
                    "mlrun.runtimes.nuclio.function.RemoteRuntime",
                    "mlrun.runtimes.nuclio.serving.ServingRuntime",
                    "mlrun.runtimes.nuclio.application.ApplicationRuntime",
                ]
            ],
            "mlrun.runtimes.nuclio.function.RemoteRuntime",
            "mlrun.runtimes.nuclio.serving.ServingRuntime",
            "mlrun.runtimes.nuclio.application.ApplicationRuntime",
        ],
        canary: Optional[list[int]] = None,
        ports: Optional[list[int]] = None,
    ):
        self.functions = self._validate_functions(project=project, functions=functions)

        # validating canary
        if canary:
            self.canary = self._validate_canary(canary)

        # validating ports
        if ports:
            self.ports = self._validate_ports(ports)

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

    def _validate_ports(self, ports):
        if len(self.functions) != len(ports):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Function and port lists lengths do not match"
            )

        return ports

    @staticmethod
    def _validate_functions(
        project: str,
        functions: Union[
            list[
                Union[
                    str,
                    "mlrun.runtimes.nuclio.function.RemoteRuntime",
                    "mlrun.runtimes.nuclio.serving.ServingRuntime",
                    "mlrun.runtimes.nuclio.application.ApplicationRuntime",
                ]
            ],
            "mlrun.runtimes.nuclio.function.RemoteRuntime",
            "mlrun.runtimes.nuclio.serving.ServingRuntime",
            "mlrun.runtimes.nuclio.application.ApplicationRuntime",
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
                # check whether the function was passed as a URI or just a name
                parsed_project, function_name, _, _ = (
                    mlrun.common.helpers.parse_versioned_object_uri(func)
                )

                if parsed_project and function_name:
                    # check that parsed project and passed project are the same
                    if parsed_project != project:
                        raise mlrun.errors.MLRunInvalidArgumentError(
                            "Function doesn't belong to passed project"
                        )
                    function_uri = func
                else:
                    function_uri = mlrun.utils.generate_object_uri(project, func)
                function_names.append(function_uri)
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
            function_uri = mlrun.utils.generate_object_uri(
                project,
                function_name,
                func.metadata.tag,
                func.metadata.hash,
            )
            function_names.append(function_uri)
        return function_names


class APIGatewayStatus(ModelObj):
    def __init__(self, state: Optional[schemas.APIGatewayState] = None):
        self.state = state or schemas.APIGatewayState.none


class APIGateway(ModelObj):
    _dict_fields = [
        "metadata",
        "spec",
        "state",
    ]

    @min_nuclio_versions("1.13.1")
    def __init__(
        self,
        metadata: APIGatewayMetadata,
        spec: APIGatewaySpec,
        status: Optional[APIGatewayStatus] = None,
    ):
        """
        Initialize the APIGateway instance.

        :param metadata: (APIGatewayMetadata) The metadata of the API gateway.
        :param spec: (APIGatewaySpec) The spec of the API gateway.
        :param status: (APIGatewayStatus) The status of the API gateway.
        """
        self.metadata = metadata
        self.spec = spec
        self.status = status

    @property
    def metadata(self) -> APIGatewayMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, "metadata", APIGatewayMetadata)

    @property
    def spec(self) -> APIGatewaySpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", APIGatewaySpec)

    @property
    def status(self) -> APIGatewayStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", APIGatewayStatus)

    def invoke(
        self,
        method="POST",
        headers: dict = None,
        credentials: Optional[tuple[str, str]] = None,
        path: Optional[str] = None,
        body: Optional[Union[str, bytes, dict]] = None,
        **kwargs,
    ):
        """
        Invoke the API gateway.

        :param method: (str, optional) The HTTP method for the invocation.
        :param headers: (dict, optional) The HTTP headers for the invocation.
        :param credentials: (Optional[tuple[str, str]], optional) The (username,password) for the invocation if required
            can also be set by the environment variable (_, V3IO_ACCESS_KEY) for access key authentication.
        :param path: (str, optional) The sub-path for the invocation.
        :param body: (Optional[Union[str, bytes, dict]]) The body of the invocation.
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
                f"API gateway is not ready. " f"Current state: {self.status.state}"
            )

        auth = None

        if (
            self.spec.authentication.authentication_mode
            == schemas.APIGatewayAuthenticationMode.basic.value
        ):
            if not credentials:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "API Gateway invocation requires authentication. Please pass credentials"
                )
            auth = NuclioAuthInfo(
                username=credentials[0], password=credentials[1]
            ).to_requests_auth()

        if (
            self.spec.authentication.authentication_mode
            == schemas.APIGatewayAuthenticationMode.access_key.value
        ):
            # inject access key from env
            if credentials:
                auth = NuclioAuthInfo(
                    username=credentials[0],
                    password=credentials[1],
                    mode=NuclioAuthKinds.iguazio,
                ).to_requests_auth()
            else:
                auth = NuclioAuthInfo().from_envvar().to_requests_auth()
            if not auth:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "API Gateway invocation requires authentication. Please set V3IO_ACCESS_KEY env var"
                )
        url = urljoin(self.invoke_url, path or "")

        # Determine the correct keyword argument for the body
        if isinstance(body, dict):
            kwargs["json"] = body
        elif isinstance(body, (str, bytes)):
            kwargs["data"] = body

        return requests.request(
            method=method,
            url=url,
            headers=headers or {},
            auth=auth,
            **kwargs,
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
        if self.status.state is not schemas.api_gateway.APIGatewayState.ready:
            # try to sync the state
            self.sync()
        return self.status.state == schemas.api_gateway.APIGatewayState.ready

    def sync(self):
        """
        Synchronize the API gateway from the server.
        """
        synced_gateway = mlrun.get_run_db().get_api_gateway(
            self.metadata.name, self.spec.project
        )
        synced_gateway = self.from_scheme(synced_gateway)

        self.spec.host = synced_gateway.spec.host
        self.spec.path = synced_gateway.spec.path
        self.spec.authentication = synced_gateway.spec.authentication
        self.spec.functions = synced_gateway.spec.functions
        self.spec.canary = synced_gateway.spec.canary
        self.spec.description = synced_gateway.spec.description
        self.status.state = synced_gateway.status.state

    def with_basic_auth(self, username: str, password: str):
        """
        Set basic authentication for the API gateway.

        :param username: (str) The username for basic authentication.
        :param password: (str) The password for basic authentication.
        """
        self.spec.authentication = BasicAuth(username=username, password=password)

    @min_iguazio_versions("3.5.5")
    def with_access_key_auth(self):
        """
        Set access key authentication for the API gateway.
        """
        self.spec.authentication = AccessKeyAuth()

    def with_canary(
        self,
        functions: Union[
            list[
                Union[
                    str,
                    "mlrun.runtimes.nuclio.function.RemoteRuntime",
                    "mlrun.runtimes.nuclio.serving.ServingRuntime",
                    "mlrun.runtimes.nuclio.application.ApplicationRuntime",
                ]
            ],
            "mlrun.runtimes.nuclio.function.RemoteRuntime",
            "mlrun.runtimes.nuclio.serving.ServingRuntime",
            "mlrun.runtimes.nuclio.application.ApplicationRuntime",
        ],
        canary: list[int],
    ):
        """
        Set canary function for the API gateway

        :param functions: The list of functions associated with the API gateway
            Can be a list of function names (["my-func1", "my-func2"])
            or a list of nuclio functions of types
            :py:class:`~mlrun.runtimes.nuclio.function.RemoteRuntime` OR
            :py:class:`~mlrun.runtimes.nuclio.serving.ServingRuntime` OR
            :py:class:`~mlrun.runtimes.nuclio.application.ApplicationRuntime`
        :param canary: The canary percents for the API gateway of type list[int]; for instance: [20,80]

        """
        if len(functions) != 2:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Gateway with canary can be created only with two functions, "
                f"the number of functions passed is {len(functions)}"
            )
        self.spec.validate(
            project=self.spec.project, functions=functions, canary=canary
        )

    def with_ports(self, ports: list[int]):
        """
        Set ports for the API gateway

        :param ports: The ports of the API gateway, as a list of integers that correspond to the functions in the
            functions list. for instance: [8050] or [8050, 8081]
        """
        self.spec.validate(
            project=self.spec.project, functions=self.spec.functions, ports=ports
        )

    def with_force_ssl_redirect(self):
        """
        Set SSL redirect annotation for the API gateway.
        """
        self.metadata.annotations["nginx.ingress.kubernetes.io/force-ssl-redirect"] = (
            "true"
        )

    def with_gateway_timeout(self, gateway_timeout: int):
        """
        Set gateway proxy connect/read/send timeout annotations
        :param gateway_timeout: The timeout in seconds
        """
        mlrun.runtimes.utils.enrich_gateway_timeout_annotations(
            self.metadata.annotations, gateway_timeout
        )

    def with_annotations(self, annotations: dict):
        """set a key/value annotations in the metadata of the api gateway"""
        for key, value in annotations.items():
            self.metadata.annotations[key] = str(value)
        return self

    @classmethod
    def from_scheme(cls, api_gateway: schemas.APIGateway):
        project = api_gateway.metadata.labels.get(
            mlrun_constants.MLRunInternalLabels.nuclio_project_name
        )
        functions, canary = cls._resolve_canary(api_gateway.spec.upstreams)
        state = (
            api_gateway.status.state
            if api_gateway.status
            else schemas.APIGatewayState.none
        )
        new_api_gateway = cls(
            metadata=APIGatewayMetadata(
                name=api_gateway.spec.name,
                annotations=api_gateway.metadata.annotations,
                labels=api_gateway.metadata.labels,
            ),
            spec=APIGatewaySpec(
                project=project,
                description=api_gateway.spec.description,
                host=api_gateway.spec.host,
                path=api_gateway.spec.path,
                authentication=APIGatewayAuthenticator.from_scheme(api_gateway.spec),
                functions=functions,
                canary=canary,
            ),
            status=APIGatewayStatus(state=state),
        )
        return new_api_gateway

    def to_scheme(self) -> schemas.APIGateway:
        upstreams = (
            [
                schemas.APIGatewayUpstream(
                    nucliofunction={"name": self.spec.functions[0]},
                    percentage=self.spec.canary[0],
                ),
                schemas.APIGatewayUpstream(
                    # do not set percent for the second function,
                    # so we can define which function to display as a primary one in UI
                    nucliofunction={"name": self.spec.functions[1]},
                ),
            ]
            if self.spec.canary
            else [
                schemas.APIGatewayUpstream(
                    nucliofunction={"name": function_name},
                )
                for function_name in self.spec.functions
            ]
        )
        if self.spec.ports:
            for i, port in enumerate(self.spec.ports):
                upstreams[i].port = port

        api_gateway = schemas.APIGateway(
            metadata=schemas.APIGatewayMetadata(
                name=self.metadata.name,
                labels=self.metadata.labels,
                annotations=self.metadata.annotations,
            ),
            spec=schemas.APIGatewaySpec(
                name=self.metadata.name,
                description=self.spec.description,
                host=self.spec.host,
                path=self.spec.path,
                authenticationMode=schemas.APIGatewayAuthenticationMode.from_str(
                    self.spec.authentication.authentication_mode
                ),
                upstreams=upstreams,
            ),
            status=schemas.APIGatewayStatus(state=self.status.state),
        )
        api_gateway.spec.authentication = self.spec.authentication.to_scheme()
        return api_gateway

    @property
    def invoke_url(
        self,
    ):
        """
        Get the invoke URL.

        :return: (str) The invoke URL.
        """
        host = self.spec.host
        if not self.spec.host.startswith("http"):
            host = f"https://{self.spec.host}"
        return urljoin(host, self.spec.path).rstrip("/")

    @staticmethod
    def _generate_basic_auth(username: str, password: str):
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        return f"Basic {token}"

    @staticmethod
    def _resolve_canary(
        upstreams: list[schemas.APIGatewayUpstream],
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

    @property
    def name(self):
        return self.metadata.name

    @name.setter
    def name(self, value):
        self.metadata.name = value

    @property
    def project(self):
        return self.spec.project

    @project.setter
    def project(self, value):
        self.spec.project = value

    @property
    def description(self):
        return self.spec.description

    @description.setter
    def description(self, value):
        self.spec.description = value

    @property
    def host(self):
        return self.spec.host

    @host.setter
    def host(self, value):
        self.spec.host = value

    @property
    def path(self):
        return self.spec.path

    @path.setter
    def path(self, value):
        self.spec.path = value

    @property
    def authentication(self):
        return self.spec.authentication

    @authentication.setter
    def authentication(self, value):
        self.spec.authentication = value
