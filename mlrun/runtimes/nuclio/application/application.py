# Copyright 2024 Iguazio
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
import pathlib
import typing

import nuclio

import mlrun.common.schemas as schemas
import mlrun.errors
from mlrun.common.runtimes.constants import NuclioIngressAddTemplatedIngressModes
from mlrun.runtimes import RemoteRuntime
from mlrun.runtimes.nuclio import min_nuclio_versions
from mlrun.runtimes.nuclio.api_gateway import (
    APIGateway,
    APIGatewayMetadata,
    APIGatewaySpec,
)
from mlrun.runtimes.nuclio.function import NuclioSpec, NuclioStatus
from mlrun.utils import logger


class ApplicationSpec(NuclioSpec):
    _dict_fields = NuclioSpec._dict_fields + [
        "internal_application_port",
    ]

    def __init__(
        self,
        command=None,
        args=None,
        image=None,
        mode=None,
        entry_points=None,
        description=None,
        replicas=None,
        min_replicas=None,
        max_replicas=None,
        volumes=None,
        volume_mounts=None,
        env=None,
        resources=None,
        config=None,
        base_spec=None,
        no_cache=None,
        source=None,
        image_pull_policy=None,
        function_kind=None,
        build=None,
        service_account=None,
        readiness_timeout=None,
        readiness_timeout_before_failure=None,
        default_handler=None,
        node_name=None,
        node_selector=None,
        affinity=None,
        disable_auto_mount=False,
        priority_class_name=None,
        pythonpath=None,
        workdir=None,
        image_pull_secret=None,
        tolerations=None,
        preemption_mode=None,
        security_context=None,
        service_type=None,
        add_templated_ingress_host_mode=None,
        clone_target_dir=None,
        state_thresholds=None,
        disable_default_http_trigger=None,
        internal_application_port=None,
    ):
        super().__init__(
            command=command,
            args=args,
            image=image,
            mode=mode,
            entry_points=entry_points,
            description=description,
            replicas=replicas,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            volumes=volumes,
            volume_mounts=volume_mounts,
            env=env,
            resources=resources,
            config=config,
            base_spec=base_spec,
            no_cache=no_cache,
            source=source,
            image_pull_policy=image_pull_policy,
            function_kind=function_kind,
            build=build,
            service_account=service_account,
            readiness_timeout=readiness_timeout,
            readiness_timeout_before_failure=readiness_timeout_before_failure,
            default_handler=default_handler,
            node_name=node_name,
            node_selector=node_selector,
            affinity=affinity,
            disable_auto_mount=disable_auto_mount,
            priority_class_name=priority_class_name,
            pythonpath=pythonpath,
            workdir=workdir,
            image_pull_secret=image_pull_secret,
            tolerations=tolerations,
            preemption_mode=preemption_mode,
            security_context=security_context,
            service_type=service_type,
            add_templated_ingress_host_mode=add_templated_ingress_host_mode,
            clone_target_dir=clone_target_dir,
            state_thresholds=state_thresholds,
            disable_default_http_trigger=disable_default_http_trigger,
        )
        self.internal_application_port = (
            internal_application_port
            or mlrun.mlconf.function.application.default_sidecar_internal_port
        )

    @property
    def internal_application_port(self):
        return self._internal_application_port

    @internal_application_port.setter
    def internal_application_port(self, port):
        port = int(port)
        if port < 0 or port > 65535:
            raise ValueError("Port must be in the range 0-65535")
        self._internal_application_port = port


class ApplicationStatus(NuclioStatus):
    def __init__(
        self,
        state=None,
        nuclio_name=None,
        address=None,
        internal_invocation_urls=None,
        external_invocation_urls=None,
        build_pod=None,
        container_image=None,
        application_image=None,
        sidecar_name=None,
        api_gateway_name=None,
        api_gateway=None,
        url=None,
    ):
        super().__init__(
            state=state,
            nuclio_name=nuclio_name,
            address=address,
            internal_invocation_urls=internal_invocation_urls,
            external_invocation_urls=external_invocation_urls,
            build_pod=build_pod,
            container_image=container_image,
        )
        self.application_image = application_image or None
        self.sidecar_name = sidecar_name or None
        self.api_gateway_name = api_gateway_name or None
        self.api_gateway = api_gateway or None
        self.url = url or None


class ApplicationRuntime(RemoteRuntime):
    kind = "application"

    @min_nuclio_versions("1.13.1")
    def __init__(self, spec=None, metadata=None):
        super().__init__(spec=spec, metadata=metadata)

    @property
    def spec(self) -> ApplicationSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", ApplicationSpec)

    @property
    def status(self) -> ApplicationStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", ApplicationStatus)

    @property
    def api_gateway(self):
        return self.status.api_gateway

    @api_gateway.setter
    def api_gateway(self, api_gateway: APIGateway):
        self.status.api_gateway = api_gateway

    @property
    def url(self):
        if not self.status.api_gateway:
            self._sync_api_gateway()
        return self.status.api_gateway.invoke_url

    @url.setter
    def url(self, url):
        self.status.url = url

    def set_internal_application_port(self, port: int):
        self.spec.internal_application_port = port

    def pre_deploy_validation(self):
        super().pre_deploy_validation()
        if not self.spec.config.get("spec.sidecars"):
            raise mlrun.errors.MLRunBadRequestError(
                "Application spec must include a sidecar configuration"
            )

        sidecars = self.spec.config["spec.sidecars"]
        for sidecar in sidecars:
            if not sidecar.get("image"):
                raise mlrun.errors.MLRunBadRequestError(
                    "Application sidecar spec must include an image"
                )

            if not sidecar.get("ports"):
                raise mlrun.errors.MLRunBadRequestError(
                    "Application sidecar spec must include at least one port"
                )

            ports = sidecar["ports"]
            for port in ports:
                if not port.get("containerPort"):
                    raise mlrun.errors.MLRunBadRequestError(
                        "Application sidecar port spec must include a containerPort"
                    )

                if not port.get("name"):
                    raise mlrun.errors.MLRunBadRequestError(
                        "Application sidecar port spec must include a name"
                    )

            if not sidecar.get("command") and sidecar.get("args"):
                raise mlrun.errors.MLRunBadRequestError(
                    "Application sidecar spec must include a command if args are provided"
                )

    def deploy(
        self,
        project="",
        tag="",
        verbose=False,
        auth_info: schemas.AuthInfo = None,
        builder_env: dict = None,
        force_build: bool = False,
        with_mlrun=None,
        skip_deployed=False,
        is_kfp=False,
        mlrun_version_specifier=None,
        show_on_failure: bool = False,
        direct_port_access: bool = False,
        authentication_mode: schemas.APIGatewayAuthenticationMode = None,
        authentication_creds: tuple[str] = None,
    ):
        """
        Deploy function, builds the application image if required (self.requires_build()) or force_build is True,
        Once the image is built, the function is deployed.
        :param project:                 Project name
        :param tag:                     Function tag
        :param verbose:                 Set True for verbose logging
        :param auth_info:               Service AuthInfo (deprecated and ignored)
        :param builder_env:             Env vars dict for source archive config/credentials
                                        e.g. builder_env={"GIT_TOKEN": token}
        :param force_build:             Set True for force building the application image
        :param with_mlrun:              Add the current mlrun package to the container build
        :param skip_deployed:           Skip the build if we already have an image for the function
        :param is_kfp:                  Deploy as part of a kfp pipeline
        :param mlrun_version_specifier: Which mlrun package version to include (if not current)
        :param show_on_failure:         Show logs only in case of build failure
        :param direct_port_access:      Set True to allow direct port access to the application sidecar
        :param authentication_mode:     API Gateway authentication mode
        :param authentication_creds:    API Gateway authentication credentials as a tuple (username, password)
        :return: True if the function is ready (deployed)
        """
        if self.requires_build() or force_build:
            self._fill_credentials()
            self._build_application_image(
                builder_env=builder_env,
                force_build=force_build,
                watch=True,
                with_mlrun=with_mlrun,
                skip_deployed=skip_deployed,
                is_kfp=is_kfp,
                mlrun_version_specifier=mlrun_version_specifier,
                show_on_failure=show_on_failure,
            )

        self._ensure_reverse_proxy_configurations()
        self._configure_application_sidecar()

        # we only allow accessing the application via the API Gateway
        name_tag = tag or self.metadata.tag
        self.status.api_gateway_name = (
            f"{self.metadata.name}-{name_tag}" if name_tag else self.metadata.name
        )
        self.spec.add_templated_ingress_host_mode = (
            NuclioIngressAddTemplatedIngressModes.never
        )

        super().deploy(
            project,
            tag,
            verbose,
            auth_info,
            builder_env,
        )

        ports = self.spec.internal_application_port if direct_port_access else []
        self.create_api_gateway(
            name=self.status.api_gateway_name,
            ports=ports,
            authentication_mode=authentication_mode,
            authentication_creds=authentication_creds,
        )

    def with_source_archive(
        self, source, workdir=None, pull_at_runtime=True, target_dir=None
    ):
        """load the code from git/tar/zip archive at runtime or build

        :param source:          valid absolute path or URL to git, zip, or tar file, e.g.
                                git://github.com/mlrun/something.git
                                http://some/url/file.zip
                                note path source must exist on the image or exist locally when run is local
                                (it is recommended to use 'workdir' when source is a filepath instead)
        :param workdir:         working dir relative to the archive root (e.g. './subdir') or absolute to the image root
        :param pull_at_runtime: load the archive into the container at job runtime vs on build/deploy
        :param target_dir:      target dir on runtime pod or repo clone / archive extraction
        """
        self._configure_mlrun_build_with_source(
            source=source,
            workdir=workdir,
            pull_at_runtime=pull_at_runtime,
            target_dir=target_dir,
        )

    @classmethod
    def get_filename_and_handler(cls) -> (str, str):
        reverse_proxy_file_path = pathlib.Path(__file__).parent / "reverse_proxy.go"
        return str(reverse_proxy_file_path), "Handler"

    def create_api_gateway(
        self,
        name: str = None,
        path: str = None,
        ports: list[int] = None,
        authentication_mode: schemas.APIGatewayAuthenticationMode = None,
        authentication_creds: tuple[str] = None,
    ):
        api_gateway = APIGateway(
            APIGatewayMetadata(
                name=name,
                namespace=self.metadata.namespace,
                labels=self.metadata.labels,
                annotations=self.metadata.annotations,
            ),
            APIGatewaySpec(
                functions=[self],
                project=self.metadata.project,
                path=path,
                ports=mlrun.utils.helpers.as_list(ports) if ports else None,
            ),
        )

        authentication_mode = (
            authentication_mode
            or mlrun.mlconf.function.application.default_authentication_mode
        )
        if authentication_mode == schemas.APIGatewayAuthenticationMode.access_key:
            api_gateway.with_access_key_auth()
        elif authentication_mode == schemas.APIGatewayAuthenticationMode.basic:
            api_gateway.with_basic_auth(*authentication_creds)

        db = self._get_db()
        api_gateway_scheme = db.store_api_gateway(
            api_gateway=api_gateway.to_scheme(), project=self.metadata.project
        )
        if not self.status.api_gateway_name:
            self.status.api_gateway_name = api_gateway_scheme.metadata.name
        self.status.api_gateway = APIGateway.from_scheme(api_gateway_scheme)
        self.status.api_gateway.wait_for_readiness()
        self.url = self.status.api_gateway.invoke_url

    def invoke(
        self,
        path: str,
        body: typing.Union[str, bytes, dict] = None,
        method: str = None,
        headers: dict = None,
        dashboard: str = "",
        force_external_address: bool = False,
        auth_info: schemas.AuthInfo = None,
        mock: bool = None,
        **http_client_kwargs,
    ):
        self._sync_api_gateway()
        # If the API Gateway is not ready or not set, try to invoke the function directly (without the API Gateway)
        if not self.status.api_gateway:
            super().invoke(
                path,
                body,
                method,
                headers,
                dashboard,
                force_external_address,
                auth_info,
                mock,
                **http_client_kwargs,
            )

        credentials = (auth_info.username, auth_info.password) if auth_info else None

        if not method:
            method = "POST" if body else "GET"
        return self.status.api_gateway.invoke(
            method=method,
            headers=headers,
            credentials=credentials,
            path=path,
            **http_client_kwargs,
        )

    def _build_application_image(
        self,
        builder_env: dict = None,
        force_build: bool = False,
        watch=True,
        with_mlrun=None,
        skip_deployed=False,
        is_kfp=False,
        mlrun_version_specifier=None,
        show_on_failure: bool = False,
    ):
        if not self.spec.command:
            logger.warning(
                "Building the application image without a command. "
                "Use spec.command and spec.args to specify the application entrypoint",
                command=self.spec.command,
                args=self.spec.args,
            )

        with_mlrun = self._resolve_build_with_mlrun(with_mlrun)
        return self._build_image(
            builder_env=builder_env,
            force_build=force_build,
            mlrun_version_specifier=mlrun_version_specifier,
            show_on_failure=show_on_failure,
            skip_deployed=skip_deployed,
            watch=watch,
            is_kfp=is_kfp,
            with_mlrun=with_mlrun,
        )

    def _ensure_reverse_proxy_configurations(self):
        if self.spec.build.functionSourceCode or self.status.container_image:
            return

        filename, handler = ApplicationRuntime.get_filename_and_handler()
        name, spec, code = nuclio.build_file(
            filename,
            name=self.metadata.name,
            handler=handler,
        )
        self.spec.function_handler = mlrun.utils.get_in(spec, "spec.handler")
        self.spec.build.functionSourceCode = mlrun.utils.get_in(
            spec, "spec.build.functionSourceCode"
        )
        self.spec.nuclio_runtime = mlrun.utils.get_in(spec, "spec.runtime")

    def _configure_application_sidecar(self):
        # Save the application image in the status to allow overriding it with the reverse proxy entry point
        if self.spec.image and (
            not self.status.application_image
            or self.spec.image != self.status.container_image
        ):
            self.status.application_image = self.spec.image
            self.spec.image = ""

        if self.status.container_image:
            self.from_image(self.status.container_image)
            # nuclio implementation detail - when providing the image and emptying out the source code,
            # nuclio skips rebuilding the image and simply takes the prebuilt image
            self.spec.build.functionSourceCode = ""

        self.status.sidecar_name = f"{self.metadata.name}-sidecar"
        self.with_sidecar(
            name=self.status.sidecar_name,
            image=self.status.application_image,
            ports=self.spec.internal_application_port,
            command=self.spec.command,
            args=self.spec.args,
        )
        self.set_env("SIDECAR_PORT", self.spec.internal_application_port)
        self.set_env("SIDECAR_HOST", "http://localhost")

    def _sync_api_gateway(self):
        if not self.status.api_gateway_name:
            return

        db = self._get_db()
        api_gateway_scheme = db.get_api_gateway(
            name=self.status.api_gateway_name, project=self.metadata.project
        )
        self.status.api_gateway = APIGateway.from_scheme(api_gateway_scheme)
        self.status.api_gateway.wait_for_readiness()
        self.url = self.status.api_gateway.invoke_url
