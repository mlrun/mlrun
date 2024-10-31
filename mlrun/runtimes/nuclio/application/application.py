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
import nuclio.auth

import mlrun.common.schemas as schemas
import mlrun.errors
import mlrun.run
from mlrun.common.runtimes.constants import NuclioIngressAddTemplatedIngressModes
from mlrun.runtimes import RemoteRuntime
from mlrun.runtimes.nuclio import min_nuclio_versions
from mlrun.runtimes.nuclio.api_gateway import (
    APIGateway,
    APIGatewayMetadata,
    APIGatewaySpec,
)
from mlrun.runtimes.nuclio.function import NuclioSpec, NuclioStatus
from mlrun.utils import logger, update_in


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

        # Override default min/max replicas (don't assume application is stateless)
        self.min_replicas = min_replicas or 1
        self.max_replicas = max_replicas or 1

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
        application_source=None,
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
        self.application_source = application_source or None
        self.sidecar_name = sidecar_name or None
        self.api_gateway_name = api_gateway_name or None
        self.api_gateway: typing.Optional[APIGateway] = api_gateway or None
        self.url = url or None


class ApplicationRuntime(RemoteRuntime):
    kind = "application"
    reverse_proxy_image = None

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

    def prepare_image_for_deploy(self):
        if self.spec.build.source and self.spec.build.load_source_on_run:
            logger.warning(
                "Application runtime requires loading the source into the application image. "
                f"Even though {self.spec.build.load_source_on_run=}, loading on build will be forced."
            )
            self.spec.build.load_source_on_run = False
        super().prepare_image_for_deploy()

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
        create_default_api_gateway: bool = True,
    ):
        """
        Deploy function, builds the application image if required (self.requires_build()) or force_build is True,
        Once the image is built, the function is deployed.

        :param project:                     Project name
        :param tag:                         Function tag
        :param verbose:                     Set True for verbose logging
        :param auth_info:                   Service AuthInfo (deprecated and ignored)
        :param builder_env:                 Env vars dict for source archive config/credentials
                                            e.g. builder_env={"GIT_TOKEN": token}
        :param force_build:                 Set True for force building the application image
        :param with_mlrun:                  Add the current mlrun package to the container build
        :param skip_deployed:               Skip the build if we already have an image for the function
        :param is_kfp:                      Deploy as part of a kfp pipeline
        :param mlrun_version_specifier:     Which mlrun package version to include (if not current)
        :param show_on_failure:             Show logs only in case of build failure
        :param create_default_api_gateway:  When deploy finishes the default API gateway will be created for the
                                            application. Disabling this flag means that the application will not be
                                            accessible until an API gateway is created for it.

        :return: The default API gateway URL if created or True if the function is ready (deployed)
        """
        if (self.requires_build() and not self.spec.image) or force_build:
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

        # This is a class method that accepts a function instance, so we pass self as the function instance
        self._ensure_reverse_proxy_configurations(self)
        self._configure_application_sidecar()

        # We only allow accessing the application via the API Gateway
        self.spec.add_templated_ingress_host_mode = (
            NuclioIngressAddTemplatedIngressModes.never
        )

        super().deploy(
            project=project,
            tag=tag,
            verbose=verbose,
            auth_info=auth_info,
            builder_env=builder_env,
        )
        logger.info(
            "Successfully deployed function.",
        )

        # Restore the source in case it was removed to make nuclio not consider it when building
        if not self.spec.build.source and self.status.application_source:
            self.spec.build.source = self.status.application_source
        self.save(versioned=False)

        if create_default_api_gateway:
            try:
                api_gateway_name = self.resolve_default_api_gateway_name()
                return self.create_api_gateway(api_gateway_name, set_as_default=True)
            except Exception as exc:
                logger.warning(
                    "Failed to create default API gateway, application may not be accessible. "
                    "Use the `create_api_gateway` method to make it accessible",
                    exc=mlrun.errors.err_to_str(exc),
                )
        elif not self.status.api_gateway:
            logger.warning(
                "Application is online but may not be accessible since default gateway creation was not requested."
                "Use the `create_api_gateway` method to make it accessible."
            )

        return True

    def with_source_archive(
        self,
        source,
        workdir=None,
        pull_at_runtime: bool = False,
        target_dir: str = None,
    ):
        """load the code from git/tar/zip archive at build

        :param source:          valid absolute path or URL to git, zip, or tar file, e.g.
                                git://github.com/mlrun/something.git
                                http://some/url/file.zip
                                note path source must exist on the image or exist locally when run is local
                                (it is recommended to use 'workdir' when source is a filepath instead)
        :param workdir:         working dir relative to the archive root (e.g. './subdir') or absolute to the image root
        :param pull_at_runtime: currently not supported, source must be loaded into the image during the build process
        :param target_dir:      target dir on runtime pod or repo clone / archive extraction
        """
        if pull_at_runtime:
            logger.warning(
                f"{pull_at_runtime=} is currently not supported for application runtime "
                "and will be overridden to False",
                pull_at_runtime=pull_at_runtime,
            )

        self._configure_mlrun_build_with_source(
            source=source,
            workdir=workdir,
            pull_at_runtime=False,
            target_dir=target_dir,
        )

    def from_image(self, image):
        """
        Deploy the function with an existing nuclio processor image.
        This applies only for the reverse proxy and not the application image.

        :param image: image name
        """
        super().from_image(image)
        # nuclio implementation detail - when providing the image and emptying out the source code and build source,
        # nuclio skips rebuilding the image and simply takes the prebuilt image
        self.spec.build.functionSourceCode = ""
        self.status.application_source = self.spec.build.source
        self.spec.build.source = ""

        # save the image in the status, so we won't repopulate the function source code
        self.status.container_image = image

        # ensure golang runtime and handler for the reverse proxy
        self.spec.nuclio_runtime = "golang"
        update_in(
            self.spec.base_spec,
            "spec.handler",
            "main:Handler",
        )

    @staticmethod
    def get_filename_and_handler() -> (str, str):
        reverse_proxy_file_path = pathlib.Path(__file__).parent / "reverse_proxy.go"
        return str(reverse_proxy_file_path), "Handler"

    def create_api_gateway(
        self,
        name: str = None,
        path: str = None,
        direct_port_access: bool = False,
        authentication_mode: schemas.APIGatewayAuthenticationMode = None,
        authentication_creds: tuple[str, str] = None,
        ssl_redirect: bool = None,
        set_as_default: bool = False,
        gateway_timeout: typing.Optional[int] = None,
    ):
        """
        Create the application API gateway. Once the application is deployed, the API gateway can be created.
        An application without an API gateway is not accessible.

        :param name:                    The name of the API gateway
        :param path:                    Optional path of the API gateway, default value is "/".
                                        The given path should be supported by the deployed application
        :param direct_port_access:      Set True to allow direct port access to the application sidecar
        :param authentication_mode:     API Gateway authentication mode
        :param authentication_creds:    API Gateway basic authentication credentials as a tuple (username, password)
        :param ssl_redirect:            Set True to force SSL redirect, False to disable. Defaults to
                                        mlrun.mlconf.force_api_gateway_ssl_redirect()
        :param set_as_default:          Set the API gateway as the default for the application (`status.api_gateway`)
        :param gateway_timeout:         nginx ingress timeout in sec (request timeout, when will the gateway return an
                                        error)
        :return:                        The API gateway URL
        """
        if not name:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "API gateway name must be specified."
            )

        if not set_as_default and name == self.resolve_default_api_gateway_name():
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Non-default API gateway cannot use the default gateway name, {name=}."
            )

        if (
            authentication_mode == schemas.APIGatewayAuthenticationMode.basic
            and not authentication_creds
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Authentication credentials not provided"
            )

        ports = self.spec.internal_application_port if direct_port_access else []

        api_gateway = APIGateway(
            APIGatewayMetadata(
                name=name,
                namespace=self.metadata.namespace,
                labels=self.metadata.labels.copy(),
            ),
            APIGatewaySpec(
                functions=[self],
                project=self.metadata.project,
                path=path,
                ports=mlrun.utils.helpers.as_list(ports) if ports else None,
            ),
        )

        api_gateway.with_gateway_timeout(gateway_timeout)
        if ssl_redirect is None:
            ssl_redirect = mlrun.mlconf.force_api_gateway_ssl_redirect()
        if ssl_redirect:
            # Force ssl redirect so that the application is only accessible via https
            api_gateway.with_force_ssl_redirect()

        # Add authentication if required
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

        if set_as_default:
            self.status.api_gateway_name = api_gateway_scheme.metadata.name
            self.status.api_gateway = APIGateway.from_scheme(api_gateway_scheme)
            self.status.api_gateway.wait_for_readiness()
            self.url = self.status.api_gateway.invoke_url
            url = self.url
        else:
            api_gateway = APIGateway.from_scheme(api_gateway_scheme)
            api_gateway.wait_for_readiness()
            url = api_gateway.invoke_url
            # Update application status (enriches invocation url)
            self._get_state(raise_on_exception=False)

        logger.info("Successfully created API gateway", url=url)
        return url

    def delete_api_gateway(self, name: str):
        """
        Delete API gateway by name.
        Refreshes the application status to update api gateway and invocation URLs.
        :param name:    The API gateway name
        """
        self._get_db().delete_api_gateway(name=name, project=self.metadata.project)
        if name == self.status.api_gateway_name:
            self.status.api_gateway_name = None
            self.status.api_gateway = None
        self._get_state()

    def invoke(
        self,
        path: str = "",
        body: typing.Optional[typing.Union[str, bytes, dict]] = None,
        method: str = None,
        headers: dict = None,
        dashboard: str = "",
        force_external_address: bool = False,
        auth_info: schemas.AuthInfo = None,
        mock: bool = None,
        credentials: tuple[str, str] = None,
        **http_client_kwargs,
    ):
        self._sync_api_gateway()

        # If the API Gateway is not ready or not set, try to invoke the function directly (without the API Gateway)
        if not self.status.api_gateway:
            logger.warning(
                "Default API gateway is not configured, invoking function invocation URL."
            )
            # create a requests auth object if credentials are provided and not already set in the http client kwargs
            auth = http_client_kwargs.pop("auth", None) or (
                nuclio.auth.AuthInfo(
                    username=credentials[0], password=credentials[1]
                ).to_requests_auth()
                if credentials
                else None
            )
            return super().invoke(
                path,
                body,
                method,
                headers,
                dashboard,
                force_external_address,
                auth_info,
                mock,
                auth=auth,
                **http_client_kwargs,
            )

        if not method:
            method = "POST" if body else "GET"

        return self.status.api_gateway.invoke(
            method=method,
            headers=headers,
            credentials=credentials,
            path=path,
            body=body,
            **http_client_kwargs,
        )

    @classmethod
    def deploy_reverse_proxy_image(cls):
        """
        Build the reverse proxy image and save it.
        The reverse proxy image is used to route requests to the application sidecar.
        This is useful when you want to decrease build time by building the application image only once.

        :param use_cache:   Use the cache when building the image
        """
        # create a function that includes only the reverse proxy, without the application

        reverse_proxy_func = mlrun.run.new_function(
            name="reverse-proxy-temp", kind="remote"
        )
        # default max replicas is 4, we only need one replica for the reverse proxy
        reverse_proxy_func.spec.max_replicas = 1

        # the reverse proxy image should not be based on another image
        reverse_proxy_func.set_config("spec.build.baseImage", None)
        reverse_proxy_func.spec.image = ""
        reverse_proxy_func.spec.build.base_image = ""

        cls._ensure_reverse_proxy_configurations(reverse_proxy_func)
        reverse_proxy_func.deploy()

        # save the created container image
        cls.reverse_proxy_image = reverse_proxy_func.status.container_image

        # delete the function to avoid cluttering the project
        mlrun.get_run_db().delete_function(
            reverse_proxy_func.metadata.name, reverse_proxy_func.metadata.project
        )

    def resolve_default_api_gateway_name(self):
        return (
            f"{self.metadata.name}-{self.metadata.tag}"
            if self.metadata.tag
            else self.metadata.name
        )

    @min_nuclio_versions("1.13.1")
    def disable_default_http_trigger(
        self,
    ):
        raise mlrun.runtimes.RunError(
            "Application runtime does not support disabling the default HTTP trigger"
        )

    @min_nuclio_versions("1.13.1")
    def enable_default_http_trigger(
        self,
    ):
        pass

    def _run(self, runobj: "mlrun.RunObject", execution):
        raise mlrun.runtimes.RunError(
            "Application runtime .run() is not yet supported. Use .invoke() instead."
        )

    def _enrich_command_from_status(self):
        pass

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

        if self.spec.build.source in [".", "./"]:
            logger.info(
                "The application is configured to use the project's source. "
                "Application runtime requires loading the source into the application image. "
                "Loading on build will be forced regardless of whether 'pull_at_runtime=True' was configured."
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

    @staticmethod
    def _ensure_reverse_proxy_configurations(function: RemoteRuntime):
        if function.spec.build.functionSourceCode or function.status.container_image:
            return

        filename, handler = ApplicationRuntime.get_filename_and_handler()
        name, spec, code = nuclio.build_file(
            filename,
            name=function.metadata.name,
            handler=handler,
        )
        function.spec.function_handler = mlrun.utils.get_in(spec, "spec.handler")
        function.spec.build.functionSourceCode = mlrun.utils.get_in(
            spec, "spec.build.functionSourceCode"
        )
        function.spec.nuclio_runtime = mlrun.utils.get_in(spec, "spec.runtime")

        # default the reverse proxy logger level to info
        logger_sinks_key = "spec.loggerSinks"
        if not function.spec.config.get(logger_sinks_key):
            function.set_config(
                logger_sinks_key, [{"level": "info", "sink": "myStdoutLoggerSink"}]
            )

    def _configure_application_sidecar(self):
        # Save the application image in the status to allow overriding it with the reverse proxy entry point
        if self.spec.image and (
            not self.status.application_image
            or self.spec.image != self.status.container_image
        ):
            self.status.application_image = self.spec.image
            self.spec.image = ""

        # reuse the reverse proxy image if it was built before
        if (
            reverse_proxy_image := self.status.container_image
            or self.reverse_proxy_image
        ):
            self.from_image(reverse_proxy_image)

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

        # configure the sidecar container as the default container for logging purposes
        self.metadata.annotations["kubectl.kubernetes.io/default-container"] = (
            self.status.sidecar_name
        )

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
