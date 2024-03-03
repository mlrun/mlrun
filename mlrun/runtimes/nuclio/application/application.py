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
import pathlib

from mlrun.common.schemas import AuthInfo
from mlrun.runtimes import RemoteRuntime
from mlrun.runtimes.nuclio import min_nuclio_versions
from mlrun.runtimes.nuclio.function import NuclioSpec, NuclioStatus


class ApplicationSpec(NuclioSpec):
    _dict_fields = NuclioSpec._dict_fields + [
        "internal_app_port",
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
        internal_app_port=None,
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
        self.internal_app_port = internal_app_port or 8080


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


class ApplicationRuntime(RemoteRuntime):
    kind = "application"
    reverse_proxy_file_path = pathlib.Path(__file__).parent / "reverse_proxy.go"

    @min_nuclio_versions("1.12.7")
    def __init__(self, spec=None, metadata=None):
        super().__init__(metadata, spec)

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

    def set_internal_app_port(self, port: int):
        port = int(port)
        if port < 0 or port > 65535:
            raise ValueError("Port must be in the range 0-65535")

        self.spec.internal_app_port = port

    def deploy(
        self,
        project="",
        tag="",
        verbose=False,
        auth_info: AuthInfo = None,
        builder_env: dict = None,
        force_build: bool = False,
    ):
        self._configure_application_sidecar()
        super().deploy(
            project,
            tag,
            verbose,
            auth_info,
            builder_env,
            force_build,
        )

    def _configure_application_sidecar(self):
        # Save the application image in the status to allow overriding it with the reverse proxy entry point
        if (
            self.spec.image and not self.status.application_image
        ) or self.spec.image != self.status.container_image:
            self.status.application_image = self.spec.image
            self.spec.image = ""

        if self.status.container_image:
            self.from_image(self.status.container_image)

        self._with_sidecar(
            name=f"{self.metadata.name}-sidecar",
            image=self.status.application_image,
            port=self.spec.internal_app_port,
        )
        self.set_env("SIDECAR_PORT", self.spec.internal_app_port)
        self.set_env("SIDECAR_HOST", "http://localhost")
