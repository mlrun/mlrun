# Copyright 2026 Iguazio
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


import abc
import base64
import pathlib
import typing

import kubernetes.client as k8s_client

import mlrun
import mlrun.common.schemas
import mlrun.k8s_utils
import mlrun.runtimes.pod
import mlrun.runtimes.utils
import mlrun.utils
import mlrun.utils.helpers

import framework.api.utils
import framework.utils.singletons.k8s


class AbstractBaseImageBuilder(abc.ABC):
    @abc.abstractmethod
    def make_build_pod(
        self,
        project: str,
        context: str,
        dest: str,
        dockerfile: typing.Optional[str] = None,
        dockertext: typing.Optional[str] = None,
        inline_code: typing.Optional[str] = None,
        inline_path: typing.Optional[str] = None,
        requirements: typing.Optional[list] = None,
        requirements_path: typing.Optional[str] = None,
        secret_name: typing.Optional[str] = None,
        name: str = "",
        verbose: bool = False,
        builder_env: typing.Optional[list[k8s_client.V1EnvVar]] = None,
        runtime_spec=None,
        registry: typing.Optional[str] = None,
        extra_args: str = "",
        extra_labels: typing.Optional[dict] = None,
        project_secrets: typing.Optional[list[k8s_client.V1EnvVar]] = None,
        project_default_fucntion_node_selector: typing.Optional[dict] = None,
        auth_info: typing.Optional[mlrun.common.schemas.AuthInfo] = None,
    ) -> framework.utils.singletons.k8s.BasePod:
        """Create the Kubernetes pod spec for building the image."""
        raise NotImplementedError()


class BaseImageBuilder(abc.ABC):
    def _resolve_registry(self, dest: str, registry: typing.Optional[str]) -> str:
        """Resolve the registry from destination if not provided."""
        if not registry:
            registry = dest.partition("/")[0]
        return registry

    def _validate_dockerfile_or_text(
        self, dockertext: typing.Optional[str], dockerfile: typing.Optional[str]
    ) -> None:
        """Validate that docker file or text is specified."""
        if not dockertext and not dockerfile:
            raise ValueError("docker file or text must be specified")

    def _extract_extra_runtime_spec(
        self,
        project: str,
        runtime_spec: mlrun.runtimes.pod.KubeResourceSpec,
        project_default_fucntion_node_selector: typing.Optional[dict],
        auth_info: typing.Optional[mlrun.common.schemas.AuthInfo],
    ) -> dict:
        """Extract runtime spec attributes to apply to the builder pod."""
        extra_runtime_spec: dict = {}
        for attribute, handler in self._get_builder_spec_attributes_from_runtime(
            project,
            runtime_spec,
            project_default_fucntion_node_selector,
            auth_info,
        ).items():
            attr_value = handler(getattr(runtime_spec, attribute, None))
            if attr_value:
                extra_runtime_spec[attribute] = attr_value
        return extra_runtime_spec

    def _combine_builder_envs(
        self,
        builder_env: typing.Optional[list[k8s_client.V1EnvVar]],
        project_secrets: typing.Optional[list[k8s_client.V1EnvVar]],
    ) -> typing.Optional[list[k8s_client.V1EnvVar]]:
        """Combine builder env and project secrets into a single list."""
        envs = (builder_env or []) + (project_secrets or [])
        return envs or None

    def _extract_repo_from_dest(self, dest: str) -> str:
        """Extract repository name from destination for ECR."""
        end = dest.find(":")
        if end == -1:
            end = len(dest)
        return dest[dest.find("/") + 1 : end]

    def _create_dockerfile_init_container(
        self,
        kpod: framework.utils.singletons.k8s.BasePod,
        init_container_image: str,
        dockertext: typing.Optional[str],
        inline_code: typing.Optional[str],
        inline_path: typing.Optional[str],
        requirements: typing.Optional[list],
        requirements_path: typing.Optional[str],
    ) -> None:
        """Create init container for dockerfile, inline code, and requirements."""
        if not (dockertext or inline_code or requirements):
            return

        kpod.mount_empty()
        commands = []
        env = {}

        if dockertext:
            env["DOCKERFILE"] = base64.b64encode(dockertext.encode("utf-8")).decode(
                "utf-8"
            )
            commands.append("echo ${DOCKERFILE} | base64 -d > /empty/Dockerfile")

        if inline_code:
            filename = inline_path or "main.py"
            env["CODE"] = base64.b64encode(inline_code.encode("utf-8")).decode("utf-8")
            commands.append("echo ${CODE} | base64 -d > /empty/" + filename)

        if requirements:
            requirements_file_content = "{}\n".format("\n".join(requirements))
            env["REQUIREMENTS"] = base64.b64encode(
                requirements_file_content.encode("utf-8")
            ).decode("utf-8")
            commands.append(
                "echo ${REQUIREMENTS}" + " | " + f"base64 -d > {requirements_path}"
            )

        kpod.append_init_container(
            init_container_image,
            args=["sh", "-c", "; ".join(commands)],
            env=env,
            name="create-dockerfile",
        )

    def _mount_pip_ca_secret(
        self,
        kpod: framework.utils.singletons.k8s.BasePod,
        context: str,
    ) -> None:
        """Mount pip CA certificate secret if configured.

        This allows pip to verify SSL certificates when installing packages
        from a private PyPI server that uses a custom CA.
        """
        if not mlrun.mlconf.is_pip_ca_configured():
            return

        path = pathlib.Path(mlrun.mlconf.httpdb.builder.pip_ca_path).name
        secret_key = mlrun.mlconf.httpdb.builder.pip_ca_secret_key
        secret_name = mlrun.mlconf.httpdb.builder.pip_ca_secret_name
        kpod.mount_secret(
            secret_name,
            str(pathlib.Path(context) / path),
            items=[
                {
                    "key": secret_key,
                    "path": path,
                }
            ],
            # using sub_path so file will be mounted inside the build pod as regular file
            # and not symlink (if it's symlink it won't work inside the job image itself)
            sub_path=path,
        )

    def _generate_build_args(
        self,
        builder_env: typing.Optional[list[k8s_client.V1EnvVar]],
        project_secrets: typing.Optional[list[k8s_client.V1EnvVar]],
    ) -> list[str]:
        """Generate --build-arg flags from builder env and project secrets.

        Builder env values are used directly (plain text).
        Project secrets use $ reference to read from injected environment variables.
        """
        builder_env = builder_env or []
        project_secrets = project_secrets or []

        args: list[str] = []
        for env in builder_env:
            args.extend(["--build-arg", f"{env.name}={env.value}"])

        for secret in project_secrets:
            args.extend(["--build-arg", f"{secret.name}=${secret.name}"])

        return args

    def _filter_aws_credentials_from_env(
        self, kpod: framework.utils.singletons.k8s.BasePod
    ) -> None:
        """Filter AWS credentials from pod env to avoid conflicts.

        Project secrets might conflict with attached instance role or docker registry secret.
        Remove AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to prevent credential conflicts.
        """
        kpod.env = kpod.env or []
        kpod.env = [
            env_var
            for env_var in kpod.env
            if env_var.name not in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        ]

    def _get_ecr_region(self, registry: str) -> str:
        """Extract AWS region from ECR registry URL.

        ECR URLs follow the pattern: <account>.dkr.ecr.<region>.amazonaws.com
        """
        return registry.split(".")[3]

    def _mount_aws_credentials_secret(
        self, kpod: framework.utils.singletons.k8s.BasePod
    ) -> dict:
        """Mount AWS credentials secret and return init container env config.

        Used when not relying on instance role for ECR authentication.
        Returns a dict with AWS_SHARED_CREDENTIALS_FILE for init container env.
        """
        aws_credentials_file_env_key = "AWS_SHARED_CREDENTIALS_FILE"
        aws_credentials_file_env_value = "/tmp/aws/credentials"

        kpod.mount_secret(
            mlrun.mlconf.httpdb.builder.docker_registry_secret,
            path="/tmp/aws",
        )

        return {aws_credentials_file_env_key: aws_credentials_file_env_value}

    def _get_builder_spec_attributes_from_runtime(
        self,
        project: str,
        runtime_spec: mlrun.runtimes.pod.KubeResourceSpec,
        project_default_fucntion_node_selector: typing.Optional[dict] = None,
        auth_info: mlrun.common.schemas.AuthInfo = None,
    ) -> dict[str, typing.Callable[[typing.Any], typing.Any]]:
        """
        Get the names of spec attributes that are defined
        for runtime but should also be applied to Builder pod.
        """

        project_default_fucntion_node_selector = (
            project_default_fucntion_node_selector or {}
        )

        # preemption mode scheduling constraints cache
        _preemption_enrichment_result: dict = {}

        def service_account_handler(attr_value):
            (
                allowed_service_accounts,
                forbidden_service_accounts,
                default_service_account,
            ) = framework.api.utils.resolve_project_service_account_details(
                project, auth_info=auth_info
            )
            if attr_value:
                runtime_spec.validate_service_account(
                    allowed_service_accounts, forbidden_service_accounts
                )
            else:
                attr_value = default_service_account
            return attr_value

        def get_merged_node_selector(attr_value):
            attr_value = mlrun.utils.to_non_empty_values_dict(
                mlrun.utils.helpers.merge_dicts_with_precedence(
                    mlrun.mlconf.get_default_function_node_selector(),
                    project_default_fucntion_node_selector,
                    attr_value,
                )
            )
            return attr_value

        def preemption_mode_handler(key):
            if key not in _preemption_enrichment_result:
                keys = ["node_selector", "tolerations", "affinity"]
                values = mlrun.k8s_utils.enrich_preemption_mode(
                    preemption_mode=runtime_spec.preemption_mode,
                    node_selector=get_merged_node_selector(runtime_spec.node_selector),
                    affinity=runtime_spec.affinity,
                    tolerations=runtime_spec.tolerations,
                )
                _preemption_enrichment_result.update(dict(zip(keys, values)))
            return _preemption_enrichment_result[key]

        def node_selector_handler(attr_value):
            return preemption_mode_handler("node_selector")

        def affinity_handler(attr_value):
            return preemption_mode_handler("affinity")

        def tolerations_handler(attr_value):
            return preemption_mode_handler("tolerations")

        def identity_handler(attr_value):
            return attr_value

        return {
            "node_name": identity_handler,
            "node_selector": node_selector_handler,
            "affinity": affinity_handler,
            "tolerations": tolerations_handler,
            "priority_class_name": identity_handler,
            "service_account": service_account_handler,
        }

    def _resolve_resources(
        self, runtime_spec: mlrun.runtimes.pod.KubeResourceSpec
    ) -> dict:
        # While requests mainly affect scheduling, setting a limit may prevent Builder pod
        # from finishing successfully (destructive), since we're not allowing to override the default
        # specifically for the builder pod, we're setting only the requests
        # we cannot specify gpu requests without specifying gpu limits, so we set requests without gpu field
        default_requests = mlrun.mlconf.get_default_function_pod_requirement_resources(
            "requests", with_gpu=False
        )
        resources = {
            "requests": mlrun.runtimes.utils.generate_resources(
                mem=default_requests.get("memory"), cpu=default_requests.get("cpu")
            )
        }
        # Some cloud providers add a toleration when a GPU limit is set.
        # If the builder pod inherits a GPU-related node selector from the function
        # but lacks a GPU limit, it may get stuck in a pending state due to unsatisfiable scheduling.
        # Setting GPU limits to zero ensures tolerations are applied while preventing GPU allocation.
        if runtime_spec:
            gpu_resources = mlrun.utils.get_enriched_gpu_limits(
                runtime_spec.resources.get("limits", {})
            )
            if gpu_resources:
                resources["limits"] = gpu_resources
        return resources

    def _materialize_http_context(
        self,
        kpod: framework.utils.singletons.k8s.BasePod,
        context_source: str,
        mount_path: str,
        init_container_image: str,
    ) -> None:
        """Fetch HTTP context tarball and extract to the specified mount path.

        :param kpod: The build pod to add the init container to.
        :param context_source: The HTTP URL of the context tarball.
        :param mount_path: The path to mount the context volume.
        :param init_container_image: The image to use for the init container.
        """
        kpod.mount_empty(name="context", mount_path=mount_path)

        # Download and extract tarball, or just download single file if not tarball
        cmd = (
            f"wget -qO- {context_source} | tar -xz -C {mount_path} || "
            f"(wget -qO {mount_path}/source {context_source} && true)"
        )

        kpod.append_init_container(
            init_container_image,
            command=["/bin/sh"],
            args=["-c", f"set -e; {cmd}"],
            name="fetch-context",
        )
