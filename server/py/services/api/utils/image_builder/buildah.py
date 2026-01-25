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


import typing

from kubernetes import client

import mlrun.common.schemas
import mlrun.runtimes.utils
import mlrun.utils
from mlrun.config import config

import framework.utils.singletons.k8s
from services.api.utils.image_builder.base import BaseImageBuilder


class BuildahImageBuilder(BaseImageBuilder):
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
        builder_env: typing.Optional[list[client.V1EnvVar]] = None,
        runtime_spec=None,
        registry: typing.Optional[str] = None,
        extra_args: str = "",
        extra_labels: typing.Optional[dict] = None,
        project_secrets: typing.Optional[list[client.V1EnvVar]] = None,
        project_default_fucntion_node_selector: typing.Optional[dict] = None,
        auth_info: typing.Optional[mlrun.common.schemas.AuthInfo] = None,
    ) -> framework.utils.singletons.k8s.BasePod:
        from services.api.utils import builder as builder_utils

        extra_runtime_spec: dict = {}
        if not registry:
            registry = dest.partition("/")[0]

        for attribute, handler in self.get_kaniko_spec_attributes_from_runtime(
            project,
            runtime_spec,
            project_default_fucntion_node_selector,
            auth_info,
        ).items():
            attr_value = handler(getattr(runtime_spec, attribute, None))
            if attr_value:
                extra_runtime_spec[attribute] = attr_value

        if not dockertext and not dockerfile:
            raise ValueError("docker file or text must be specified")

        context_source = None
        if "://" in context and not context.startswith("/"):
            context_source = context
            context = "/context"

        if dockertext:
            dockerfile = "/empty/Dockerfile"

        tls_verify_pull = self._resolve_tls_verify_mode(
            config.httpdb.builder.insecure_pull_registry_mode, secret_name
        )
        tls_verify_push = self._resolve_tls_verify_mode(
            config.httpdb.builder.insecure_push_registry_mode, secret_name
        )

        build_args_flags = self._buildah_build_args_flags(
            builder_env=builder_env,
            project_secrets=project_secrets,
            extra_args=extra_args,
        )

        log_level = "debug" if verbose else "info"

        build_cmd = " ".join(
            [
                "buildah",
                "--log-level",
                log_level,
                "bud",
                "--storage-driver=vfs",
                f"--tls-verify={'true' if tls_verify_pull else 'false'}",
                f"--file {dockerfile}",
                f"--tag {dest}",
                *build_args_flags,
                context,
            ]
        )
        push_cmd = " ".join(
            [
                "buildah",
                "--log-level",
                log_level,
                "push",
                "--storage-driver=vfs",
                f"--tls-verify={'true' if tls_verify_push else 'false'}",
                dest,
                f"docker://{dest}",
            ]
        )

        default_requests = config.get_default_function_pod_requirement_resources(
            "requests", with_gpu=False
        )
        resources = {
            "requests": mlrun.runtimes.utils.generate_resources(
                mem=default_requests.get("memory"), cpu=default_requests.get("cpu")
            )
        }
        if runtime_spec:
            gpu_resources = mlrun.utils.get_enriched_gpu_limits(
                runtime_spec.resources.get("limits", {})
            )
            if gpu_resources:
                resources["limits"] = gpu_resources

        kpod = framework.utils.singletons.k8s.BasePod(
            name or "mlrun-build",
            config.httpdb.builder.buildah_image,
            command=["/bin/sh", "-c"],
            args=["set -e; " + build_cmd + "; " + push_cmd],
            kind="build",
            project=project,
            default_pod_spec_attributes=extra_runtime_spec,
            resources=resources,
            labels=extra_labels,
        )

        kpod.mount_empty(name="varlibcontainers", mount_path="/var/lib/containers")

        envs = (builder_env or []) + (project_secrets or [])
        kpod.env = envs or None

        if dockertext or inline_code or requirements:
            kpod.mount_empty()
            commands = []
            env = {}
            if dockertext:
                from base64 import b64encode

                env["DOCKERFILE"] = b64encode(dockertext.encode("utf-8")).decode(
                    "utf-8"
                )
                commands.append("echo ${DOCKERFILE} | base64 -d > /empty/Dockerfile")
            if inline_code:
                from base64 import b64encode

                filename = inline_path or "main.py"
                env["CODE"] = b64encode(inline_code.encode("utf-8")).decode("utf-8")
                commands.append("echo ${CODE} | base64 -d > /empty/" + filename)
            if requirements:
                from base64 import b64encode

                requirements_file_content = "{}\n".format("\n".join(requirements))
                env["REQUIREMENTS"] = b64encode(
                    requirements_file_content.encode("utf-8")
                ).decode("utf-8")
                commands.append(
                    "echo ${REQUIREMENTS}" + " | " + f"base64 -d > {requirements_path}"
                )

            kpod.append_init_container(
                config.httpdb.builder.kaniko_init_container_image,
                args=["sh", "-c", "; ".join(commands)],
                env=env,
                name="create-dockerfile",
            )

        if context_source:
            self._materialize_remote_context(kpod, context_source)

        if mlrun.utils.helpers.is_ecr_url(registry):
            end = dest.find(":")
            if end == -1:
                end = len(dest)
            repo = dest[dest.find("/") + 1 : end]
            kpod.env = kpod.env or []
            kpod.env.append(client.V1EnvVar(name="DOCKER_CONFIG", value="/tmp/.docker"))
            self.configure_buildah_ecr_env_and_init_container(kpod, registry, repo)
        elif secret_name:
            items = [{"key": ".dockerconfigjson", "path": "config.json"}]
            kpod.mount_secret(secret_name, "/tmp/.docker", items=items)
            kpod.env = kpod.env or []
            kpod.env.append(client.V1EnvVar(name="DOCKER_CONFIG", value="/tmp/.docker"))

        builder_utils._validate_extra_args(extra_args)

        return kpod

    def _resolve_tls_verify_mode(
        self, mode: str, secret_name: typing.Optional[str]
    ) -> bool:
        if mode == "disabled":
            return True
        if mode == "enabled":
            return False
        # auto
        return bool(secret_name)

    def _buildah_build_args_flags(
        self,
        builder_env: typing.Optional[list[client.V1EnvVar]],
        project_secrets: typing.Optional[list[client.V1EnvVar]],
        extra_args: str,
    ) -> list[str]:
        from services.api.utils import builder as builder_utils

        builder_env = builder_env or []
        project_secrets = project_secrets or []

        flags: list[str] = []
        for env in builder_env:
            flags.extend(["--build-arg", f"{env.name}={env.value}"])

        for secret in project_secrets:
            flags.extend(["--build-arg", f"{secret.name}=${secret.name}"])

        if not extra_args:
            return flags

        parsed = builder_utils._parse_extra_args(extra_args)
        for val in parsed.get("--build-arg", []):
            flags.extend(["--build-arg", val])

        if "--skip-tls-verify" in parsed:
            flags.extend(["--tls-verify=false"])

        return flags

    def _materialize_remote_context(
        self, kpod: framework.utils.singletons.k8s.BasePod, context_source: str
    ):
        kpod.mount_empty(name="context", mount_path="/context")

        if context_source.startswith("http://") or context_source.startswith(
            "https://"
        ):
            cmd = (
                f"wget -qO- {context_source} | tar -xz -C /context || "
                f"(wget -qO /context/source {context_source} && true)"
            )

            kpod.append_init_container(
                # TODO: enrich image from config
                "alpine:3.20",
                command=["/bin/sh"],
                args=["-c", f"set -e; apk add --no-cache wget tar; {cmd}"],
                name="fetch-context",
            )
            return

        # treat as git url (supports git://...#refs/heads/<branch>)
        repo_url, _, fragment = context_source.partition("#")
        branch = fragment or ""
        if branch.startswith("refs/heads/"):
            branch = branch.removeprefix("refs/heads/")

        clone_cmd = (
            f"git clone --depth 1 {repo_url} /context"
            if not branch
            else f"git clone --depth 1 --branch {branch} {repo_url} /context"
        )

        kpod.append_init_container(
            # TODO: enrich image from config
            "alpine/git:2.45.2",
            command=["/bin/sh"],
            args=["-c", f"set -e; rm -rf /context/*; {clone_cmd}"],
            name="clone-context",
        )

    def configure_buildah_ecr_env_and_init_container(
        self, kpod: framework.utils.singletons.k8s.BasePod, registry: str, repo: str
    ):
        kpod.mount_empty(name="docker-config", mount_path="/tmp/.docker")

        assume_instance_role = not config.httpdb.builder.docker_registry_secret
        region = registry.split(".")[3]

        command = (
            f"set -e; "
            f"aws ecr create-repository --region {region} --repository-name {repo} || true; "
            f"aws ecr create-repository --region {region} --repository-name {repo}/cache || true; "
            f"PASS=$(aws ecr get-login-password --region {region}); "
            r'AUTH=$(printf "AWS:%s" "$PASS" | base64 | tr -d "\n"); '
            f'cat > /tmp/.docker/config.json <<EOF\n{{"auths": {{"{registry}": {{"auth": "$AUTH"}}}}}}\nEOF'
        )

        init_container_env = {}

        kpod.env = kpod.env or []
        kpod.env = [
            env_var
            for env_var in kpod.env
            if env_var.name not in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        ]

        if not assume_instance_role:
            aws_credentials_file_env_key = "AWS_SHARED_CREDENTIALS_FILE"
            aws_credentials_file_env_value = "/tmp/aws/credentials"
            init_container_env[aws_credentials_file_env_key] = (
                aws_credentials_file_env_value
            )
            kpod.mount_secret(
                config.httpdb.builder.docker_registry_secret,
                path="/tmp/aws",
            )

        kpod.append_init_container(
            config.httpdb.builder.kaniko_aws_cli_image,
            command=["/bin/sh"],
            args=["-c", command],
            env=init_container_env,
            name="ecr-setup",
        )
