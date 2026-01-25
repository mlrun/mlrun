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

import kubernetes.client as k8s_client

import mlrun
import mlrun.common.schemas
import mlrun.utils

import framework.utils.singletons.k8s
import services.api.utils.builder as builder_utils
import services.api.utils.image_builder.base as image_builder


class BuildahImageBuilder(image_builder.BaseImageBuilder):
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
        registry = self._resolve_registry(dest, registry)
        extra_runtime_spec = self._extract_extra_runtime_spec(
            project,
            runtime_spec,
            project_default_fucntion_node_selector,
            auth_info,
        )
        self._validate_dockerfile_or_text(dockertext, dockerfile)

        context_source = None
        if "://" in context and not context.startswith("/"):
            context_source = context
            context = "/context"

        if dockertext:
            dockerfile = "/empty/Dockerfile"

        tls_verify_pull = self._resolve_tls_verify_mode(
            mlrun.mlconf.httpdb.builder.insecure_pull_registry_mode, secret_name
        )
        self._resolve_tls_verify_mode(
            mlrun.mlconf.httpdb.builder.insecure_push_registry_mode, secret_name
        )

        build_args_flags = self._buildah_build_args_flags(
            builder_env=builder_env,
            project_secrets=project_secrets,
            extra_args=extra_args,
        )

        log_level = "debug" if verbose else "info"
        common_args = [
            "--storage-driver=vfs",
            f"--tls-verify={str(tls_verify_pull).lower()}",
        ]

        build_cmd = " ".join(
            [
                "buildah",
                f"--log-level={log_level}",
                "build",
                *common_args,
                f"--file={dockerfile}",
                f"--tag={dest}",
                *build_args_flags,
                context,
            ]
        )
        push_cmd = " ".join(
            [
                "buildah",
                f"--log-level={log_level}",
                "push",
                *common_args,
                dest,
                f"docker://{dest}",
            ]
        )

        resources = self._resolve_resources(runtime_spec)

        kpod = framework.utils.singletons.k8s.BasePod(
            name or "mlrun-build",
            mlrun.mlconf.httpdb.builder.buildah_image,
            command=["/bin/sh", "-c"],
            args=["set -e; " + build_cmd + "; " + push_cmd],
            kind="build",
            project=project,
            default_pod_spec_attributes=extra_runtime_spec,
            resources=resources,
            labels=extra_labels,
        )

        kpod.mount_empty(name="varlibcontainers", mount_path="/var/lib/containers")
        kpod.env = self._combine_builder_envs(builder_env, project_secrets)

        self._mount_pip_ca_secret(kpod, context)

        self._create_dockerfile_init_container(
            kpod,
            mlrun.mlconf.httpdb.builder.buildah_init_container_image,
            dockertext,
            inline_code,
            inline_path,
            requirements,
            requirements_path,
        )

        if context_source:
            self._materialize_remote_context(kpod, context_source)

        if mlrun.utils.helpers.is_ecr_url(registry):
            repo = self._extract_repo_from_dest(dest)
            kpod.env = kpod.env or []
            kpod.env.append(
                k8s_client.V1EnvVar(name="DOCKER_CONFIG", value="/tmp/.docker")
            )
            self._configure_ecr_env_and_init_container(kpod, registry, repo)
        elif secret_name:
            items = [{"key": ".dockerconfigjson", "path": "config.json"}]
            kpod.mount_secret(secret_name, "/tmp/.docker", items=items)
            kpod.env = kpod.env or []
            kpod.env.append(
                k8s_client.V1EnvVar(name="DOCKER_CONFIG", value="/tmp/.docker")
            )

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
        builder_env: typing.Optional[list[k8s_client.V1EnvVar]],
        project_secrets: typing.Optional[list[k8s_client.V1EnvVar]],
        extra_args: str,
    ) -> list[str]:
        flags = self._generate_build_args(builder_env, project_secrets)

        if not extra_args:
            return flags

        parsed = builder_utils._parse_extra_args(extra_args)
        for val in parsed.get("--build-arg", []):
            flags.extend(["--build-arg", val])

        if "--skip-tls-verify" in parsed:
            flags.append("--tls-verify=false")

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
                mlrun.mlconf.httpdb.builder.buildah_init_container_image,
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
            mlrun.mlconf.httpdb.builder.buildah_git_init_container_image,
            command=["/bin/sh"],
            args=["-c", f"set -e; rm -rf /context/*; {clone_cmd}"],
            name="clone-context",
        )

    def _configure_ecr_env_and_init_container(
        self, kpod: framework.utils.singletons.k8s.BasePod, registry: str, repo: str
    ):
        kpod.mount_empty(name="docker-config", mount_path="/tmp/.docker")

        assume_instance_role = not mlrun.mlconf.httpdb.builder.docker_registry_secret
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

        self._filter_aws_credentials_from_env(kpod)

        if not assume_instance_role:
            aws_credentials_file_env_key = "AWS_SHARED_CREDENTIALS_FILE"
            aws_credentials_file_env_value = "/tmp/aws/credentials"
            init_container_env[aws_credentials_file_env_key] = (
                aws_credentials_file_env_value
            )
            kpod.mount_secret(
                mlrun.mlconf.httpdb.builder.docker_registry_secret,
                path="/tmp/aws",
            )

        kpod.append_init_container(
            mlrun.mlconf.httpdb.builder.kaniko_aws_cli_image,
            command=["/bin/sh"],
            args=["-c", command],
            env=init_container_env,
            name="ecr-setup",
        )
