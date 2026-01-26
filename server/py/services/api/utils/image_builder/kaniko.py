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


class KanikoImageBuilder(image_builder.BaseImageBuilder):
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

        if dockertext:
            dockerfile = "/empty/Dockerfile"

        # Determine if we need to materialize remote context via init container
        # Kaniko supports https://, git://, gs://, s3://, tar:// natively,
        # but NOT http:// - we need to fetch http:// contexts with an init container
        context_source = None
        effective_context = context
        if context.startswith("http://"):
            # Use Kaniko's dir:// prefix to point to /workspace where we'll download
            context_source = context
            effective_context = "dir:///workspace"

        args = [
            "--dockerfile",
            dockerfile,
            "--context",
            effective_context,
            "--destination",
            dest,
            "--image-fs-extract-retry",
            mlrun.mlconf.httpdb.builder.kaniko_image_fs_extraction_retries,
            "--push-retry",
            mlrun.mlconf.httpdb.builder.kaniko_image_push_retry,
        ]
        for value, flag in [
            (
                mlrun.mlconf.httpdb.builder.insecure_pull_registry_mode,
                "--insecure-pull",
            ),
            (mlrun.mlconf.httpdb.builder.insecure_push_registry_mode, "--insecure"),
        ]:
            if value == "disabled":
                continue
            if value == "enabled" or (value == "auto" and not secret_name):
                args.append(flag)
        if verbose:
            args += ["--verbosity", "debug"]

        args = self._add_args_with_all_build_args(
            args, builder_env, project_secrets, extra_args
        )

        resources = self._resolve_resources(runtime_spec)

        kpod = framework.utils.singletons.k8s.BasePod(
            name or "mlrun-build",
            mlrun.mlconf.httpdb.builder.kaniko_image,
            args=args,
            kind="build",
            project=project,
            default_pod_spec_attributes=extra_runtime_spec,
            resources=resources,
            labels=extra_labels,
        )
        kpod.env = self._combine_builder_envs(builder_env, project_secrets)

        self._mount_pip_ca_secret(kpod, context)

        self._create_dockerfile_init_container(
            kpod,
            mlrun.mlconf.httpdb.builder.kaniko_init_container_image,
            dockertext,
            inline_code,
            inline_path,
            requirements,
            requirements_path,
        )

        # Materialize remote HTTP context if needed
        if context.startswith("http://"):
            self._handle_http_context(kpod, context_source)

        # when using ECR we need init container to create the image repository
        if mlrun.utils.helpers.is_ecr_url(registry):
            dest_repo = self._extract_repo_from_dest(dest)
            self._configure_ecr_env_and_init_container(kpod, registry, dest_repo)

        # mount regular docker config secret
        elif secret_name:
            items = [{"key": ".dockerconfigjson", "path": "config.json"}]
            kpod.mount_secret(secret_name, "/kaniko/.docker", items=items)

        return kpod

    def _configure_ecr_env_and_init_container(
        self, kpod: framework.utils.singletons.k8s.BasePod, registry: str, repo: str
    ):
        # if no secret is given, assume ec2 instance has attached role which provides read/write access to ECR
        assume_instance_role = not mlrun.mlconf.httpdb.builder.docker_registry_secret
        region = self._get_ecr_region(registry)

        # fail silently in order to ignore "repository already exists" errors
        # if any other error occurs - kaniko will fail similarly
        command = (
            f"aws ecr create-repository --region {region} --repository-name {repo} || true"
            + f" && aws ecr create-repository --region {region} --repository-name {repo}/cache || true"
        )
        init_container_env = {}

        self._filter_aws_credentials_from_env(kpod)

        if assume_instance_role:
            # assume instance role has permissions to register and store a container image
            # https://github.com/GoogleContainerTools/kaniko#pushing-to-amazon-ecr
            # we only need this in the kaniko container
            kpod.env.append(
                k8s_client.V1EnvVar(name="AWS_SDK_LOAD_CONFIG", value="true")
            )
        else:
            init_container_env = self._mount_aws_credentials_secret(kpod)
            # set the kaniko container AWS credentials location to the mount's path
            for key, value in init_container_env.items():
                kpod.env.append(k8s_client.V1EnvVar(name=key, value=value))

        kpod.append_init_container(
            mlrun.mlconf.httpdb.builder.kaniko_aws_cli_image,
            command=["/bin/sh"],
            args=["-c", command],
            env=init_container_env,
            name="create-repos",
        )

    def _add_args_with_all_build_args(
        self,
        args: list,
        builder_env: typing.Optional[list[k8s_client.V1EnvVar]],
        project_secrets: typing.Optional[list[k8s_client.V1EnvVar]],
        extra_args: str,
    ) -> list:
        args.extend(self._generate_build_args(builder_env, project_secrets))
        return builder_utils.validate_and_merge_args_with_extra_args(args, extra_args)

    def _handle_http_context(
        self, kpod: framework.utils.singletons.k8s.BasePod, context_source: str
    ):
        """Fetch HTTP context for Kaniko.

        Kaniko doesn't support http:// contexts natively (only https://),
        so we use an init container to download and extract the tarball.
        """
        self._materialize_http_context(
            kpod,
            context_source,
            mount_path="/workspace",
            init_container_image=mlrun.mlconf.httpdb.builder.kaniko_init_container_image,
        )
