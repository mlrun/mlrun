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

import base64
import pathlib
import typing

import kubernetes.client as k8s_client

import mlrun
import mlrun.common.schemas
import mlrun.runtimes.utils
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
        extra_runtime_spec: dict = {}
        if not registry:
            # if registry was not given, infer it from the image destination
            registry = dest.partition("/")[0]

        # set kaniko's spec attributes from the runtime spec
        for attribute, handler in self.get_builder_spec_attributes_from_runtime(
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

        if dockertext:
            dockerfile = "/empty/Dockerfile"

        args = [
            "--dockerfile",
            dockerfile,
            "--context",
            context,
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
        envs = (builder_env or []) + (project_secrets or [])
        kpod.env = envs or None

        if mlrun.mlconf.is_pip_ca_configured():
            items = [
                {
                    "key": mlrun.mlconf.httpdb.builder.pip_ca_secret_key,
                    "path": pathlib.Path(mlrun.mlconf.httpdb.builder.pip_ca_path).name,
                }
            ]
            kpod.mount_secret(
                mlrun.mlconf.httpdb.builder.pip_ca_secret_name,
                str(
                    pathlib.Path(context)
                    / pathlib.Path(mlrun.mlconf.httpdb.builder.pip_ca_path).name
                ),
                items=items,
                # using sub_path so file will be mounted inside kaniko pod as regular file and not symlink
                # (if it's symlink it's then not working inside the job image itself)
                sub_path=pathlib.Path(mlrun.mlconf.httpdb.builder.pip_ca_path).name,
            )

        if dockertext or inline_code or requirements:
            kpod.mount_empty()
            commands = []
            env = {}
            if dockertext:
                # set and encode docker content to the DOCKERFILE environment variable in the kaniko pod
                env["DOCKERFILE"] = base64.b64encode(dockertext.encode("utf-8")).decode(
                    "utf-8"
                )
                # dump dockerfile content and decode to Dockerfile destination
                commands.append("echo ${DOCKERFILE} | base64 -d > /empty/Dockerfile")
            if inline_code:
                filename = inline_path or "main.py"
                env["CODE"] = base64.b64encode(inline_code.encode("utf-8")).decode(
                    "utf-8"
                )
                commands.append("echo ${CODE} | base64 -d > /empty/" + filename)
            if requirements:
                # set and encode requirements to the REQUIREMENTS environment variable in the kaniko pod
                requirements_file_content = "{}\n".format("\n".join(requirements))
                env["REQUIREMENTS"] = base64.b64encode(
                    requirements_file_content.encode("utf-8")
                ).decode("utf-8")
                # dump requirement content and decode to the requirement.txt destination
                commands.append(
                    "echo ${REQUIREMENTS}" + " | " + f"base64 -d > {requirements_path}"
                )

            kpod.append_init_container(
                mlrun.mlconf.httpdb.builder.kaniko_init_container_image,
                args=["sh", "-c", "; ".join(commands)],
                env=env,
                name="create-dockerfile",
            )

        # when using ECR we need init container to create the image repository
        if mlrun.utils.helpers.is_ecr_url(registry):
            end = dest.find(":")
            if end == -1:
                end = len(dest)
            repo = dest[dest.find("/") + 1 : end]

            self._configure_ecr_env_and_init_container(kpod, registry, repo)

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
        region = registry.split(".")[3]

        # fail silently in order to ignore "repository already exists" errors
        # if any other error occurs - kaniko will fail similarly
        command = (
            f"aws ecr create-repository --region {region} --repository-name {repo} || true"
            + f" && aws ecr create-repository --region {region} --repository-name {repo}/cache || true"
        )
        init_container_env = {}

        kpod.env = kpod.env or []

        # project secret might conflict with the attached instance role/docker registry secret
        # ensure "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY" have no values or else kaniko will fail
        # due to credentials conflict / lack of permission on given credentials
        kpod.env = [
            env_var
            for env_var in kpod.env
            if env_var.name not in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        ]

        if assume_instance_role:
            # assume instance role has permissions to register and store a container image
            # https://github.com/GoogleContainerTools/kaniko#pushing-to-amazon-ecr
            # we only need this in the kaniko container
            kpod.env.append(
                k8s_client.V1EnvVar(name="AWS_SDK_LOAD_CONFIG", value="true")
            )

        else:
            aws_credentials_file_env_key = "AWS_SHARED_CREDENTIALS_FILE"
            aws_credentials_file_env_value = "/tmp/aws/credentials"

            # set the credentials file location in the init container
            init_container_env[aws_credentials_file_env_key] = (
                aws_credentials_file_env_value
            )

            # set the kaniko container AWS credentials location to the mount's path
            kpod.env.append(
                k8s_client.V1EnvVar(
                    name=aws_credentials_file_env_key,
                    value=aws_credentials_file_env_value,
                )
            )
            # mount the AWS credentials secret
            kpod.mount_secret(
                mlrun.mlconf.httpdb.builder.docker_registry_secret,
                path="/tmp/aws",
            )

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
        builder_env = builder_env or []
        project_secrets = project_secrets or []

        # Utilizing plain values as they were explicitly compiled by the user
        for env in builder_env:
            args.extend(["--build-arg", f"{env.name}={env.value}"])

        # Utilizing '$' ensures that the value is not in plain text but rather
        # read from the injected environment variables
        for secret in project_secrets:
            args.extend(["--build-arg", f"{secret.name}=${secret.name}"])

        # Combine all the arguments into the Dockerfile
        args = builder_utils.validate_and_merge_args_with_extra_args(args, extra_args)

        return args
