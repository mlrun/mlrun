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
import os.path
import pathlib
import re
import tempfile
import textwrap
import typing
from base64 import b64decode, b64encode
from collections import defaultdict
from os import path
from urllib.parse import urlparse

from kubernetes import client

import mlrun.api.utils.singletons.k8s

# import mlrun.api.crud
import mlrun.common.constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.runtimes.utils
import mlrun.utils
from mlrun.config import config
from mlrun.utils.helpers import remove_image_protocol_prefix


def make_dockerfile(
    base_image: str,
    commands: list = None,
    source: str = None,
    requirements_path: str = None,
    workdir: str = "/mlrun",
    extra: str = "",
    user_unix_id: int = None,
    enriched_group_id: int = None,
    builder_env: typing.List[client.V1EnvVar] = None,
    project_secrets: dict = None,
    extra_args: str = "",
):
    dock = f"FROM {base_image}\n"

    builder_env = builder_env or []
    project_secrets = project_secrets or {}
    extra_args = _parse_extra_args_for_dockerfile(extra_args) if extra_args else []

    # combine a list of all args (including builder_env, project_secrets and extra_args)
    # to add in each of the Dockerfile stages.
    all_args = (
        [f"{env.name}" for env in builder_env]
        + [f"{key}" for key, val in project_secrets.items()]
        + [arg.split("=")[0] for arg in extra_args]
    )
    args = ""

    for arg in all_args:
        args += f"ARG {arg}\n"
    dock += args

    envs = ""
    for env in builder_env:
        envs += f"ENV {env.name}=${env.name}\n"

    for key, value in project_secrets.items():
        envs += f"ENV {key}=${key}\n"

    for env in extra_args:
        key = env.split("=")[0]
        envs += f"ENV {key}=${key}\n"
    dock += envs

    if config.is_pip_ca_configured():
        dock += f"COPY ./{pathlib.Path(config.httpdb.builder.pip_ca_path).name} {config.httpdb.builder.pip_ca_path}\n"
        dock += f"ARG PIP_CERT={config.httpdb.builder.pip_ca_path}\n"

    build_args = config.get_build_args()
    for build_arg_key, build_arg_value in build_args.items():
        dock += f"ARG {build_arg_key}={build_arg_value}\n"

    if source:
        args = args.rstrip("\n")
        envs = envs.rstrip("\n")
        dock += f"WORKDIR {workdir}\n"
        # 'ADD' command does not extract zip files - add extraction stage to the dockerfile
        if source.endswith(".zip"):
            source_dir = os.path.join(workdir, "source")
            stage1_lines = [
                f"FROM {base_image} AS extractor",
                f"{args}",
                f"{envs}",
                "RUN apt-get update -qqy && apt install --assume-yes unzip",
                f"RUN mkdir -p {source_dir}",
                f"COPY {source} {source_dir}",
                f"RUN cd {source_dir} && unzip {source} && rm {source}",
            ]
            stage1 = textwrap.dedent("\n".join(stage1_lines)).strip()
            dock = stage1 + "\n" + dock

            dock += f"COPY --from=extractor {source_dir}/ {workdir}\n"
        else:
            dock += f"ADD {source} {workdir}\n"

        if user_unix_id is not None and enriched_group_id is not None:
            dock += f"RUN chown -R {user_unix_id}:{enriched_group_id} {workdir}\n"

        dock += f"ENV PYTHONPATH {workdir}\n"
    if commands:
        dock += "".join([f"RUN {command}\n" for command in commands])
    if requirements_path:
        dock += (
            f"RUN echo 'Installing {requirements_path}...'; cat {requirements_path}\n"
        )
        dock += f"RUN python -m pip install -r {requirements_path}\n"
    if extra:
        dock += extra
    mlrun.utils.logger.debug("Resolved dockerfile", dockfile_contents=dock)

    # Add all args including the project secrets and extra_args at the top of the dockerfile to define their values
    args = ""
    for env in builder_env:
        args += f"ARG {env.name}={env.value}\n"

    for key, value in project_secrets.items():
        args += f"ARG {key}={value}\n"

    for arg in extra_args:
        args += f"ARG {arg}\n"

    dock = args + dock
    return dock


def make_kaniko_pod(
    project: str,
    context,
    dest,
    dockerfile=None,
    dockertext=None,
    inline_code=None,
    inline_path=None,
    requirements=None,
    requirements_path=None,
    secret_name=None,
    name="",
    verbose=False,
    builder_env=None,
    runtime_spec=None,
    registry=None,
    project_secrets=None,
    extra_args="",
):
    extra_runtime_spec = {}
    if not registry:

        # if registry was not given, infer it from the image destination
        registry = dest.partition("/")[0]

    # set kaniko's spec attributes from the runtime spec
    for attribute in get_kaniko_spec_attributes_from_runtime():
        attr_value = getattr(runtime_spec, attribute, None)
        if attribute == "service_account":
            from mlrun.api.api.utils import resolve_project_default_service_account

            (
                allowed_service_accounts,
                default_service_account,
            ) = resolve_project_default_service_account(project)
            if attr_value:
                runtime_spec.validate_service_account(allowed_service_accounts)
            else:
                attr_value = default_service_account

        if not attr_value:
            continue

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
        config.httpdb.builder.kaniko_image_fs_extraction_retries,
    ]
    for value, flag in [
        (config.httpdb.builder.insecure_pull_registry_mode, "--insecure-pull"),
        (config.httpdb.builder.insecure_push_registry_mode, "--insecure"),
    ]:
        if value == "disabled":
            continue
        if value == "enabled" or (value == "auto" and not secret_name):
            args.append(flag)
    if verbose:
        args += ["--verbosity", "debug"]

    builder_env = builder_env or []
    project_secrets = project_secrets or {}

    for env in builder_env:
        args += ["--build-arg", f"{env.name}={env.value}"]

    for key, value in project_secrets.items():
        args += ["--build-arg", f"{key}={value}"]

    # add extra_args to args
    args = _validate_and_merge_args_with_extra_args(args, extra_args)

    # While requests mainly affect scheduling, setting a limit may prevent Kaniko
    # from finishing successfully (destructive), since we're not allowing to override the default
    # specifically for the Kaniko pod, we're setting only the requests
    # we cannot specify gpu requests without specifying gpu limits, so we set requests without gpu field
    default_requests = config.get_default_function_pod_requirement_resources(
        "requests", with_gpu=False
    )
    resources = {
        "requests": mlrun.runtimes.utils.generate_resources(
            mem=default_requests.get("memory"), cpu=default_requests.get("cpu")
        )
    }

    kpod = mlrun.api.utils.singletons.k8s.BasePod(
        name or "mlrun-build",
        config.httpdb.builder.kaniko_image,
        args=args,
        kind="build",
        project=project,
        default_pod_spec_attributes=extra_runtime_spec,
        resources=resources,
    )
    kpod.env = builder_env

    if config.is_pip_ca_configured():
        items = [
            {
                "key": config.httpdb.builder.pip_ca_secret_key,
                "path": pathlib.Path(config.httpdb.builder.pip_ca_path).name,
            }
        ]
        kpod.mount_secret(
            config.httpdb.builder.pip_ca_secret_name,
            str(
                pathlib.Path(context)
                / pathlib.Path(config.httpdb.builder.pip_ca_path).name
            ),
            items=items,
            # using sub_path so file will be mounted inside kaniko pod as regular file and not symlink (if it's symlink
            # it's then not working inside the job image itself)
            sub_path=pathlib.Path(config.httpdb.builder.pip_ca_path).name,
        )

    if dockertext or inline_code or requirements:
        kpod.mount_empty()
        commands = []
        env = {}
        if dockertext:
            # set and encode docker content to the DOCKERFILE environment variable in the kaniko pod
            env["DOCKERFILE"] = b64encode(dockertext.encode("utf-8")).decode("utf-8")
            # dump dockerfile content and decode to Dockerfile destination
            commands.append("echo ${DOCKERFILE} | base64 -d > /empty/Dockerfile")
        if inline_code:
            name = inline_path or "main.py"
            env["CODE"] = b64encode(inline_code.encode("utf-8")).decode("utf-8")
            commands.append("echo ${CODE} | base64 -d > /empty/" + name)
        if requirements:
            # set and encode requirements to the REQUIREMENTS environment variable in the kaniko pod
            env["REQUIREMENTS"] = b64encode(
                "\n".join(requirements).encode("utf-8")
            ).decode("utf-8")
            # dump requirement content and decode to the requirement.txt destination
            commands.append(
                "echo ${REQUIREMENTS}" + " | " + f"base64 -d > {requirements_path}"
            )

        kpod.append_init_container(
            config.httpdb.builder.kaniko_init_container_image,
            args=["sh", "-c", "; ".join(commands)],
            env=env,
            name="create-dockerfile",
        )

    # when using ECR we need init container to create the image repository
    # example URL: <aws_account_id>.dkr.ecr.<region>.amazonaws.com
    if ".ecr." in registry and ".amazonaws.com" in registry:
        end = dest.find(":")
        if end == -1:
            end = len(dest)
        repo = dest[dest.find("/") + 1 : end]

        # if no secret is given, assume ec2 instance has attached role which provides read/write access to ECR
        assume_instance_role = not config.httpdb.builder.docker_registry_secret
        configure_kaniko_ecr_init_container(kpod, registry, repo, assume_instance_role)

        # project secret might conflict with the attached instance role
        # ensure "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY" have no values or else kaniko will fail
        # due to credentials conflict / lack of permission on given credentials
        if assume_instance_role:
            kpod.pod.spec.containers[0].env.extend(
                [
                    client.V1EnvVar(name="AWS_ACCESS_KEY_ID", value=""),
                    client.V1EnvVar(name="AWS_SECRET_ACCESS_KEY", value=""),
                ]
            )

    # mount regular docker config secret
    elif secret_name:
        items = [{"key": ".dockerconfigjson", "path": "config.json"}]
        kpod.mount_secret(secret_name, "/kaniko/.docker", items=items)

    return kpod


def configure_kaniko_ecr_init_container(
    kpod, registry, repo, assume_instance_role=True
):
    region = registry.split(".")[3]

    # fail silently in order to ignore "repository already exists" errors
    # if any other error occurs - kaniko will fail similarly
    command = (
        f"aws ecr create-repository --region {region} --repository-name {repo} || true"
        + f" && aws ecr create-repository --region {region} --repository-name {repo}/cache || true"
    )
    init_container_env = {}

    if assume_instance_role:

        # assume instance role has permissions to register and store a container image
        # https://github.com/GoogleContainerTools/kaniko#pushing-to-amazon-ecr
        # we only need this in the kaniko container
        kpod.env.append(client.V1EnvVar(name="AWS_SDK_LOAD_CONFIG", value="true"))

    else:
        aws_credentials_file_env_key = "AWS_SHARED_CREDENTIALS_FILE"
        aws_credentials_file_env_value = "/tmp/credentials"

        # set the credentials file location in the init container
        init_container_env[
            aws_credentials_file_env_key
        ] = aws_credentials_file_env_value

        # set the kaniko container AWS credentials location to the mount's path
        kpod.env.append(
            client.V1EnvVar(
                name=aws_credentials_file_env_key, value=aws_credentials_file_env_value
            )
        )
        # mount the AWS credentials secret
        kpod.mount_secret(
            config.httpdb.builder.docker_registry_secret,
            path="/tmp",
        )

    kpod.append_init_container(
        config.httpdb.builder.kaniko_aws_cli_image,
        command=["/bin/sh"],
        args=["-c", command],
        env=init_container_env,
        name="create-repos",
    )


def build_image(
    auth_info: mlrun.common.schemas.AuthInfo,
    project: str,
    image_target,
    commands=None,
    source="",
    base_image=None,
    requirements=None,
    inline_code=None,
    inline_path=None,
    secret_name=None,
    namespace=None,
    with_mlrun=True,
    mlrun_version_specifier=None,
    registry=None,
    interactive=True,
    name="",
    extra=None,
    verbose=False,
    builder_env=None,
    client_version=None,
    runtime=None,
    extra_args=None,
):
    runtime_spec = runtime.spec if runtime else None
    builder_env = builder_env or {}
    image_target, secret_name = resolve_image_target_and_registry_secret(
        image_target, registry, secret_name
    )

    commands, requirements_list, requirements_path = _resolve_build_requirements(
        requirements, commands, with_mlrun, mlrun_version_specifier, client_version
    )

    if not inline_code and not source and not commands and not requirements:
        mlrun.utils.logger.info("skipping build, nothing to add")
        return "skipped"

    context = "/context"
    to_mount = False
    is_v3io_source = False
    if source:
        is_v3io_source = source.startswith("v3io://") or source.startswith("v3ios://")

    access_key = builder_env.get(
        "V3IO_ACCESS_KEY", auth_info.data_session or auth_info.access_key
    )
    username = builder_env.get("V3IO_USERNAME", auth_info.username)

    builder_env = _generate_builder_env(project, builder_env)

    parsed_url = urlparse(source)
    source_to_copy = None
    source_dir_to_mount = None
    if inline_code or runtime_spec.build.load_source_on_run or not source:
        context = "/empty"

    # source is remote
    elif source and "://" in source and not is_v3io_source:
        if source.startswith("git://"):
            # if the user provided branch (w/o refs/..) we add the "refs/.."
            fragment = parsed_url.fragment or ""
            if not fragment.startswith("refs/"):
                source = source.replace("#" + fragment, f"#refs/heads/{fragment}")

        # set remote source as kaniko's build context and copy it
        context = source
        source_to_copy = "."

    # source is local / v3io
    else:
        if is_v3io_source:
            source = parsed_url.path
            to_mount = True
            source_dir_to_mount, source_to_copy = path.split(source)

        # source is a path without a scheme, we allow to copy absolute paths assuming they are valid paths
        # in the image, however, it is recommended to use `workdir` instead in such cases
        # which is set during runtime (mlrun.runtimes.local.LocalRuntime._pre_run).
        # relative paths are not supported at build time
        # "." and "./" are considered as 'project context'
        # TODO: enrich with project context if pulling on build time
        elif path.isabs(source):
            source_to_copy = source

        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Load of relative source ({source}) is not supported at build time "
                "see 'mlrun.runtimes.kubejob.KubejobRuntime.with_source_archive' or "
                "'mlrun.projects.project.MlrunProject.set_source' for more details"
            )

    user_unix_id = None
    enriched_group_id = None
    if (
        mlrun.mlconf.function.spec.security_context.enrichment_mode
        != mlrun.common.schemas.SecurityContextEnrichmentModes.disabled.value
    ):
        from mlrun.api.api.utils import ensure_function_security_context

        ensure_function_security_context(runtime, auth_info)
        user_unix_id = runtime.spec.security_context.run_as_user
        enriched_group_id = runtime.spec.security_context.run_as_group

    if source_to_copy and (
        not runtime.spec.clone_target_dir
        or not os.path.isabs(runtime.spec.clone_target_dir)
    ):
        # use a temp dir for permissions and set it as the workdir
        tmpdir = tempfile.mkdtemp()
        relative_workdir = runtime.spec.clone_target_dir or ""
        if relative_workdir.startswith("./"):
            # TODO: use 'removeprefix' when we drop python 3.7 support
            # relative_workdir.removeprefix("./")
            relative_workdir = relative_workdir[2:]

        runtime.spec.clone_target_dir = path.join(tmpdir, "mlrun", relative_workdir)

    from mlrun.api.crud import Secrets

    provider = mlrun.common.schemas.SecretProviderName.kubernetes
    project_secrets = (
        Secrets().list_project_secrets(project=project, provider=provider).secrets
    )
    validate_extra_args(extra_args)

    dock = make_dockerfile(
        base_image,
        commands,
        source=source_to_copy,
        requirements_path=requirements_path,
        extra=extra,
        user_unix_id=user_unix_id,
        enriched_group_id=enriched_group_id,
        workdir=runtime.spec.clone_target_dir,
        builder_env=builder_env,
        project_secrets=project_secrets,
        extra_args=extra_args,
    )

    kpod = make_kaniko_pod(
        project,
        context,
        image_target,
        dockertext=dock,
        inline_code=inline_code,
        inline_path=inline_path,
        requirements=requirements_list,
        requirements_path=requirements_path,
        secret_name=secret_name,
        name=name,
        verbose=verbose,
        builder_env=builder_env,
        runtime_spec=runtime_spec,
        registry=registry,
        project_secrets=project_secrets,
        extra_args=extra_args,
    )

    if to_mount:
        kpod.mount_v3io(
            remote=source_dir_to_mount,
            mount_path="/context",
            access_key=access_key,
            user=username,
        )

    k8s = mlrun.api.utils.singletons.k8s.get_k8s_helper(silent=False)
    kpod.namespace = k8s.resolve_namespace(namespace)

    if interactive:
        return k8s.run_job(kpod)
    else:
        pod, ns = k8s.create_pod(kpod)
        mlrun.utils.logger.info(
            "Build started", pod=pod, namespace=ns, project=project, image=image_target
        )
        return f"build:{pod}"


def get_kaniko_spec_attributes_from_runtime():
    """get the names of Kaniko spec attributes that are defined for runtime but should also be applied to kaniko"""
    return [
        "node_name",
        "node_selector",
        "affinity",
        "tolerations",
        "priority_class_name",
        "service_account",
    ]


def resolve_mlrun_install_command_version(
    mlrun_version_specifier=None, client_version=None, commands=None
):
    commands = commands or []
    install_mlrun_regex = re.compile(r".*pip install .*mlrun.*")
    for command in commands:
        if install_mlrun_regex.match(command):
            return None

    unstable_versions = ["unstable", "0.0.0+unstable"]
    unstable_mlrun_version_specifier = (
        f"{config.package_path}[complete] @ git+"
        f"https://github.com/mlrun/mlrun@development"
    )
    if not mlrun_version_specifier:
        if config.httpdb.builder.mlrun_version_specifier:
            mlrun_version_specifier = config.httpdb.builder.mlrun_version_specifier
        elif client_version:
            if client_version not in unstable_versions:
                mlrun_version_specifier = (
                    f"{config.package_path}[complete]=={client_version}"
                )
            else:
                mlrun_version_specifier = unstable_mlrun_version_specifier
        elif config.version in unstable_versions:
            mlrun_version_specifier = unstable_mlrun_version_specifier
        else:
            mlrun_version_specifier = (
                f"{config.package_path}[complete]=={config.version}"
            )
    return mlrun_version_specifier


def resolve_upgrade_pip_command(commands=None):
    commands = commands or []
    pip_upgrade_regex = re.compile(r".*pip install --upgrade .*pip.*")
    for command in commands:
        if pip_upgrade_regex.match(command):
            return None

    return f"python -m pip install --upgrade pip{config.httpdb.builder.pip_version}"


def build_runtime(
    auth_info: mlrun.common.schemas.AuthInfo,
    runtime,
    with_mlrun=True,
    mlrun_version_specifier=None,
    skip_deployed=False,
    interactive=False,
    builder_env=None,
    client_version=None,
    client_python_version=None,
):
    build = runtime.spec.build
    namespace = runtime.metadata.namespace
    project = runtime.metadata.project
    if skip_deployed and runtime.is_deployed():
        runtime.status.state = mlrun.common.schemas.FunctionState.ready
        return True
    if build.base_image:
        mlrun_images = [
            "mlrun/mlrun",
            "mlrun/ml-base",
            "mlrun/ml-models",
            "mlrun/ml-models-gpu",
        ]
        # if the base is one of mlrun images - no need to install mlrun
        if any([image in build.base_image for image in mlrun_images]):
            with_mlrun = False
    if (
        not build.source
        and not build.commands
        and not build.requirements
        and not build.extra
        and not with_mlrun
    ):
        if not runtime.spec.image:
            if build.base_image:
                runtime.spec.image = build.base_image
            elif runtime.kind in mlrun.mlconf.function_defaults.image_by_kind.to_dict():
                runtime.spec.image = (
                    mlrun.mlconf.function_defaults.image_by_kind.to_dict()[runtime.kind]
                )
        if not runtime.spec.image:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "The deployment was not successful because no image was specified or there are missing build parameters"
                " (commands/source)"
            )
        runtime.status.state = mlrun.common.schemas.FunctionState.ready
        return True

    build.image = mlrun.runtimes.utils.resolve_function_image_name(runtime, build.image)
    runtime.status.state = ""

    inline = None  # noqa: F841
    if build.functionSourceCode:
        inline = b64decode(build.functionSourceCode).decode("utf-8")  # noqa: F841
    if not build.image:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "build spec must have a target image, set build.image = <target image>"
        )
    name = mlrun.utils.normalize_name(f"mlrun-build-{runtime.metadata.name}")

    base_image: str = (
        build.base_image or runtime.spec.image or config.default_base_image
    )
    enriched_base_image = mlrun.utils.enrich_image_url(
        base_image,
        client_version,
        client_python_version,
    )
    mlrun.utils.logger.info(
        "Building runtime image",
        base_image=enriched_base_image,
        image=build.image,
        project=project,
        name=name,
    )

    status = build_image(
        auth_info,
        project,
        image_target=build.image,
        base_image=enriched_base_image,
        commands=build.commands,
        requirements=build.requirements,
        namespace=namespace,
        source=build.source,
        secret_name=build.secret,
        interactive=interactive,
        name=name,
        with_mlrun=with_mlrun,
        mlrun_version_specifier=mlrun_version_specifier,
        extra=build.extra,
        extra_args=build.extra_args,
        verbose=runtime.verbose,
        builder_env=builder_env,
        client_version=client_version,
        runtime=runtime,
    )
    runtime.status.build_pod = None
    if status == "skipped":
        # using enriched base image for the runtime spec image, because this will be the image that the function will
        # run with
        runtime.spec.image = enriched_base_image
        runtime.status.state = mlrun.common.schemas.FunctionState.ready
        return True

    if status.startswith("build:"):
        runtime.status.state = mlrun.common.schemas.FunctionState.deploying
        runtime.status.build_pod = status[6:]
        # using the base_image, and not the enriched one so we won't have the client version in the image, useful for
        # exports and other cases where we don't want to have the client version in the image, but rather enriched on
        # API level
        runtime.spec.build.base_image = base_image
        return False

    mlrun.utils.logger.info(f"build completed with {status}")
    if status in ["failed", "error"]:
        runtime.status.state = mlrun.common.schemas.FunctionState.error
        return False

    local = "" if build.secret or build.image.startswith(".") else "."
    runtime.spec.image = local + build.image
    runtime.status.state = mlrun.common.schemas.FunctionState.ready
    return True


def resolve_image_target_and_registry_secret(
    image_target: str, registry: str = None, secret_name: str = None
) -> (str, str):
    if registry:
        return "/".join([registry, image_target]), secret_name

    # if dest starts with a dot, we add the configured registry to the start of the dest
    if image_target.startswith(
        mlrun.common.constants.IMAGE_NAME_ENRICH_REGISTRY_PREFIX
    ):

        # remove prefix from image name
        image_target = image_target[
            len(mlrun.common.constants.IMAGE_NAME_ENRICH_REGISTRY_PREFIX) :
        ]

        registry, repository = mlrun.utils.get_parsed_docker_registry()
        secret_name = secret_name or config.httpdb.builder.docker_registry_secret
        if not registry:
            raise ValueError(
                "Default docker registry is not defined, set "
                "MLRUN_HTTPDB__BUILDER__DOCKER_REGISTRY/MLRUN_HTTPDB__BUILDER__DOCKER_REGISTRY_SECRET env vars"
            )
        image_target_components = [registry, image_target]
        if repository and repository not in image_target:
            image_target_components = [registry, repository, image_target]

        return "/".join(image_target_components), secret_name

    image_target = remove_image_protocol_prefix(image_target)

    return image_target, secret_name


def _generate_builder_env(
    project: str, builder_env: typing.Dict
) -> typing.List[client.V1EnvVar]:
    k8s = mlrun.api.utils.singletons.k8s.get_k8s_helper(silent=False)
    secret_name = k8s.get_project_secret_name(project)
    existing_secret_keys = k8s.get_project_secret_keys(project, filter_internal=True)

    # generate env list from builder env and project secrets
    env = []
    for key in existing_secret_keys:
        if key not in builder_env:
            value_from = client.V1EnvVarSource(
                secret_key_ref=client.V1SecretKeySelector(name=secret_name, key=key)
            )
            env.append(client.V1EnvVar(name=key, value_from=value_from))
    for key, value in builder_env.items():
        env.append(client.V1EnvVar(name=key, value=value))
    return env


def _parse_extra_args_for_dockerfile(extra_args: str) -> list:
    build_arg_values = []
    is_build_arg = False

    for arg in extra_args.split():
        if arg == "--build-arg":
            is_build_arg = True
        elif arg.startswith("--"):
            is_build_arg = False
        elif is_build_arg:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=[^=]+$", arg):
                raise ValueError(f"Invalid --build-arg value: {arg}")
            build_arg_values.append(arg)
        else:
            is_build_arg = False

    return build_arg_values


def _resolve_build_requirements(
    requirements: typing.Union[typing.List, str],
    commands: typing.List,
    with_mlrun: bool,
    mlrun_version_specifier: typing.Optional[str],
    client_version: typing.Optional[str],
):
    """
    Resolve build requirements list, requirements path and commands.
    If mlrun requirement is needed, we add a pip upgrade command to the commands list (prerequisite).
    """
    requirements_path = "/empty/requirements.txt"
    if requirements and isinstance(requirements, list):
        requirements_list = requirements
    else:
        requirements_list = []
        requirements_path = requirements or requirements_path
    commands = commands or []

    if with_mlrun:
        # mlrun prerequisite - upgrade pip
        upgrade_pip_command = resolve_upgrade_pip_command(commands)
        if upgrade_pip_command:
            commands.append(upgrade_pip_command)

        mlrun_version = resolve_mlrun_install_command_version(
            mlrun_version_specifier, client_version, commands
        )

        # mlrun must be installed with other python requirements in the same pip command to avoid version conflicts
        if mlrun_version:
            requirements_list.insert(0, mlrun_version)

    if not requirements_list:
        # no requirements, we don't need a requirements file
        requirements_path = ""

    return commands, requirements_list, requirements_path


def _parse_extra_args(extra_args: str) -> dict:
    extra_args = extra_args.split()
    args = defaultdict(list)

    for arg in extra_args:
        if arg.startswith("--"):
            current_flag = arg
            args[current_flag]
        else:
            args[current_flag].append(arg)
    return args


def validate_extra_args(extra_args: str):
    """
     Validate extra_args string for Docker commands:
    - Ensure --build-arg is followed by a non-flag argument.
    - Validate all --build-arg values are in a valid format of 'KEY=VALUE' using allowed characters only.

    :raises ValueError: If the extra_args sequence is invalid or contains incorrectly formatted '--build-arg' values.
    """

    if not extra_args.startswith("--"):
        raise ValueError(
            "Invalid argument sequence. Value must be followed by a flag preceding it."
        )
    args = _parse_extra_args(extra_args)
    for arg, values in args.items():
        if arg == "--build-arg":
            if not values:
                raise ValueError(
                    "Invalid '--build-arg' usage. It must be followed by a non-flag argument."
                )
            invalid_build_arg_values = [
                val
                for val in values
                if not re.match(r"^[a-zA-Z0-9_]+=[a-zA-Z0-9_]+$", val)
            ]
            if invalid_build_arg_values:
                raise ValueError(
                    f"Invalid arguments format: '{','.join(invalid_build_arg_values)}'."
                    " Please make sure all arguments are in a valid format"
                )


def _validate_and_merge_args_with_extra_args(args: list, extra_args: str) -> list:
    """
    Validate and merge the given args and extra_args for Kaniko pod.

    :return: A merged list of strings containing the command-line arguments
             from 'args' and 'extra_args' in args format.

    :raises ValueError: If an arg in 'extra_args' is duplicated with different values then in the 'args'.
    """

    extra_args = _parse_extra_args(extra_args)
    # Create a set to store the keys from the --build-arg flags in args
    build_arg_keys = {
        key: value
        for arg in args
        if arg == "--build-arg"
        for key, value in [args[args.index(arg) + 1].split("=")]
    }

    # Create a new list to store the merged args and extra_args
    merged_args = args[:]

    # Iterate through extra_args and add flags and their values to the merged_args list
    for flag, values in extra_args.items():
        if flag == "--build-arg":
            for value in values:
                key, val = value.split("=")
                if key not in build_arg_keys:
                    merged_args.extend([flag, value])
                    build_arg_keys[key] = val
                else:
                    if build_arg_keys[key] != val:
                        raise ValueError(
                            f"Duplicate --build-arg '{key}' with different values"
                        )
        elif flag not in args:
            if not values:
                merged_args.append(flag)
            else:
                for val in values:
                    merged_args.extend([flag, val])

    return merged_args
