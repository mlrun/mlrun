# Copyright 2018 Iguazio
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
import tarfile
import tempfile
from base64 import b64decode, b64encode
from os import path
from urllib.parse import urlparse

from kubernetes import client

import mlrun.api.schemas
import mlrun.errors
import mlrun.runtimes.utils

from .config import config
from .datastore import store_manager
from .k8s_utils import BasePod, get_k8s_helper
from .utils import enrich_image_url, get_parsed_docker_registry, logger, normalize_name

IMAGE_NAME_ENRICH_REGISTRY_PREFIX = "."


def make_dockerfile(
    base_image,
    commands=None,
    source=None,
    requirements=None,
    workdir="/mlrun",
    extra="",
    user_unix_id=None,
    enriched_group_id=None,
):
    dock = f"FROM {base_image}\n"

    if config.is_pip_ca_configured():
        dock += f"COPY ./{pathlib.Path(config.httpdb.builder.pip_ca_path).name} {config.httpdb.builder.pip_ca_path}\n"
        dock += f"ARG PIP_CERT={config.httpdb.builder.pip_ca_path}\n"

    build_args = config.get_build_args()
    for build_arg_key, build_arg_value in build_args.items():
        dock += f"ARG {build_arg_key}={build_arg_value}\n"

    if source:
        dock += f"RUN mkdir -p {workdir}\n"
        dock += f"WORKDIR {workdir}\n"
        # 'ADD' command does not extract zip files - add extraction stage to the dockerfile
        if source.endswith(".zip"):
            stage1 = f"""
            FROM {base_image} AS extractor
            RUN apt-get update -qqy && apt install --assume-yes unzip
            RUN mkdir -p /source
            COPY {source} /source
            RUN cd /source && unzip {source} && rm {source}
            """
            dock = stage1 + "\n" + dock

            dock += f"COPY --from=extractor /source/ {workdir}\n"
        else:
            dock += f"ADD {source} {workdir}\n"

        if user_unix_id is not None and enriched_group_id is not None:
            dock += f"RUN chown -R {user_unix_id}:{enriched_group_id} {workdir}\n"

        dock += f"ENV PYTHONPATH {workdir}\n"
    if requirements:
        dock += f"RUN python -m pip install -r {requirements}\n"
    if commands:
        dock += "".join([f"RUN {command}\n" for command in commands])
    if extra:
        dock += extra
    logger.debug("Resolved dockerfile", dockfile_contents=dock)
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
    secret_name=None,
    name="",
    verbose=False,
    builder_env=None,
    runtime_spec=None,
    registry=None,
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

    args = ["--dockerfile", dockerfile, "--context", context, "--destination", dest]
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
    kpod = BasePod(
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
            commands.append("echo ${DOCKERFILE} | base64 -d > /empty/Dockerfile")
            env["DOCKERFILE"] = b64encode(dockertext.encode("utf-8")).decode("utf-8")
        if inline_code:
            name = inline_path or "main.py"
            commands.append("echo ${CODE} | base64 -d > /empty/" + name)
            env["CODE"] = b64encode(inline_code.encode("utf-8")).decode("utf-8")
        if requirements:
            commands.append(
                "echo ${REQUIREMENTS} | base64 -d > /empty/requirements.txt"
            )
            env["REQUIREMENTS"] = b64encode(
                "\n".join(requirements).encode("utf-8")
            ).decode("utf-8")

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
        configure_kaniko_ecr_init_container(kpod, registry, repo)

    # mount regular docker config secret
    elif secret_name:
        items = [{"key": ".dockerconfigjson", "path": "config.json"}]
        kpod.mount_secret(secret_name, "/kaniko/.docker", items=items)

    return kpod


def configure_kaniko_ecr_init_container(kpod, registry, repo):
    region = registry.split(".")[3]

    # fail silently in order to ignore "repository already exists" errors
    # if any other error occurs - kaniko will fail similarly
    command = (
        f"aws ecr create-repository --region {region} --repository-name {repo} || true"
        + f" && aws ecr create-repository --region {region} --repository-name {repo}/cache || true"
    )
    init_container_env = {}

    if not config.httpdb.builder.docker_registry_secret:

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


def upload_tarball(source_dir, target, secrets=None):

    # will delete the temp file
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as temp_fh:
        with tarfile.open(mode="w:gz", fileobj=temp_fh) as tar:
            tar.add(source_dir, arcname="")
        stores = store_manager.set(secrets)
        datastore, subpath = stores.get_or_create_store(target)
        datastore.upload(subpath, temp_fh.name)


def build_image(
    auth_info: mlrun.api.schemas.AuthInfo,
    project: str,
    image_target,
    commands=None,
    source="",
    mounter="v3io",
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
):
    runtime_spec = runtime.spec if runtime else None
    builder_env = builder_env or {}
    image_target, secret_name = _resolve_image_target_and_registry_secret(
        image_target, registry, secret_name
    )

    if isinstance(requirements, list):
        requirements_list = requirements
        requirements_path = "requirements.txt"
        if source:
            raise ValueError("requirements list only works with inline code")
    else:
        requirements_list = None
        requirements_path = requirements

    if with_mlrun:
        commands = commands or []
        mlrun_command = resolve_mlrun_install_command(
            mlrun_version_specifier, client_version
        )
        if mlrun_command not in commands:
            commands.append(mlrun_command)

    if not inline_code and not source and not commands:
        logger.info("skipping build, nothing to add")
        return "skipped"

    context = "/context"
    to_mount = False
    v3io = (
        source.startswith("v3io://") or source.startswith("v3ios://")
        if source
        else None
    )
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

    elif source and "://" in source and not v3io:
        if source.startswith("git://"):
            # if the user provided branch (w/o refs/..) we add the "refs/.."
            fragment = parsed_url.fragment or ""
            if not fragment.startswith("refs/"):
                source = source.replace("#" + fragment, f"#refs/heads/{fragment}")

        # set remote source as kaniko's build context and copy it
        context = source
        source_to_copy = "."

    else:
        if v3io:
            source = parsed_url.path
            to_mount = True
            source_dir_to_mount, source_to_copy = path.split(source)
        else:
            source_to_copy = source

    user_unix_id = None
    enriched_group_id = None
    if (
        mlrun.mlconf.function.spec.security_context.enrichment_mode
        != mlrun.api.schemas.SecurityContextEnrichmentModes.disabled.value
    ):
        from mlrun.api.api.utils import ensure_function_security_context

        ensure_function_security_context(runtime, auth_info)
        user_unix_id = runtime.spec.security_context.run_as_user
        enriched_group_id = runtime.spec.security_context.run_as_group

    dock = make_dockerfile(
        base_image,
        commands,
        source=source_to_copy,
        requirements=requirements_path,
        extra=extra,
        user_unix_id=user_unix_id,
        enriched_group_id=enriched_group_id,
    )

    kpod = make_kaniko_pod(
        project,
        context,
        image_target,
        dockertext=dock,
        inline_code=inline_code,
        inline_path=inline_path,
        requirements=requirements_list,
        secret_name=secret_name,
        name=name,
        verbose=verbose,
        builder_env=builder_env,
        runtime_spec=runtime_spec,
        registry=registry,
    )

    if to_mount:
        kpod.mount_v3io(
            remote=source_dir_to_mount,
            mount_path="/context",
            access_key=access_key,
            user=username,
        )

    k8s = get_k8s_helper()
    kpod.namespace = k8s.resolve_namespace(namespace)

    if interactive:
        return k8s.run_job(kpod)
    else:
        pod, ns = k8s.create_pod(kpod)
        logger.info(f'started build, to watch build logs use "mlrun watch {pod} {ns}"')
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


def resolve_mlrun_install_command(mlrun_version_specifier=None, client_version=None):
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
    return f'python -m pip install "{mlrun_version_specifier}"'


def build_runtime(
    auth_info: mlrun.api.schemas.AuthInfo,
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
        runtime.status.state = mlrun.api.schemas.FunctionState.ready
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
    if not build.source and not build.commands and not build.extra and not with_mlrun:
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
        runtime.status.state = mlrun.api.schemas.FunctionState.ready
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
    logger.info(f"building image ({build.image})")

    name = normalize_name(f"mlrun-build-{runtime.metadata.name}")
    base_image: str = (
        build.base_image or runtime.spec.image or config.default_base_image
    )
    enriched_base_image = enrich_image_url(
        base_image,
        client_version,
        client_python_version,
    )

    status = build_image(
        auth_info,
        project,
        image_target=build.image,
        base_image=enriched_base_image,
        commands=build.commands,
        namespace=namespace,
        # inline_code=inline,
        source=build.source,
        secret_name=build.secret,
        interactive=interactive,
        name=name,
        with_mlrun=with_mlrun,
        mlrun_version_specifier=mlrun_version_specifier,
        extra=build.extra,
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
        runtime.status.state = mlrun.api.schemas.FunctionState.ready
        return True

    if status.startswith("build:"):
        runtime.status.state = mlrun.api.schemas.FunctionState.deploying
        runtime.status.build_pod = status[6:]
        # using the base_image, and not the enriched one so we won't have the client version in the image, useful for
        # exports and other cases where we don't want to have the client version in the image, but rather enriched on
        # API level
        runtime.spec.build.base_image = base_image
        return False

    logger.info(f"build completed with {status}")
    if status in ["failed", "error"]:
        runtime.status.state = mlrun.api.schemas.FunctionState.error
        return False

    local = "" if build.secret or build.image.startswith(".") else "."
    runtime.spec.image = local + build.image
    runtime.status.state = mlrun.api.schemas.FunctionState.ready
    return True


def _generate_builder_env(project, builder_env):
    k8s = get_k8s_helper()
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


def _resolve_image_target_and_registry_secret(
    image_target: str, registry: str = None, secret_name: str = None
) -> (str, str):
    if registry:
        return "/".join([registry, image_target]), secret_name

    # if dest starts with a dot, we add the configured registry to the start of the dest
    if image_target.startswith(IMAGE_NAME_ENRICH_REGISTRY_PREFIX):

        # remove prefix from image name
        image_target = image_target[len(IMAGE_NAME_ENRICH_REGISTRY_PREFIX) :]

        registry, repository = get_parsed_docker_registry()
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

    return image_target, secret_name
