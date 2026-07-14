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
import textwrap
import typing
from collections import defaultdict

from kubernetes import client

import mlrun.common.constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.model
import mlrun.runtimes
import mlrun.runtimes.mounts
import mlrun.runtimes.pod
import mlrun.runtimes.utils
import mlrun.utils
from mlrun.config import config
from mlrun.utils.helpers import remove_image_protocol_prefix

import framework.utils.helpers
import framework.utils.singletons.k8s


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


def add_mlrun_to_requirements(build, enriched_base_image, mlrun_version_specifier=None):
    # Add mlrun to the requirements even though it is already installed because
    # we want pip to include mlrun constraints when installing other packages
    image_tag, has_py_package = framework.utils.helpers.extract_image_tag(
        enriched_base_image
    )
    if has_py_package or mlrun_version_specifier:
        installed_mlrun_version_command = resolve_mlrun_install_command_version(
            mlrun_version_specifier, client_version=image_tag
        )
        mlrun.utils.logger.debug(
            "Enriching build requirements with mlrun package",
            enriched_base_image=enriched_base_image,
            installed_mlrun_version_command=installed_mlrun_version_command,
            image_tag=image_tag,
            mlrun_version_specifier=mlrun_version_specifier,
        )
        build.requirements.insert(0, installed_mlrun_version_command)

    else:
        mlrun.utils.logger.warning(
            "Cannot resolve mlrun pypi version from base image, mlrun requirements may be overriden",
            base_image=enriched_base_image,
        )


def is_mlrun_image(base_image):
    mlrun_images = [
        "mlrun/mlrun",
        "mlrun/mlrun-gpu",
        "mlrun/mlrun-kfp",
    ]
    return any([image in base_image for image in mlrun_images])


def resolve_and_enrich_image_target(
    image_target: str,
    registry: str | None = None,
    client_version: str | None = None,
    client_python_version: str | None = None,
) -> str:
    image_target = resolve_image_target(image_target, registry)
    image_target = mlrun.utils.enrich_image_url(
        image_target, client_version, client_python_version
    )
    return image_target


def resolve_image_target(image_target: str, registry: str | None = None) -> str:
    if registry:
        return "/".join([registry, image_target])

    # if dest starts with a dot, we add the configured registry to the start of the dest
    if image_target.startswith(
        mlrun.common.constants.IMAGE_NAME_ENRICH_REGISTRY_PREFIX
    ):
        # remove prefix from image name
        image_target = image_target[
            len(mlrun.common.constants.IMAGE_NAME_ENRICH_REGISTRY_PREFIX) :
        ]

        registry, repository = mlrun.utils.get_parsed_docker_registry()
        if not registry:
            raise ValueError(
                "Default docker registry is not defined, set "
                "MLRUN_HTTPDB__BUILDER__DOCKER_REGISTRY/MLRUN_HTTPDB__BUILDER__DOCKER_REGISTRY_SECRET env vars"
            )
        image_target_components = [registry, image_target]
        if repository and repository not in image_target:
            image_target_components = [registry, repository, image_target]

        return "/".join(image_target_components)

    image_target = remove_image_protocol_prefix(image_target)
    return image_target


def make_dockerfile(
    base_image: str,
    commands: list | None = None,
    source: str | None = None,
    requirements_path: str | None = None,
    target_dir: str = "/mlrun",
    extra: str = "",
    user_unix_id: int | None = None,
    enriched_group_id: int | None = None,
    builder_env: list[client.V1EnvVar] | None = None,
    extra_args: str = "",
    project_secrets: list[client.V1EnvVar] | None = None,
):
    """
    Generates the content of a Dockerfile for building a container image.

    There is a single renderer while Kaniko is the only backend; when a second
    backend needs the source staging done differently, factor out the part that
    actually diverges then (rather than pre-splitting it now).

    :param base_image: The base image for the Dockerfile.
    :param commands: A list of shell commands to be included in the Dockerfile as RUN instructions.
    :param source: The path to the source code directory to be included in the Docker image.
    :param requirements_path: The path to the requirements file (e.g., requirements.txt) containing
                              the Python dependencies to be installed in the Docker image.
    :param target_dir: The directory to which source code will be copied. Default is "/mlrun".
    :param extra: Additional content to be appended to the generated Dockerfile.
    :param user_unix_id: The Unix user ID to be used in the Docker image for running processes.
                         This is useful for matching the user ID with the host environment
                         to avoid permission issues.
    :param enriched_group_id: The group ID to be used in the Docker image for running processes.
    :param builder_env: A list of Kubernetes V1EnvVar objects representing build-time arguments
                        to be set during the build process.
    :param extra_args:  A string containing additional builder arguments in the format of command-line options,
            e.g. extra_args="--skip-tls-verify --build-arg A=val"
    :param project_secrets: A list of Kubernetes V1EnvVar objects representing the project secrets,
            which will be used as build-time arguments in the Dockerfile.
    :return: The content of the Dockerfile as a string.
    """
    dock = f"FROM {base_image}\n"

    builder_env = builder_env or []
    project_secrets = project_secrets or []
    extra_args = _parse_extra_args_for_dockerfile(extra_args)

    # combine a list of all args (including builder_env, project_secrets and extra_args)
    # to add in each of the Dockerfile stages.
    all_args = []
    # Include all builder_env and extra_args as 'ARG arg_name',
    # where the value will be set by the user using the --build-arg flag.
    all_args.extend([env.name for env in builder_env])
    all_args.extend([arg for arg in extra_args])

    # Include project secrets as ARGs, formatted like 'ARG SECRET_NAME=$ARG_NAME',
    # to prevent direct inclusion of the secret as plain text within the Dockerfile.
    all_args.extend([f"{secret.name}=${secret.name}" for secret in project_secrets])

    # add all args to the dockerfile
    args = ""
    for arg in all_args:
        args += f"ARG {arg}\n"
    dock += args

    if config.is_pip_ca_configured():
        dock += f"COPY ./{pathlib.Path(config.httpdb.builder.pip_ca_path).name} {config.httpdb.builder.pip_ca_path}\n"
        dock += f"ARG PIP_CERT={config.httpdb.builder.pip_ca_path}\n"

    build_args = config.get_build_args()
    for build_arg_key, build_arg_value in build_args.items():
        dock += f"ARG {build_arg_key}={build_arg_value}\n"

    if source:
        args = args.rstrip("\n")
        # 'ADD' command does not extract zip files - add extraction stage to the dockerfile
        # it is up to base image to have unzip included in case source is zip
        if source.endswith(".zip"):
            source_dir = os.path.join(target_dir, "source")
            filename = os.path.basename(source)
            stage_lines = [
                f"FROM {base_image} AS extractor",
                args,
                f"RUN mkdir -p {source_dir}",
                f"ADD {source} {source_dir}",
                f"RUN cd {source_dir} && unzip {filename} && rm {filename}",
            ]
            stage = textwrap.dedent("\n".join(stage_lines)).strip()
            dock = stage + "\n" + dock

            dock += f"COPY --from=extractor {source_dir}/ {target_dir}\n"
        else:
            dock += f"ADD {source} {target_dir}\n"

        if user_unix_id is not None and enriched_group_id is not None:
            dock += f"RUN chown -R {user_unix_id}:{enriched_group_id} {target_dir}\n"

        dock += f"ENV PYTHONPATH {target_dir}\n"
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

    return dock


def _generate_builder_env(
    project: str, builder_env: dict
) -> (list[client.V1EnvVar], list[client.V1EnvVar]):
    k8s = framework.utils.singletons.k8s.get_k8s_helper(silent=False)
    secret_name = k8s.get_project_secret_name(project)
    existing_secret_keys = k8s.get_project_secret_keys(project, filter_internal=True)

    # generate env list from builder env and project secrets
    project_secrets = []
    for key in existing_secret_keys:
        if key not in builder_env:
            value_from = client.V1EnvVarSource(
                secret_key_ref=client.V1SecretKeySelector(name=secret_name, key=key)
            )
            project_secrets.append(client.V1EnvVar(name=key, value_from=value_from))
    env = []
    for key, value in builder_env.items():
        env.append(client.V1EnvVar(name=key, value=value))
    return env, project_secrets


def _parse_extra_args_for_dockerfile(extra_args: str) -> dict:
    if not extra_args:
        return {}

    build_arg_values = {}
    is_build_arg = False

    for arg in extra_args.split():
        if arg == "--build-arg":
            is_build_arg = True
        elif arg.startswith("--"):
            is_build_arg = False
        elif is_build_arg:
            # Ensure 'arg' is in a valid format: starts with a letter or underscore,
            # followed by alphanumerics and an equal sign
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=[^=]+$", arg):
                raise ValueError(f"Invalid --build-arg value: {arg}")
            key, val = arg.split("=")
            build_arg_values[key] = val
        else:
            is_build_arg = False

    return build_arg_values


def _resolve_build_requirements(
    requirements: typing.Union[list, str],
    commands: list,
    with_mlrun: bool,
    mlrun_version_specifier: str | None,
    client_version: str | None,
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
    """
    Parses a string of extra arguments into a dictionary format.

    :param extra_args:  A string containing additional builder arguments in the format of command-line options,
            e.g. extra_args="--skip-tls-verify --build-arg A=val"

    :return: A dictionary where each key corresponds to an option flag (e.g., "--option_name"),
             and the associated value is a list of values provided for that option.

    :example:
    >>> extra_args = "--option1 value1 --option2 value3 --option3 --option1 value2"
    >>> parsed_args = _parse_extra_args(extra_args)
    >>> print(parsed_args)
    {
        '--option1': ['value1', 'value2'],
        '--option2': ['value3'],
        '--option3': []
    }
    """
    if not extra_args:
        return {}
    extra_args = extra_args.split()
    args = defaultdict(list)

    current_flag = None
    for arg in extra_args:
        if arg.startswith("--"):
            current_flag = arg
            # explicitly set the key in the dictionary
            args.setdefault(current_flag, [])
        elif current_flag:
            args[current_flag].append(arg)

        # sanity, args should be validated by now
        else:
            raise ValueError(
                "Invalid argument sequence. Value must be followed by a flag preceding it."
            )
    return args


def _validate_extra_args(extra_args: str):
    """
     Validate extra_args string for Docker commands:
    - Ensure --build-arg is followed by a non-flag argument.
    - Validate all --build-arg values are in a valid format of 'KEY=VALUE' using allowed characters only.

    :raises ValueError: If the extra_args sequence is invalid or contains incorrectly formatted '--build-arg' values.
    """
    if not extra_args:
        return

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


def _resolve_function_image_name(function, image: str | None = None) -> str:
    project = function.metadata.project
    name = function.metadata.name
    tag = function.metadata.tag or "latest"
    if image:
        image_name_prefix = (
            mlrun.runtimes.utils.resolve_function_target_image_name_prefix(
                project, name
            )
        )
        registries_to_enforce_prefix = mlrun.runtimes.utils.resolve_function_target_image_registries_to_enforce_prefix()
        for registry in registries_to_enforce_prefix:
            if image.startswith(registry):
                prefix_with_registry = f"{registry}{image_name_prefix}"
                if not image.startswith(prefix_with_registry):
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"Configured registry enforces image name to start with this prefix: {image_name_prefix}"
                    )
        return image
    return _generate_function_image_name(project, name, tag)


def _generate_function_image_name(project: str, name: str, tag: str) -> str:
    _, repository = mlrun.utils.get_parsed_docker_registry()
    repository = mlrun.utils.helpers.get_docker_repository_or_default(repository)
    return mlrun.runtimes.utils.fill_function_image_name_template(
        mlrun.common.constants.IMAGE_NAME_ENRICH_REGISTRY_PREFIX,
        repository,
        project,
        name,
        tag,
    )


def _resolve_function_image_secret(
    resolved_target_image: str, secret: str | None = None
) -> str:
    if not secret:
        parsed_registry, _ = mlrun.utils.get_parsed_docker_registry()

        # populate default secret if target image prefix equals to either the implicit or explicit default registry
        if (
            parsed_registry and resolved_target_image.startswith(parsed_registry)
        ) or resolved_target_image.startswith(
            mlrun.common.constants.IMAGE_NAME_ENRICH_REGISTRY_PREFIX
        ):
            secret = config.httpdb.builder.docker_registry_secret
    return secret
