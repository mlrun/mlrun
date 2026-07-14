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
from base64 import b64decode
from os import path
from urllib.parse import urlparse

import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.model
import mlrun.runtimes
import mlrun.utils
from mlrun.config import config

import framework.utils.singletons.k8s
from services.api.utils.builder import base, kaniko
from services.api.utils.builder.base import (
    _generate_builder_env,
    _resolve_build_requirements,
    _resolve_function_image_name,
    _resolve_function_image_secret,
    _validate_extra_args,
    add_mlrun_to_requirements,
    is_mlrun_image,
    resolve_and_enrich_image_target,
    resolve_image_target,
    resolve_mlrun_install_command_version,
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
    force_build=None,
):
    runtime_spec = runtime.spec if runtime else None
    runtime_builder_env = runtime_spec.build.builder_env or {}

    project_default_function_node_selector = {}
    if runtime and runtime._get_db():
        project_obj = runtime._get_db().get_project(runtime.metadata.project)
        if project_obj:
            project_default_function_node_selector = (
                project_obj.spec.default_function_node_selector
            )

    extra_args = extra_args or {}
    builder_env = builder_env or {}

    builder_env = runtime_builder_env | builder_env or {}
    # no need to enrich extra args because we get them from the build anyway
    _validate_extra_args(extra_args)

    image_target = resolve_image_target(image_target, registry)
    commands, requirements_list, requirements_path = _resolve_build_requirements(
        requirements, commands, with_mlrun, mlrun_version_specifier, client_version
    )

    if force_build:
        mlrun.utils.logger.info("Forcefully building image")
    elif not inline_code and not source and not commands and not requirements:
        mlrun.utils.logger.info("Skipping build, nothing to add")
        return "skipped"

    context = "/context"
    to_mount = False
    is_v3io_source, is_http_source = False, False
    if source:
        is_v3io_source = source.startswith("v3io://") or source.startswith("v3ios://")
        is_http_source = source.startswith("http")

    access_key = builder_env.get(
        "V3IO_ACCESS_KEY", auth_info.data_session or auth_info.access_key
    )
    username = builder_env.get("V3IO_USERNAME", auth_info.username)

    builder_env_list, project_secrets = _generate_builder_env(project, builder_env)

    parsed_url = urlparse(source)
    source_to_copy = None
    source_dir_to_mount = None
    needs_source_fetch_init_container = False
    if inline_code or runtime_spec.build.load_source_on_run or not source:
        context = "/empty"

    # http is not officially supported by kaniko's context so we handle it explicitly
    elif is_http_source:
        source_to_copy = source

    # source is in a scheme kaniko cannot resolve; fetch in a dedicated init container
    elif source and kaniko._needs_source_fetch_init_container(source):
        kaniko._validate_source_fetch_archive(source)
        context = "/empty"
        source_to_copy = f"./{kaniko._FETCHED_SOURCE_SUBDIR}"
        needs_source_fetch_init_container = True

    # source is remote (kaniko-native)
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
            source_dir_to_mount = path.normpath(source_dir_to_mount)

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
        from framework.api.utils import ensure_function_security_context

        ensure_function_security_context(runtime, auth_info)
        user_unix_id = runtime.spec.security_context.run_as_user
        enriched_group_id = runtime.spec.security_context.run_as_group

    source_code_target_dir = runtime.spec.build.source_code_target_dir
    if source_to_copy and (
        not source_code_target_dir or not os.path.isabs(source_code_target_dir)
    ):
        relative_workdir = source_code_target_dir or ""
        relative_workdir = relative_workdir.removeprefix("./")

        runtime.spec.build.source_code_target_dir = path.join(
            "/home/mlrun_code", relative_workdir
        )

    dock = base.make_dockerfile(
        base_image,
        commands,
        source=source_to_copy,
        requirements_path=requirements_path,
        extra=extra,
        user_unix_id=user_unix_id,
        enriched_group_id=enriched_group_id,
        target_dir=runtime.spec.build.source_code_target_dir,
        builder_env=builder_env_list,
        project_secrets=project_secrets,
        extra_args=extra_args,
    )

    kpod = kaniko.make_kaniko_pod(
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
        builder_env=builder_env_list,
        project_secrets=project_secrets,
        runtime_spec=runtime_spec,
        registry=registry,
        extra_args=extra_args,
        extra_labels={
            mlrun_constants.MLRunInternalLabels.name: name,
            mlrun_constants.MLRunInternalLabels.function: runtime.metadata.name,
            mlrun_constants.MLRunInternalLabels.tag: runtime.metadata.tag or "latest",
        },
        project_default_fucntion_node_selector=project_default_function_node_selector,
        auth_info=auth_info,
        source_to_fetch=source if needs_source_fetch_init_container else None,
    )

    if to_mount:
        kpod.mount_v3io(
            remote=source_dir_to_mount,
            mount_path="/context",
            access_key=access_key,
            user=username,
        )

    k8s = framework.utils.singletons.k8s.get_k8s_helper(silent=False)
    kpod.namespace = k8s.resolve_namespace(namespace)

    if interactive:
        return k8s.run_job(kpod)
    else:
        pod, ns = k8s.create_pod(kpod)
        mlrun.utils.logger.info(
            "Build started", pod=pod, namespace=ns, project=project, image=image_target
        )
        return f"build:{pod}"


def build_runtime(
    auth_info: mlrun.common.schemas.AuthInfo,
    runtime: mlrun.runtimes.BaseRuntime,
    with_mlrun=True,
    mlrun_version_specifier=None,
    skip_deployed=False,
    interactive=False,
    builder_env=None,
    client_version=None,
    client_python_version=None,
    force_build=False,
):
    build: mlrun.model.ImageBuilder = runtime.spec.build
    namespace = runtime.metadata.namespace
    project = runtime.metadata.project
    if skip_deployed and runtime.is_deployed():
        mlrun.utils.logger.info(
            "Skipping build, runtime is already deployed",
            runtime_name=runtime.metadata.name,
            project=project,
        )
        runtime.status.state = mlrun.common.schemas.FunctionState.ready
        return True

    base_image: str = build.base_image or runtime.spec.image
    if not base_image:
        base_image = mlrun.mlconf.function_defaults.image_by_kind.to_dict().get(
            runtime.kind, config.default_base_image
        )

    mlrun_image = False
    # If the base is one of mlrun images - set with_mlrun to False, so it won't be added later
    if base_image and is_mlrun_image(base_image):
        mlrun_image = True
        with_mlrun = False

    if force_build:
        mlrun.utils.logger.info("Forcefully building image")
    elif (
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

    build.image = _resolve_function_image_name(runtime, build.image)

    # config.httpdb.builder.docker_registry_secret
    build.secret = _resolve_function_image_secret(build.image, build.secret)
    runtime.status.state = ""

    inline = None  # noqa: F841
    if build.functionSourceCode:
        inline = b64decode(build.functionSourceCode).decode("utf-8")  # noqa: F841
    if not build.image:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "Build spec must have a target image, set build.image = <target image>"
        )
    name = mlrun.utils.normalize_name(f"mlrun-build-{runtime.metadata.name}")

    enriched_base_image = runtime.full_image_path(
        base_image, client_version, client_python_version
    )

    if mlrun_image and build.requirements:
        add_mlrun_to_requirements(build, enriched_base_image, mlrun_version_specifier)

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
        force_build=force_build,
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

    mlrun.utils.logger.info("Build completed", status=status)
    if status in ["failed", "error"]:
        runtime.status.state = mlrun.common.schemas.FunctionState.error
        return False

    local = "" if build.secret or build.image.startswith(".") else "."
    runtime.spec.image = local + build.image
    runtime.status.state = mlrun.common.schemas.FunctionState.ready
    return True
