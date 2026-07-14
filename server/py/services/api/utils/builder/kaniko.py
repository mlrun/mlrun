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
from base64 import b64encode
from urllib.parse import urlparse

from kubernetes import client

import mlrun.common.constants
import mlrun.common.constants as mlrun_constants
import mlrun.common.schemas
import mlrun.errors
import mlrun.model
import mlrun.runtimes
import mlrun.runtimes.mounts
import mlrun.runtimes.pod
import mlrun.runtimes.utils
import mlrun.utils
from mlrun.config import config
from mlrun.k8s_utils import enrich_preemption_mode

import framework.utils.helpers
import framework.utils.singletons.k8s
from services.api.utils.builder import base

# mlrun datastore schemes routed through the fetch-source init container instead of being
# handed to kaniko as --context: schemes kaniko cannot resolve (az, wasb, wasbs, ds, oss),
# plus gs/gcs - kaniko resolves gs:// natively but GCP wants file-based creds its
# storage-blind container never gets, so we let `mlrun load-source` fetch them instead.
# excludes s3/http(s) (kaniko-native) and v3io (igz FUSE-mount).
# aligned with `mlrun.datastore.datastore.schema_to_store`.
_FETCH_SUPPORTED_SCHEMES = frozenset({"az", "wasb", "wasbs", "ds", "oss", "gs", "gcs"})

# matches what ``mlrun load-source`` extracts.
# TODO: support .tgz
_FETCHABLE_ARCHIVE_EXTENSIONS = (".tar.gz", ".zip")

_FETCHED_SOURCE_SUBDIR = "source"

_DEFAULT_SOURCE_FETCH_IMAGE = "mlrun/mlrun"


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
    extra_args="",
    extra_labels=None,
    project_secrets=None,
    project_default_fucntion_node_selector=None,
    auth_info: mlrun.common.schemas.AuthInfo = None,
    *,
    source_to_fetch: str | None = None,
):
    extra_runtime_spec = {}
    if not registry:
        # if registry was not given, infer it from the image destination
        registry = dest.partition("/")[0]

    # set kaniko's spec attributes from the runtime spec
    for attribute, handler in get_kaniko_spec_attributes_from_runtime(
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
        config.httpdb.builder.kaniko_image_fs_extraction_retries,
        "--push-retry",
        config.httpdb.builder.kaniko_image_push_retry,
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

    args = _add_kaniko_args_with_all_build_args(
        args, builder_env, project_secrets, extra_args
    )

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
    # Some cloud providers add a toleration when a GPU limit is set.
    # If the Kaniko pod inherits a GPU-related node selector from the function
    # but lacks a GPU limit, it may get stuck in a pending state due to unsatisfiable scheduling.
    # Setting GPU limits to zero ensures tolerations are applied while preventing GPU allocation.
    if runtime_spec:
        gpu_resources = mlrun.utils.get_enriched_gpu_limits(
            runtime_spec.resources.get("limits", {})
        )
        if gpu_resources:
            resources["limits"] = gpu_resources

    # apply the configured builder pod labels (e.g. the azure.workload.identity/use label that lets the
    # Azure workload-identity webhook inject ACR push credentials into the builder pod).
    # these platform-level labels are the lowest-precedence layer: mlrun's own internal labels
    # (mlrun/class, mlrun/project, etc., set by BasePod and the call site) must never be clobbered by them.
    configured_pod_labels = {
        key: value
        for key, value in config.get_builder_pod_labels().items()
        if key not in mlrun_constants.MLRunInternalLabels.all()
    }
    extra_labels = mlrun.utils.helpers.merge_dicts_with_precedence(
        configured_pod_labels,
        extra_labels or {},
    )

    kpod = framework.utils.singletons.k8s.BasePod(
        name or "mlrun-build",
        config.httpdb.builder.kaniko_image,
        args=args,
        kind="build",
        project=project,
        default_pod_spec_attributes=extra_runtime_spec,
        resources=resources,
        labels=extra_labels,
    )
    envs = (builder_env or []) + (project_secrets or [])
    kpod.env = envs or None

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
            requirements_file_content = "{}\n".format("\n".join(requirements))
            env["REQUIREMENTS"] = b64encode(
                requirements_file_content.encode("utf-8")
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
    if mlrun.utils.helpers.is_ecr_url(registry):
        end = dest.find(":")
        if end == -1:
            end = len(dest)
        repo = dest[dest.find("/") + 1 : end]

        configure_kaniko_ecr_env_and_init_container(kpod, registry, repo)

    # mount regular docker config secret
    elif secret_name:
        items = [{"key": ".dockerconfigjson", "path": "config.json"}]
        kpod.mount_secret(secret_name, "/kaniko/.docker", items=items)

    if source_to_fetch:
        _append_source_fetch_init_container(
            kpod=kpod,
            source=source_to_fetch,
            builder_env_list=builder_env,
            project_secrets=project_secrets,
        )

    return kpod


def configure_kaniko_ecr_env_and_init_container(kpod, registry, repo):
    # if no secret is given, assume ec2 instance has attached role which provides read/write access to ECR
    assume_instance_role = not config.httpdb.builder.docker_registry_secret
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
    kpod.env = kpod.env = [
        env_var
        for env_var in kpod.env
        if env_var.name not in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    ]

    if assume_instance_role:
        # assume instance role has permissions to register and store a container image
        # https://github.com/GoogleContainerTools/kaniko#pushing-to-amazon-ecr
        # we only need this in the kaniko container
        kpod.env.append(client.V1EnvVar(name="AWS_SDK_LOAD_CONFIG", value="true"))

    else:
        aws_credentials_file_env_key = "AWS_SHARED_CREDENTIALS_FILE"
        aws_credentials_file_env_value = "/tmp/aws/credentials"

        # set the credentials file location in the init container
        init_container_env[aws_credentials_file_env_key] = (
            aws_credentials_file_env_value
        )

        # set the kaniko container AWS credentials location to the mount's path
        kpod.env.append(
            client.V1EnvVar(
                name=aws_credentials_file_env_key, value=aws_credentials_file_env_value
            )
        )
        # mount the AWS credentials secret
        kpod.mount_secret(
            config.httpdb.builder.docker_registry_secret,
            path="/tmp/aws",
        )

    kpod.append_init_container(
        config.httpdb.builder.kaniko_aws_cli_image,
        command=["/bin/sh"],
        args=["-c", command],
        env=init_container_env,
        name="create-repos",
    )


def get_kaniko_spec_attributes_from_runtime(
    project,
    runtime_spec,
    project_default_fucntion_node_selector,
    auth_info: mlrun.common.schemas.AuthInfo = None,
):
    """Get the names of Kaniko spec attributes that are defined for runtime but should also be applied to Kaniko."""
    # preemption mode scheduling constraints cache
    _preemption_enrichment_result = {}

    def service_account_handler(attr_value):
        from framework.api.utils import resolve_project_service_account_details

        (
            allowed_service_accounts,
            forbidden_service_accounts,
            default_service_account,
        ) = resolve_project_service_account_details(project, auth_info=auth_info)
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
            values = enrich_preemption_mode(
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


def _needs_source_fetch_init_container(source: str) -> bool:
    return urlparse(source).scheme in _FETCH_SUPPORTED_SCHEMES


def _validate_source_fetch_archive(source: str) -> None:
    # match ``load_source_code``'s case-sensitive extension check, so uppercased
    # variants (.TAR.GZ, .ZIP) are rejected at the API boundary instead of slipping
    # through and failing inside the init container with a worse error.
    if not source.endswith(_FETCHABLE_ARCHIVE_EXTENSIONS):
        scheme = urlparse(source).scheme
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Source {source} uses scheme '{scheme}://' which is not natively "
            "supported as a kaniko build context. Provide the source as an "
            f"archive ending in one of: {', '.join(_FETCHABLE_ARCHIVE_EXTENSIONS)}"
        )


def _append_source_fetch_init_container(
    kpod,
    source: str,
    builder_env_list: list,
    project_secrets: list,
) -> None:
    # Env precedence: builder_env_list > project_secrets > storage.auto_mount_params.
    # First-write wins so caller-supplied values are not overwritten by auto-mount defaults.
    image = config.httpdb.builder.kaniko_source_fetch_init_container_image
    if not image:
        image = mlrun.utils.enrich_image_url(_DEFAULT_SOURCE_FETCH_IMAGE)

    target_dir = f"/empty/{_FETCHED_SOURCE_SUBDIR}"
    args = ["-m", "mlrun", "load-source", source, "--target", target_dir]

    env_list = list(builder_env_list or []) + list(project_secrets or [])
    already_set = {env_var.name for env_var in env_list}
    for env_var in _resolve_storage_auto_mount_env():
        if env_var.name in already_set:
            continue
        env_list.append(env_var)
        already_set.add(env_var.name)

    mlrun.utils.logger.debug(
        "Adding source-fetch init container",
        image=image,
        source=source,
        target=target_dir,
    )
    kpod.append_init_container(
        image,
        command=["python"],
        args=args,
        env=env_list,
        name="fetch-source",
    )


def _resolve_storage_auto_mount_env() -> list:
    # Gate on env_style_modifiers so mount-style outputs (volumes/volume_mounts) are
    # not silently dropped. KubeResource.apply sanitizes spec.env to plain dicts, so
    # rebuild V1EnvVar for callers that rely on attribute access.
    auto_mount_type = mlrun.runtimes.pod.AutoMountType(
        mlrun.mlconf.storage.auto_mount_type
    )
    modifier = auto_mount_type.get_modifier()
    if (
        modifier is None
        or modifier not in mlrun.runtimes.pod.AutoMountType.env_style_modifiers()
    ):
        return []
    scratch = mlrun.runtimes.KubejobRuntime()
    scratch.try_auto_mount_based_on_config()
    return [
        client.V1EnvVar(
            name=env_var["name"],
            value=env_var.get("value"),
            value_from=env_var.get("valueFrom"),
        )
        if isinstance(env_var, dict)
        else env_var
        for env_var in scratch.spec.env or []
    ]


def _add_kaniko_args_with_all_build_args(
    args, builder_env, project_secrets, extra_args
):
    builder_env = builder_env or []
    project_secrets = project_secrets or []

    # Utilizing plain values as they were explicitly compiled by the user
    for env in builder_env:
        args.extend(["--build-arg", f"{env.name}={env.value}"])

    # Utilizing '$' ensures that the value is not in plain text but rather read from the injected environment variables
    for secret in project_secrets:
        args.extend(["--build-arg", f"{secret.name}=${secret.name}"])

    # Combine all the arguments into the Dockerfile
    args = _validate_and_merge_args_with_extra_args(args, extra_args)

    return args


def _validate_and_merge_args_with_extra_args(args: list, extra_args: str) -> list:
    """
    Validate and merge the given args and extra_args for Kaniko pod.

    :return: A merged list of strings containing the command-line arguments
             from 'args' and 'extra_args' in args format.

    :raises ValueError: If an arg in 'extra_args' is duplicated with different values then in the 'args'.
    """
    if not extra_args:
        return args
    extra_args = base._parse_extra_args(extra_args)
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
                    merged_args.extend([flag, f"{key}={val}"])
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
