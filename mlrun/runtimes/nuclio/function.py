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

import asyncio
import copy
import json
import typing
import warnings
from datetime import datetime
from time import sleep

import inflection
import nuclio
import nuclio.utils
import requests
import semver
from aiohttp.client import ClientSession
from kubernetes import client
from nuclio.deploy import find_dashboard_url, get_deploy_status
from nuclio.triggers import V3IOStreamTrigger

import mlrun.db
import mlrun.errors
import mlrun.k8s_utils
import mlrun.utils
import mlrun.utils.helpers
from mlrun.common.schemas import AuthInfo
from mlrun.config import config as mlconf
from mlrun.errors import err_to_str
from mlrun.lists import RunList
from mlrun.model import RunObject
from mlrun.platforms.iguazio import (
    parse_path,
    split_path,
)
from mlrun.runtimes.base import FunctionStatus, RunError
from mlrun.runtimes.mounts import VolumeMount, mount_v3io, v3io_cred
from mlrun.runtimes.pod import KubeResource, KubeResourceSpec
from mlrun.runtimes.utils import get_item_name, log_std
from mlrun.utils import get_in, logger, update_in
from mlrun_pipelines.common.ops import deploy_op

SENSITIVE_PATHS_IN_TRIGGER_CONFIG = {
    "password",
    "secret",
    "attributes/password",
    "attributes/accesskeyid",
    "attributes/secretaccesskey",
    "attributes/cacert",
    "attributes/accesskey",
    "attributes/accesscertificate",
    "attributes/sasl/password",
    "attributes/sasl/oauth/clientsecret",
}


def validate_nuclio_version_compatibility(*min_versions):
    """
    :param min_versions: Valid minimum version(s) required, assuming no 2 versions has equal major and minor.
    """
    return mlrun.utils.helpers.validate_component_version_compatibility(
        "nuclio", *min_versions
    )


def min_nuclio_versions(*versions):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if validate_nuclio_version_compatibility(*versions):
                return function(*args, **kwargs)

            if function.__name__ == "__init__":
                name = inflection.titleize(function.__qualname__.split(".")[0])
            else:
                name = function.__qualname__

            message = (
                f"'{name}' function requires Nuclio v{' or v'.join(versions)} or higher"
            )
            raise mlrun.errors.MLRunIncompatibleVersionError(message)

        return wrapper

    return decorator


class NuclioSpec(KubeResourceSpec):
    _dict_fields = KubeResourceSpec._dict_fields + [
        "min_replicas",
        "max_replicas",
        "config",
        "base_spec",
        "no_cache",
        "source",
        "function_kind",
        "readiness_timeout",
        "readiness_timeout_before_failure",
        "function_handler",
        "nuclio_runtime",
        "base_image_pull",
        "service_type",
        "add_templated_ingress_host_mode",
        "disable_default_http_trigger",
    ]

    def __init__(
        self,
        command=None,
        args=None,
        image=None,
        mode=None,
        entry_points=None,
        description=None,
        replicas=None,
        min_replicas=None,
        max_replicas=None,
        volumes=None,
        volume_mounts=None,
        env=None,
        resources=None,
        config=None,
        base_spec=None,
        no_cache=None,
        source=None,
        image_pull_policy=None,
        function_kind=None,
        build=None,
        service_account=None,
        readiness_timeout=None,
        readiness_timeout_before_failure=None,
        default_handler=None,
        node_name=None,
        node_selector=None,
        affinity=None,
        disable_auto_mount=False,
        priority_class_name=None,
        pythonpath=None,
        workdir=None,
        image_pull_secret=None,
        tolerations=None,
        preemption_mode=None,
        security_context=None,
        service_type=None,
        add_templated_ingress_host_mode=None,
        clone_target_dir=None,
        state_thresholds=None,
        disable_default_http_trigger=None,
    ):
        super().__init__(
            command=command,
            args=args,
            image=image,
            mode=mode,
            volumes=volumes,
            volume_mounts=volume_mounts,
            env=env,
            resources=resources,
            replicas=replicas,
            image_pull_policy=image_pull_policy,
            service_account=service_account,
            build=build,
            entry_points=entry_points,
            description=description,
            default_handler=default_handler,
            node_name=node_name,
            node_selector=node_selector,
            affinity=affinity,
            disable_auto_mount=disable_auto_mount,
            priority_class_name=priority_class_name,
            pythonpath=pythonpath,
            workdir=workdir,
            image_pull_secret=image_pull_secret,
            tolerations=tolerations,
            preemption_mode=preemption_mode,
            security_context=security_context,
            clone_target_dir=clone_target_dir,
            state_thresholds=state_thresholds,
        )

        self.base_spec = base_spec or {}
        self.function_kind = function_kind
        self.source = source or ""
        self.config = config or {}
        self.function_handler = None
        self.nuclio_runtime = None
        self.no_cache = no_cache
        self.readiness_timeout = readiness_timeout
        self.readiness_timeout_before_failure = readiness_timeout_before_failure
        self.service_type = service_type
        self.add_templated_ingress_host_mode = add_templated_ingress_host_mode
        self.state_thresholds = None  # not supported in nuclio

        self.min_replicas = min_replicas or 1
        self.max_replicas = max_replicas or 4

        self.disable_default_http_trigger = disable_default_http_trigger

        # When True it will set Nuclio spec.noBaseImagesPull to False (negative logic)
        # indicate that the base image should be pulled from the container registry (not cached)
        self.base_image_pull = False

    def generate_nuclio_volumes(self):
        nuclio_volumes = []
        volume_with_volume_mounts_names = set()
        for volume_mount in self._volume_mounts.values():
            volume_name = get_item_name(volume_mount, "name")
            if volume_name not in self._volumes:
                raise ValueError(
                    f"Found volume mount without a volume. name={volume_name}"
                )
            volume_with_volume_mounts_names.add(volume_name)
            nuclio_volumes.append(
                {"volume": self._volumes[volume_name], "volumeMount": volume_mount}
            )

        volumes_without_volume_mounts = (
            volume_with_volume_mounts_names.symmetric_difference(self._volumes.keys())
        )
        if volumes_without_volume_mounts:
            raise ValueError(
                f"Found volumes without volume mounts. names={volumes_without_volume_mounts}"
            )

        return nuclio_volumes


class NuclioStatus(FunctionStatus):
    def __init__(
        self,
        state=None,
        nuclio_name=None,
        address=None,
        internal_invocation_urls=None,
        external_invocation_urls=None,
        build_pod=None,
        container_image=None,
    ):
        super().__init__(state, build_pod)

        self.nuclio_name = nuclio_name

        # exists on nuclio >= 1.6.x
        # infers the function invocation urls
        self.internal_invocation_urls = internal_invocation_urls or []
        self.external_invocation_urls = external_invocation_urls or []

        # still exists for backwards compatability reasons.
        # on latest Nuclio (>= 1.6.x) versions, use external_invocation_urls / internal_invocation_urls instead
        self.address = address

        # the name of the image that was built and pushed to the registry, and used by the nuclio function
        self.container_image = container_image


class RemoteRuntime(KubeResource):
    kind = "remote"
    _is_nested = False
    _mock_server = None

    @property
    def spec(self) -> NuclioSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", NuclioSpec)

    @property
    def status(self) -> NuclioStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", NuclioStatus)

    def pre_deploy_validation(self):
        if self.metadata.tag:
            mlrun.utils.validate_tag_name(self.metadata.tag, "function.metadata.tag")

    def mask_sensitive_data_in_config(self):
        if not self.spec.config:
            return {}

        raw_config = copy.deepcopy(self.spec.config)

        for key, value in self.spec.config.items():
            if key.startswith("spec.triggers"):
                trigger_name = key.split(".")[-1]

                for path in SENSITIVE_PATHS_IN_TRIGGER_CONFIG:
                    # Handle nested keys
                    nested_keys = path.split("/")
                    target = value
                    for sub_key in nested_keys[:-1]:
                        target = target.get(sub_key, {})

                    last_key = nested_keys[-1]
                    if last_key in target:
                        sensitive_field = target[last_key]
                        if sensitive_field.startswith(
                            mlrun.model.Credentials.secret_reference_prefix
                        ):
                            # already masked
                            continue
                        target[last_key] = (
                            f"{mlrun.model.Credentials.secret_reference_prefix}/spec/triggers/{trigger_name}/{path}"
                        )

        return raw_config

    def set_config(self, key, value):
        self.spec.config[key] = value
        return self

    def with_annotations(self, annotations: dict):
        """set a key/value annotations for function"""

        self.spec.base_spec.setdefault("metadata", {})
        self.spec.base_spec["metadata"].setdefault("annotations", {})
        for key, value in annotations.items():
            self.spec.base_spec["metadata"]["annotations"][key] = str(value)

        return self

    def add_trigger(self, name, spec):
        """add a nuclio trigger object/dict

        :param name: trigger name
        :param spec: trigger object or dict
        """
        if hasattr(spec, "to_dict"):
            spec = spec.to_dict()

        self._validate_triggers(spec)

        spec["name"] = name
        self.spec.config[f"spec.triggers.{name}"] = spec
        return self

    def _validate_triggers(self, spec):
        # ML-7763 / NUC-233
        min_nuclio_version = "1.13.12"
        if (
            mlconf.nuclio_version
            and mlconf.nuclio_version != "unstable"
            and semver.VersionInfo.parse(mlconf.nuclio_version)
            < semver.VersionInfo.parse(min_nuclio_version)
        ):
            explicit_ack_enabled = False
            num_triggers = 0
            trigger_name = spec.get("name", "UNKNOWN")
            for key, config in [(f"spec.triggers.{trigger_name}", spec)] + list(
                self.spec.config.items()
            ):
                if key.startswith("spec.triggers."):
                    num_triggers += 1
                    explicit_ack_enabled = (
                        config.get("explicitAckMode", "disable") != "disable"
                    )

            if num_triggers > 1 and explicit_ack_enabled:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Multiple triggers cannot be used in conjunction with explicit ack. "
                    f"Please upgrade to nuclio {min_nuclio_version} or newer."
                )

    def with_source_archive(
        self,
        source,
        workdir=None,
        handler=None,
        runtime="",
    ):
        """Load nuclio function from remote source

        Note: remote source may require credentials, those can be stored in the project secrets or passed
        in the function.deploy() using the builder_env dict, see the required credentials per source:

        - v3io - "V3IO_ACCESS_KEY".
        - git - "GIT_USERNAME", "GIT_PASSWORD".
        - AWS S3 - "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY" or "AWS_SESSION_TOKEN".

        :param source: a full path to the nuclio function source (code entry) to load the function from
        :param handler: a path to the function's handler, including path inside archive/git repo
        :param workdir: working dir  relative to the archive root (e.g. 'subdir')
        :param runtime: (optional) the runtime of the function (defaults to mlrun.mlconf.default_nuclio_runtime)

        :Examples:

            git::

                fn.with_source_archive(
                    "git://github.com/org/repo#my-branch",
                    handler="main:handler",
                    workdir="path/inside/repo",
                )

            s3::

                fn.spec.nuclio_runtime = "golang"
                fn.with_source_archive(
                    "s3://my-bucket/path/in/bucket/my-functions-archive",
                    handler="my_func:Handler",
                    workdir="path/inside/functions/archive",
                    runtime="golang",
                )
        """
        self.spec.build.source = source
        # update handler in function_handler if needed
        if handler:
            self.spec.function_handler = handler
        if workdir:
            self.spec.workdir = workdir
        if runtime:
            self.spec.nuclio_runtime = runtime

        return self

    def with_v3io(self, local="", remote=""):
        """Add v3io volume to the function

        :param local: local path (mount path inside the function container)
        :param remote: v3io path
        """
        if local and remote:
            self.apply(
                mount_v3io(
                    remote=remote, volume_mounts=[VolumeMount(path=local, sub_path="")]
                )
            )
        else:
            self.apply(v3io_cred())
        return self

    def with_http(
        self,
        workers: typing.Optional[int] = 8,
        port: typing.Optional[int] = None,
        host: typing.Optional[str] = None,
        paths: typing.Optional[list[str]] = None,
        canary: typing.Optional[float] = None,
        secret: typing.Optional[str] = None,
        worker_timeout: typing.Optional[int] = None,
        gateway_timeout: typing.Optional[int] = None,
        trigger_name: typing.Optional[str] = None,
        annotations: typing.Optional[typing.Mapping[str, str]] = None,
        extra_attributes: typing.Optional[typing.Mapping[str, str]] = None,
    ):
        """update/add nuclio HTTP trigger settings

        Note: gateway timeout is the maximum request time before an error is returned, while the worker timeout
        if the max time a request will wait for until it will start processing, gateway_timeout must be greater than
        the worker_timeout.

        :param workers:    number of worker processes (default=8). set 0 to use Nuclio's default workers count
        :param port:       TCP port to listen on. by default, nuclio will choose a random port as long as
                           the function service is NodePort. if the function service is ClusterIP, the port
                           is ignored.
        :param host:       Ingress hostname
        :param paths:      list of Ingress sub paths
        :param canary:     k8s ingress canary (% traffic value between 0 and 100)
        :param secret:     k8s secret name for SSL certificate
        :param worker_timeout:  worker wait timeout in sec (how long a message should wait in the worker queue
                                before an error is returned)
        :param gateway_timeout: nginx ingress timeout in sec (request timeout, when will the gateway return an error)
        :param trigger_name:    alternative nuclio trigger name
        :param annotations:     key/value dict of ingress annotations
        :param extra_attributes: key/value dict of extra nuclio trigger attributes
        :return: function object (self)
        """
        if self.disable_default_http_trigger:
            logger.warning(
                "Adding HTTP trigger despite the default HTTP trigger creation being disabled"
            )

        annotations = annotations or {}
        if worker_timeout:
            gateway_timeout = gateway_timeout or (worker_timeout + 60)
        if workers is None:
            workers = 0
        if gateway_timeout:
            if worker_timeout and worker_timeout >= gateway_timeout:
                raise ValueError(
                    "gateway timeout must be greater than the worker timeout"
                )
            mlrun.runtimes.utils.enrich_gateway_timeout_annotations(
                annotations, gateway_timeout
            )

        trigger = nuclio.HttpTrigger(
            workers=workers,
            port=port,
            host=host,
            paths=paths,
            canary=canary,
            secret=secret,
            annotations=annotations,
            extra_attributes=extra_attributes,
        )
        if worker_timeout:
            trigger._struct["workerAvailabilityTimeoutMilliseconds"] = (
                worker_timeout
            ) * 1000
        self.add_trigger(trigger_name or "http", trigger)
        return self

    def from_image(self, image):
        """
        Deploy the function with an existing nuclio processor image.

        :param image: image name
        """
        config = nuclio.config.new_config()
        update_in(
            config,
            "spec.handler",
            self.spec.function_handler or "main:handler",
        )
        update_in(config, "spec.image", image)
        update_in(config, "spec.build.codeEntryType", "image")
        self.spec.base_spec = config

    def add_v3io_stream_trigger(
        self,
        stream_path,
        name="stream",
        group="serving",
        seek_to="earliest",
        shards=1,
        extra_attributes=None,
        ack_window_size=None,
        **kwargs,
    ):
        """add v3io stream trigger to the function

        :param stream_path:    v3io stream path (e.g. 'v3io:///projects/myproj/stream1')
        :param name:           trigger name
        :param group:          consumer group
        :param seek_to:        start seek from: "earliest", "latest", "time", "sequence"
        :param shards:         number of shards (used to set number of replicas)
        :param extra_attributes: key/value dict with extra trigger attributes
        :param ack_window_size:  stream ack window size (the consumer group will be updated with the
                                 event id - ack_window_size, on failure the events in the window will be retransmitted)
        :param kwargs:         extra V3IOStreamTrigger class attributes
        """
        endpoint = None
        if "://" in stream_path:
            endpoint, stream_path = parse_path(stream_path, suffix="")

        # verify v3io stream trigger name is valid
        mlrun.utils.helpers.validate_v3io_stream_consumer_group(group)

        if "consumer_group" in kwargs:
            logger.warning(
                "'consumer_group' in kwargs will be ignored. Use group parameter instead."
            )

        container, path = split_path(stream_path)
        shards = shards or 1
        extra_attributes = extra_attributes or {}
        if ack_window_size:
            extra_attributes["ackWindowSize"] = ack_window_size

        access_key = kwargs.pop("access_key", None)
        if not access_key:
            access_key = self._resolve_v3io_access_key()
        engine = "sync"
        explicit_ack_mode = kwargs.pop("explicit_ack_mode", None)
        if self.spec and hasattr(self.spec, "graph"):
            engine = getattr(self.spec.graph, "engine", None) or engine

        if mlrun.mlconf.is_explicit_ack_enabled() and engine == "async":
            explicit_ack_mode = explicit_ack_mode or "explicitOnly"

        self.add_trigger(
            name,
            V3IOStreamTrigger(
                name=name,
                container=container,
                path=path[1:],
                consumer_group=group,
                seek_to=seek_to,
                webapi=endpoint or "http://v3io-webapi:8081",
                extra_attributes=extra_attributes,
                read_batch_size=256,
                access_key=access_key,
                explicit_ack_mode=explicit_ack_mode,
                **kwargs,
            ),
        )
        if self.spec.min_replicas != shards or self.spec.max_replicas != shards:
            logger.warning(f"Setting function replicas to {shards}")
            self.spec.min_replicas = shards
            self.spec.max_replicas = shards

    def deploy(
        self,
        project="",
        tag="",
        verbose=False,
        auth_info: AuthInfo = None,
        builder_env: typing.Optional[dict] = None,
        force_build: bool = False,
    ):
        """Deploy the nuclio function to the cluster

        :param project:    project name
        :param tag:        function tag
        :param verbose:    set True for verbose logging
        :param auth_info:  service AuthInfo (deprecated and ignored)
        :param builder_env: env vars dict for source archive config/credentials e.g. builder_env={"GIT_TOKEN": token}
        :param force_build: set True for force building the image
        """
        if auth_info:
            # TODO: remove in 1.10.0
            warnings.warn(
                "'auth_info' is deprecated for nuclio runtimes in 1.7.0 and will be removed in 1.10.0",
                FutureWarning,
            )

        old_http_session = getattr(self, "_http_session", None)
        if old_http_session:
            # ensure existing http session is terminated prior to (re)deploy to ensure that a connection to an old
            # replica will not be reused
            old_http_session.close()
            self._http_session = None

        verbose = verbose or self.verbose
        if verbose:
            self.set_env("MLRUN_LOG_LEVEL", "DEBUG")
        if project:
            self.metadata.project = project
        if tag:
            self.metadata.tag = tag

        # Attempt auto-mounting, before sending to remote build
        self.try_auto_mount_based_on_config()
        self._fill_credentials()
        db = self._get_db()
        logger.info("Starting remote function deploy")
        data = db.deploy_nuclio_function(func=self, builder_env=builder_env)
        self.status = data["data"].get("status")
        self._update_credentials_from_remote_build(data["data"])

        # when a function is deployed, we wait for it to be ready by default
        # this also means that the function object will be updated with the function status
        self._wait_for_function_deployment(db, verbose=verbose)
        # check if there are any background tasks related to creating model endpoints
        model_endpoints_creation_background_tasks = (
            mlrun.common.schemas.BackgroundTaskList(
                **data.pop("background_tasks", {"background_tasks": []})
            ).background_tasks
        )
        if model_endpoints_creation_background_tasks:
            self._check_model_endpoint_task_state(
                db=db,
                background_task=model_endpoints_creation_background_tasks[0],
                wait_for_completion=False,
            )

        return self._enrich_command_from_status()

    def _enrich_command_from_status(self):
        # NOTE: on older mlrun versions & nuclio versions, function are exposed via NodePort
        #       now, functions can be not exposed (using service type ClusterIP) and hence
        #       for BC we first try to populate the external invocation url, and then
        #       if not exists, take the internal invocation url
        if (
            self.status.external_invocation_urls
            and self.status.external_invocation_urls[0] != ""
        ):
            self.spec.command = f"http://{self.status.external_invocation_urls[0]}"
        elif (
            self.status.internal_invocation_urls
            and self.status.internal_invocation_urls[0] != ""
        ):
            self.spec.command = f"http://{self.status.internal_invocation_urls[0]}"
        elif self.status.address and self.status.address != "":
            self.spec.command = f"http://{self.status.address}"

        logger.info(
            "Successfully deployed function",
            internal_invocation_urls=self.status.internal_invocation_urls,
            external_invocation_urls=self.status.external_invocation_urls,
        )

        self.save(versioned=False)

        return self.spec.command

    def _wait_for_function_deployment(self, db, verbose=False):
        state = ""
        last_log_timestamp = 1
        while state not in ["ready", "error", "unhealthy"]:
            sleep(
                int(mlrun.mlconf.httpdb.logs.nuclio.pull_deploy_status_default_interval)
            )
            try:
                text, last_log_timestamp = db.get_nuclio_deploy_status(
                    self, last_log_timestamp=last_log_timestamp, verbose=verbose
                )
            except mlrun.db.RunDBError:
                raise ValueError("function or deploy process not found")
            state = self.status.state
            if text:
                print(text)

        if state != "ready":
            logger.error("Nuclio function failed to deploy", function_state=state)
            raise RunError(f"Function {self.metadata.name} deployment failed")

    @min_nuclio_versions("1.5.20", "1.6.10")
    def with_node_selection(
        self,
        node_name: typing.Optional[str] = None,
        node_selector: typing.Optional[dict[str, str]] = None,
        affinity: typing.Optional[client.V1Affinity] = None,
        tolerations: typing.Optional[list[client.V1Toleration]] = None,
    ):
        """k8s node selection attributes"""
        if tolerations and not validate_nuclio_version_compatibility("1.7.5"):
            raise mlrun.errors.MLRunIncompatibleVersionError(
                "tolerations are only supported from nuclio version 1.7.5"
            )
        super().with_node_selection(node_name, node_selector, affinity, tolerations)

    @min_nuclio_versions("1.8.6")
    def with_preemption_mode(self, mode):
        """
        Preemption mode controls whether pods can be scheduled on preemptible nodes.
        Tolerations, node selector, and affinity are populated on preemptible nodes corresponding to the function spec.

        The supported modes are:

        * **allow** - The function can be scheduled on preemptible nodes
        * **constrain** - The function can only run on preemptible nodes
        * **prevent** - The function cannot be scheduled on preemptible nodes
        * **none** - No preemptible configuration will be applied on the function

        The default preemption mode is configurable in mlrun.mlconf.function_defaults.preemption_mode,
        by default it's set to **prevent**

        :param mode: allow | constrain | prevent | none defined in :py:class:`~mlrun.common.schemas.PreemptionModes`
        """
        super().with_preemption_mode(mode=mode)

    @min_nuclio_versions("1.6.18")
    def with_priority_class(self, name: typing.Optional[str] = None):
        """k8s priority class"""
        super().with_priority_class(name)

    def with_service_type(
        self,
        service_type: str,
        add_templated_ingress_host_mode: typing.Optional[str] = None,
    ):
        """
        Enables to control the service type of the pod and the addition of templated ingress host

        :param service_type:                      service type (ClusterIP, NodePort), defaults to
                                                  mlrun.mlconf.httpdb.nuclio.service_type
        :param add_templated_ingress_host_mode:   add templated ingress host mode (never, always, onClusterIP),
                                                  see mlrun.mlconf.httpdb.nuclio.add_templated_ingress_host_mode
                                                  for the default and more information

        """
        self.spec.service_type = service_type
        self.spec.add_templated_ingress_host_mode = add_templated_ingress_host_mode

    def set_state_thresholds(
        self,
        state_thresholds: dict[str, int],
        patch: bool = True,
    ):
        raise NotImplementedError(
            "State thresholds do not apply for nuclio as it has its own function pods healthiness monitoring"
        )

    @min_nuclio_versions("1.13.1")
    def disable_default_http_trigger(
        self,
    ):
        """
        Disables nuclio's default http trigger creation
        """
        self.spec.disable_default_http_trigger = True

    @min_nuclio_versions("1.13.1")
    def enable_default_http_trigger(
        self,
    ):
        """
        Enables nuclio's default http trigger creation
        """
        self.spec.disable_default_http_trigger = False

    def skip_image_enrichment(self):
        # make sure the API does not enrich the base image if the function is not a python function
        return self.spec.nuclio_runtime and "python" not in self.spec.nuclio_runtime

    def _get_state(
        self,
        last_log_timestamp=0,
        verbose=False,
        raise_on_exception=True,
    ) -> tuple[str, str, typing.Optional[float]]:
        try:
            text, last_log_timestamp = self._get_db().get_nuclio_deploy_status(
                self, last_log_timestamp=last_log_timestamp, verbose=verbose
            )
        except mlrun.db.RunDBError:
            if raise_on_exception:
                return "", "", None
            raise ValueError("function or deploy process not found")
        return self.status.state, text, last_log_timestamp

    def _get_runtime_env(self):
        # for runtime specific env var enrichment (before deploy)
        runtime_env = {
            "MLRUN_DEFAULT_PROJECT": self.metadata.project or mlconf.default_project,
        }
        if mlconf.httpdb.api_url:
            runtime_env["MLRUN_DBPATH"] = mlconf.httpdb.api_url
        if mlconf.namespace:
            runtime_env["MLRUN_NAMESPACE"] = mlconf.namespace
        if self.metadata.credentials.access_key:
            runtime_env[
                mlrun.common.runtimes.constants.FunctionEnvironmentVariables.auth_session
            ] = self.metadata.credentials.access_key
        return runtime_env

    def _get_serving_spec(self):
        return None

    def _get_nuclio_config_spec_env(self):
        env_dict = {}
        external_source_env_dict = {}

        api = client.ApiClient()
        for env_var in self.spec.env:
            # sanitize env if not sanitized
            if isinstance(env_var, dict):
                sanitized_env_var = env_var
            else:
                sanitized_env_var = api.sanitize_for_serialization(env_var)

            value = sanitized_env_var.get("value")
            if value is not None:
                env_dict[sanitized_env_var.get("name")] = value
                continue

            value_from = sanitized_env_var.get("valueFrom")
            if value_from is not None:
                external_source_env_dict[sanitized_env_var.get("name")] = value_from

        for key, value in self._get_runtime_env().items():
            env_dict[key] = value

        return env_dict, external_source_env_dict

    def deploy_step(
        self,
        project="",
        models=None,
        env=None,
        tag=None,
        verbose=None,
        use_function_from_db=None,
    ):
        """return as a Kubeflow pipeline step (ContainerOp), recommended to use mlrun.deploy_function() instead

        :param project:        project name, defaults to function project
        :param models:         model name and paths
        :param env:            dict of environment variables
        :param tag:            version tag
        :param verbose:        verbose output
        :param use_function_from_db:  use the function from the DB instead of the local function object
        """
        models = {} if models is None else models
        function_name = self.metadata.name or "function"
        name = f"deploy_{function_name}"
        project = project or self.metadata.project
        if models and isinstance(models, dict):
            models = [{"key": k, "model_path": v} for k, v in models.items()]

        # verify auto mount is applied (with the client credentials)
        self.try_auto_mount_based_on_config()

        if (
            use_function_from_db
            or use_function_from_db is None
            # if the function contain KFP PipelineParams (futures) pass the full spec to the
            # ContainerOp this way KFP will substitute the params with previous step outputs
            and not self._has_pipeline_param()
        ):
            url = self.save(versioned=True, refresh=True)
        else:
            url = None

        return deploy_op(
            name,
            self,
            func_url=url,
            project=project,
            models=models,
            env=env,
            tag=tag,
            verbose=verbose,
        )

    def invoke(
        self,
        path: str,
        body: typing.Optional[typing.Union[str, bytes, dict]] = None,
        method: typing.Optional[str] = None,
        headers: typing.Optional[dict] = None,
        force_external_address: bool = False,
        auth_info: AuthInfo = None,
        mock: typing.Optional[bool] = None,
        **http_client_kwargs,
    ):
        """Invoke the remote (live) function and return the results

        example::

            function.invoke("/api", body={"inputs": x})

        :param path:     request sub path (e.g. /images)
        :param body:     request body (str, bytes or a dict for json requests)
        :param method:   HTTP method (GET, PUT, ..)
        :param headers:  key/value dict with http headers
        :param force_external_address:   use the external ingress URL
        :param auth_info: service AuthInfo
        :param mock:     use mock server vs a real Nuclio function (for local simulations)
        :param http_client_kwargs:   allow the user to pass any parameter supported in requests.request method
                                     see this link for more information:
                                     https://requests.readthedocs.io/en/latest/api/#requests.request
        """
        if not method:
            method = "POST" if body else "GET"

        # if no path was provided, use the default handler to be invoked
        if not path and self.spec.default_handler:
            path = self.spec.default_handler

        if (self._mock_server and mock is None) or mlconf.use_nuclio_mock(mock):
            # if we deployed mock server or in simulated nuclio environment use mock
            if not self._mock_server:
                self._set_as_mock(True)
            return self._mock_server.test(path, body, method, headers)

        # clear the mock server when using the real endpoint
        self._mock_server = None

        if "://" not in path:
            if not self.status.address:
                # here we check that if default http trigger is disabled, function contains a custom http trigger
                # Otherwise, the function is not invokable, so we raise an error
                if (
                    not self._trigger_of_kind_exists(kind="http")
                    and self.spec.disable_default_http_trigger
                ):
                    raise mlrun.errors.MLRunPreconditionFailedError(
                        "Default http trigger creation is disabled and there is no any other custom http trigger, "
                        "so function can not be invoked via http. Either enable default http trigger creation or "
                        "create custom http trigger"
                    )
                state, _, _ = self._get_state()
                if state not in ["ready", "scaledToZero"]:
                    logger.warning(f"Function is in the {state} state")
                if not self.status.address:
                    raise ValueError("no function address first run .deploy()")

            path = self._resolve_invocation_url(path, force_external_address)

        if headers is None:
            headers = {}

        # if function is scaled to zero, let the DLX know we want to wake it up
        full_function_name = get_fullname(
            self.metadata.name, self.metadata.project, self.metadata.tag
        )
        headers.setdefault("x-nuclio-target", full_function_name)
        if not http_client_kwargs:
            http_client_kwargs = {}
        if body:
            if isinstance(body, (str, bytes)):
                http_client_kwargs["data"] = body
            else:
                http_client_kwargs["json"] = body
        try:
            logger.debug("Invoking function", method=method, path=path)
            if not getattr(self, "_http_session", None):
                self._http_session = requests.Session()
            resp = self._http_session.request(
                method, path, headers=headers, **http_client_kwargs
            )
        except OSError as err:
            raise OSError(
                f"error: cannot run function at url {path}, {err_to_str(err)}"
            )
        if not resp.ok:
            raise RuntimeError(f"bad function response {resp.status_code}: {resp.text}")

        data = resp.content
        if resp.headers["content-type"] == "application/json":
            data = json.loads(data)
        return data

    def with_sidecar(
        self,
        name: typing.Optional[str] = None,
        image: typing.Optional[str] = None,
        ports: typing.Optional[typing.Union[int, list[int]]] = None,
        command: typing.Optional[str] = None,
        args: typing.Optional[list[str]] = None,
    ):
        """
        Add a sidecar container to the function pod

        :param name:    Sidecar container name.
        :param image:   Sidecar container image.
        :param ports:   Sidecar container ports to expose. Can be a single port or a list of ports.
        :param command: Sidecar container command instead of the image entrypoint.
        :param args:    Sidecar container command args (requires command to be set).
        """
        name = name or f"{self.metadata.name}-sidecar"
        sidecar = self._set_sidecar(name)
        if image:
            sidecar["image"] = image

        ports = mlrun.utils.helpers.as_list(ports)
        # according to RFC-6335, port name should be less than 15 characters,
        # so we truncate it if needed and leave room for the index
        port_name = name[:13].rstrip("-_") if len(name) > 13 else name
        sidecar["ports"] = [
            {
                "name": f"{port_name}-{i}",
                "containerPort": port,
                "protocol": "TCP",
            }
            for i, port in enumerate(ports)
        ]

        # if it is a redeploy, 'command' might be set with the previous invocation url.
        # in this case, we don't want to use it as the sidecar command
        if command and not command.startswith("http"):
            sidecar["command"] = mlrun.utils.helpers.as_list(command)

        if args and sidecar.get("command"):
            sidecar["args"] = mlrun.utils.helpers.as_list(args)

        # put the configured resources on the sidecar container instead of the reverse proxy container
        if self.spec.resources:
            sidecar["resources"] = self.spec.resources
            self.spec.resources = None

    def _set_sidecar(self, name: str) -> dict:
        self.spec.config.setdefault("spec.sidecars", [])
        sidecars = self.spec.config["spec.sidecars"]
        for sidecar in sidecars:
            if sidecar["name"] == name:
                return sidecar

        sidecars.append({"name": name})
        return sidecars[-1]

    def _trigger_of_kind_exists(self, kind: str) -> bool:
        if not self.spec.config:
            return False

        for name, spec in self.spec.config.items():
            if isinstance(spec, dict):
                if spec.get("kind") == kind:
                    return True

        return False

    def _pre_run_validations(self):
        if self.spec.function_kind != "mlrun":
            raise RunError(
                '.run() can only be execute on "mlrun" kind'
                ', recreate with function kind "mlrun"'
            )

        state = self.status.state
        if state != "ready":
            if state:
                raise RunError(f"Cannot run, function in state {state}")
            state, _, _ = self._get_state(raise_on_exception=True)
            if state != "ready":
                logger.info("Starting nuclio build!")
                self.deploy()

    def _run(self, runobj: RunObject, execution):
        self._pre_run_validations()
        self.store_run(runobj)
        if self._secrets:
            runobj.spec.secret_sources = self._secrets.to_serial()
        log_level = execution.log_level
        command = self.spec.command
        if runobj.spec.handler:
            command = f"{command}/{runobj.spec.handler_name}"
        headers = {"x-nuclio-log-level": log_level}
        try:
            resp = requests.put(command, json=runobj.to_dict(), headers=headers)
        except OSError as err:
            logger.error(f"error invoking function: {err_to_str(err)}")
            raise OSError(f"error: cannot run function at url {command}")

        if not resp.ok:
            logger.error(f"bad function resp!!\n{resp.text}")
            raise RunError("bad function response")

        logs = resp.headers.get("X-Nuclio-Logs")
        if logs:
            log_std(self._db_conn, runobj, parse_logs(logs))

        return self._update_state(resp.json())

    def _run_many(self, generator, execution, runobj: RunObject):
        self._pre_run_validations()
        tasks = generator.generate(runobj)
        secrets = self._secrets.to_serial() if self._secrets else None
        log_level = execution.log_level
        headers = {"x-nuclio-log-level": log_level}

        command = self.spec.command
        if runobj.spec.handler:
            command = f"{command}/{runobj.spec.handler_name}"
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(
            self._invoke_async(tasks, command, headers, secrets, generator=generator)
        )

        loop.run_until_complete(future)
        return future.result()

    def _update_state(self, rundict: dict):
        last_state = get_in(rundict, "status.state", "")
        if last_state != "error":
            update_in(rundict, "status.state", "completed")
        self._store_run_dict(rundict)
        return rundict

    async def _invoke_async(self, tasks, url, headers, secrets, generator):
        results = RunList()
        runs = []
        num_errors = 0
        stop = False
        parallel_runs = generator.options.parallel_runs or 1
        semaphore = asyncio.Semaphore(parallel_runs)

        async with ClientSession() as session:
            for task in tasks:
                # TODO: store run using async calls to improve performance
                self.store_run(task)
                task.spec.secret_sources = secrets or []
                resp = submit(session, url, task, semaphore, headers=headers)
                runs.append(
                    asyncio.ensure_future(
                        resp,
                    )
                )

            for result in asyncio.as_completed(runs):
                status, resp, logs, task = await result

                if status != 200:
                    err_message = f"failed to access {url} - {resp}"
                    # TODO: store logs using async calls to improve performance
                    log_std(
                        self._db_conn,
                        task,
                        parse_logs(logs) if logs else None,
                        err_message,
                        silent=True,
                    )
                    # TODO: update run using async calls to improve performance
                    results.append(self._update_run_state(task=task, err=err_message))
                    num_errors += 1
                else:
                    if logs:
                        log_std(self._db_conn, task, parse_logs(logs))
                    resp = self._update_run_state(json.loads(resp))
                    state = get_in(resp, "status.state", "")
                    if state == "error":
                        num_errors += 1
                    results.append(resp)

                    run_results = get_in(resp, "status.results", {})
                    stop = generator.eval_stop_condition(run_results)
                    if stop:
                        logger.info(
                            f"Reached early stop condition ({generator.options.stop_condition}), stopping iterations!"
                        )
                        break

                if num_errors > generator.max_errors:
                    logger.error("Max errors reached, stopping iterations!")
                    stop = True
                    break

        if stop:
            for task in runs:
                task.cancel()
        return results

    def _resolve_invocation_url(self, path, force_external_address):
        # internal / external invocation urls is a nuclio >= 1.6.x feature
        # try to infer the invocation url from the internal and if not exists, use external.
        # $$$$ we do not want to use the external invocation url (e.g.: ingress, nodePort, etc.)
        if (
            not force_external_address
            and self.status.internal_invocation_urls
            and mlrun.k8s_utils.is_running_inside_kubernetes_cluster()
        ):
            return mlrun.utils.helpers.join_urls(
                f"http://{self.status.internal_invocation_urls[0]}", path
            )

        if self.status.external_invocation_urls:
            return mlrun.utils.helpers.join_urls(
                f"http://{self.status.external_invocation_urls[0]}", path
            )
        else:
            return mlrun.utils.helpers.join_urls(f"http://{self.status.address}", path)

    def _update_credentials_from_remote_build(self, remote_data):
        self.metadata.credentials = remote_data.get("metadata", {}).get(
            "credentials", {}
        )

        credentials_env_var_names = ["V3IO_ACCESS_KEY", "MLRUN_AUTH_SESSION"]
        new_env = []

        # the env vars in the local spec and remote spec are in the format of a list of dicts
        # e.g.:
        # env = [
        #   {
        #     "name": "V3IO_ACCESS_KEY",
        #     "value": "some-value"
        #   },
        #   ...
        # ]
        # remove existing credentials env vars
        for env in self.spec.env:
            if isinstance(env, dict):
                env_name = env["name"]
            elif isinstance(env, client.V1EnvVar):
                env_name = env.name
            else:
                continue

            if env_name not in credentials_env_var_names:
                new_env.append(env)

        # add credentials env vars from remote build
        for remote_env in remote_data.get("spec", {}).get("env", []):
            if remote_env.get("name") in credentials_env_var_names:
                new_env.append(remote_env)

        # update nuclio-specific credentials
        self.mask_sensitive_data_in_config()

        self.spec.env = new_env

    def _set_as_mock(self, enable):
        # todo: create mock_server for Nuclio
        if enable:
            raise NotImplementedError(
                "Mock (simulation) is currently not supported for Nuclio, Turn off the mock (mock=False) "
                "and make sure Nuclio is installed for real deployment to Nuclio"
            )

    def get_url(
        self,
        force_external_address: bool = False,
        auth_info: AuthInfo = None,
    ):
        """
        This method returns function's url.

        :param force_external_address:   use the external ingress URL
        :param auth_info:                service AuthInfo

        :return: returns function's url
        """
        if not self.status.address:
            state, _, _ = self._get_state(auth_info=auth_info)
            if state != "ready" or not self.status.address:
                raise ValueError(
                    "no function address or not ready, first run .deploy()"
                )

        return self._resolve_invocation_url("", force_external_address)

    @staticmethod
    def _resolve_v3io_access_key():
        # Nuclio supports generating access key for v3io stream trigger only from version 1.13.11
        if validate_nuclio_version_compatibility("1.13.11"):
            return mlrun.model.Credentials.generate_access_key
        return None

    def _check_model_endpoint_task_state(
        self,
        db: mlrun.db.RunDBInterface,
        background_task: mlrun.common.schemas.BackgroundTask,
        wait_for_completion: bool,
    ):
        if wait_for_completion:
            background_task = db._wait_for_background_task_to_reach_terminal_state(
                name=background_task.metadata.name, project=self.metadata.project
            )
        else:
            background_task = db.get_project_background_task(
                project=self.metadata.project, name=background_task.metadata.name
            )
        if (
            background_task.status.state
            in mlrun.common.schemas.BackgroundTaskState.terminal_states()
        ):
            logger.info(
                f"Model endpoint creation task completed with state {background_task.status.state}"
            )
        else:
            logger.warning(
                f"Model endpoint creation task is still running with state {background_task.status.state}"
                f"You can use the serving function, but it won't be monitored for the next few minutes"
            )


def parse_logs(logs):
    logs = json.loads(logs)
    lines = ""
    for line in logs:
        extra = []
        for key, value in line.items():
            if key not in ["time", "level", "name", "message"]:
                extra.append(f"{key}={value}")
        extra = ", ".join(extra)
        time = datetime.fromtimestamp(float(line["time"]) / 1000).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )
        lines += f"{time}  {line['level']:<6} {line['message']}  {extra}\n"

    return lines


async def submit(session, url, run, semaphore, headers=None):
    async with semaphore:
        async with session.put(url, json=run.to_dict(), headers=headers) as response:
            text = await response.text()
            logs = response.headers.get("X-Nuclio-Logs", None)
            return response.status, text, logs, run


def fake_nuclio_context(body, headers=None):
    return nuclio.Context(), nuclio.Event(body=body, headers=headers)


def get_fullname(name, project, tag):
    if project:
        name = f"{project}-{name}"
    if tag and tag != "latest":
        name = f"{name}-{tag}"
    return name


def get_nuclio_deploy_status(
    name,
    project,
    tag,
    last_log_timestamp=0,
    verbose=False,
    resolve_address=True,
    auth_info: AuthInfo = None,
):
    """
    Get nuclio function deploy status

    :param name:                function name
    :param project:             project name
    :param tag:                 function tag
    :param last_log_timestamp:  last log timestamp
    :param verbose:             print logs
    :param resolve_address:     whether to resolve function address
    :param auth_info:           authentication information
    """
    api_address = find_dashboard_url(mlconf.nuclio_dashboard_url)
    name = get_fullname(name, project, tag)
    get_err_message = f"Failed to get function {name} deploy status"

    try:
        (
            state,
            address,
            last_log_timestamp,
            outputs,
            function_status,
        ) = get_deploy_status(
            api_address,
            name,
            last_log_timestamp,
            verbose,
            resolve_address,
            return_function_status=True,
            auth_info=auth_info.to_nuclio_auth_info() if auth_info else None,
        )
    except requests.exceptions.ConnectionError as exc:
        mlrun.errors.raise_for_status(
            exc.response,
            get_err_message,
        )

    except nuclio.utils.DeployError as exc:
        if exc.err:
            mlrun.errors.raise_for_status(
                exc.err.response,
                get_err_message,
            )
        raise exc
    else:
        text = "\n".join(outputs) if outputs else ""
        return state, address, name, last_log_timestamp, text, function_status


def enrich_nuclio_function_from_headers(
    func: RemoteRuntime,
    headers: dict,
):
    func.status.state = headers.get("x-mlrun-function-status", "")
    func.status.address = headers.get("x-mlrun-address", "")
    func.status.nuclio_name = headers.get("x-mlrun-name", "")
    func.status.internal_invocation_urls = (
        headers.get("x-mlrun-internal-invocation-urls", "").split(",")
        if headers.get("x-mlrun-internal-invocation-urls")
        else []
    )
    func.status.external_invocation_urls = (
        headers.get("x-mlrun-external-invocation-urls", "").split(",")
        if headers.get("x-mlrun-external-invocation-urls")
        else []
    )
    func.status.container_image = headers.get("x-mlrun-container-image", "")
