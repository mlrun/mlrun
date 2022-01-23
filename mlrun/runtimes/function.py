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

import asyncio
import json
import typing
import warnings
from datetime import datetime
from os import getenv
from time import sleep

import nuclio
import requests
import semver
from aiohttp.client import ClientSession
from kubernetes import client
from nuclio.deploy import find_dashboard_url, get_deploy_status
from nuclio.triggers import V3IOStreamTrigger

import mlrun.errors
from mlrun.datastore import parse_s3_bucket_and_key
from mlrun.db import RunDBError

from ..api.schemas import AuthInfo
from ..config import config as mlconf
from ..k8s_utils import get_k8s_helper
from ..kfpops import deploy_op
from ..lists import RunList
from ..model import RunObject
from ..platforms.iguazio import mount_v3io, parse_v3io_path, split_path, v3io_cred
from ..utils import enrich_image_url, get_in, logger, update_in
from .base import FunctionStatus, RunError
from .constants import NuclioIngressAddTemplatedIngressModes
from .pod import KubeResource, KubeResourceSpec
from .utils import get_item_name, log_std

default_max_replicas = 4


def validate_nuclio_version_compatibility(*min_versions):
    """
    :param min_versions: Valid minimum version(s) required, assuming no 2 versions has equal major and minor.
    """
    parsed_min_versions = [
        semver.VersionInfo.parse(min_version) for min_version in min_versions
    ]
    try:
        parsed_current_version = semver.VersionInfo.parse(mlconf.nuclio_version)
    except ValueError:
        logger.warning(
            "Unable to parse nuclio version, assuming compatibility",
            nuclio_version=mlconf.nuclio_version,
            min_versions=min_versions,
        )
        return True

    parsed_min_versions.sort(reverse=True)
    for parsed_min_version in parsed_min_versions:
        if (
            parsed_current_version.major == parsed_min_version.major
            and parsed_current_version.minor == parsed_min_version.minor
            and parsed_current_version.patch < parsed_min_version.patch
        ):
            return False

        if parsed_current_version >= parsed_min_version:
            return True
    return False


def min_nuclio_versions(*versions):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if validate_nuclio_version_compatibility(*versions):
                return function(*args, **kwargs)

            message = (
                f"{function.__name__} is supported since nuclio {' or '.join(versions)}, currently using "
                f"nuclio {mlconf.nuclio_version}, please upgrade."
            )
            raise mlrun.errors.MLRunIncompatibleVersionError(message)

        return wrapper

    return decorator


class NuclioSpec(KubeResourceSpec):
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
        default_handler=None,
        node_name=None,
        node_selector=None,
        affinity=None,
        disable_auto_mount=False,
        priority_class_name=None,
        pythonpath=None,
        workdir=None,
        image_pull_secret=None,
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
        )

        self.base_spec = base_spec or ""
        self.function_kind = function_kind
        self.source = source or ""
        self.config = config or {}
        self.function_handler = ""
        self.no_cache = no_cache
        self.replicas = replicas
        self.readiness_timeout = readiness_timeout

        # TODO: we would prefer to default to 0, but invoking a scaled to zero function requires to either add the
        #  x-nuclio-target header or to create the function with http trigger and invoke the function through it - so
        #  we need to do one of the two
        self.min_replicas = min_replicas or 1
        self.max_replicas = max_replicas or default_max_replicas

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

        volumes_without_volume_mounts = volume_with_volume_mounts_names.symmetric_difference(
            self._volumes.keys()
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


class RemoteRuntime(KubeResource):
    kind = "remote"
    _is_nested = False

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

    def set_config(self, key, value):
        self.spec.config[key] = value
        return self

    def add_volume(self, local, remote, name="fs", access_key="", user=""):
        raise Exception("deprecated, use .apply(mount_v3io())")

    def add_trigger(self, name, spec):
        """add a nuclio trigger object/dict

        :param name: trigger name
        :param spec: trigger object or dict
        """
        if hasattr(spec, "to_dict"):
            spec = spec.to_dict()
        spec["name"] = name
        self.spec.config[f"spec.triggers.{name}"] = spec
        return self

    def with_source_archive(
        self, source, handler="", runtime="", secrets=None,
    ):
        """Load nuclio function from remote source
        :param source: a full path to the nuclio function source (code entry) to load the function from
        :param handler: a path to the function's handler, including path inside archive/git repo
        :param runtime: (optional) the runtime of the function (defaults to python:3.7)
        :param secrets: a dictionary of secrets to be used to fetch the function from the source.
               (can also be passed using env vars). options:
               ["V3IO_ACCESS_KEY",
               "GIT_USERNAME",
               "GIT_PASSWORD",
               "AWS_ACCESS_KEY_ID",
               "AWS_SECRET_ACCESS_KEY",
               "AWS_SESSION_TOKEN"]

        Examples::
            git:
                ("git://github.com/org/repo#my-branch",
                 handler="path/inside/repo#main:handler",
                 secrets={"GIT_PASSWORD": "my-access-token"})
            s3:
                ("s3://my-bucket/path/in/bucket/my-functions-archive",
                 handler="path/inside/functions/archive#main:Handler",
                 runtime="golang",
                 secrets={"AWS_ACCESS_KEY_ID": "some-id", "AWS_SECRET_ACCESS_KEY": "some-secret"})
        """
        code_entry_type = self._resolve_code_entry_type(source)
        if code_entry_type == "":
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Couldn't resolve code entry type from source"
            )

        code_entry_attributes = {}

        # resolve work_dir and handler
        work_dir, handler = self._resolve_work_dir_and_handler(handler)
        if work_dir != "":
            code_entry_attributes["workDir"] = work_dir

        if secrets is None:
            secrets = {}

        # set default runtime if not specified otherwise
        if runtime == "":
            runtime = mlrun.config.config.default_nuclio_runtime

        # archive
        if code_entry_type == "archive":
            if source.startswith("v3io"):
                source = f"http{source[len('v3io'):]}"

            v3io_access_key = secrets.get(
                "V3IO_ACCESS_KEY", getenv("V3IO_ACCESS_KEY", "")
            )
            if v3io_access_key:
                code_entry_attributes["headers"] = {
                    "headers": {"X-V3io-Session-Key": v3io_access_key}
                }

        # s3
        if code_entry_type == "s3":
            bucket, item_key = parse_s3_bucket_and_key(source)

            code_entry_attributes["s3Bucket"] = bucket
            code_entry_attributes["s3ItemKey"] = item_key

            code_entry_attributes["s3AccessKeyId"] = secrets.get(
                "AWS_ACCESS_KEY_ID", getenv("AWS_ACCESS_KEY_ID", "")
            )
            code_entry_attributes["s3SecretAccessKey"] = secrets.get(
                "AWS_SECRET_ACCESS_KEY", getenv("AWS_SECRET_ACCESS_KEY", "")
            )
            code_entry_attributes["s3SessionToken"] = secrets.get(
                "AWS_SESSION_TOKEN", getenv("AWS_SESSION_TOKEN", "")
            )

        # git
        if code_entry_type == "git":

            # change git:// to https:// as nuclio expects it to be
            if source.startswith("git://"):
                source = source.replace("git://", "https://")

            source, reference = self._resolve_git_reference_from_source(source)
            if reference:
                code_entry_attributes["reference"] = reference

            code_entry_attributes["username"] = secrets.get("GIT_USERNAME", "")
            code_entry_attributes["password"] = secrets.get(
                "GIT_PASSWORD", getenv("GITHUB_TOKEN", "")
            )

        # update handler in function_handler
        self.spec.function_handler = handler

        # populate spec with relevant fields
        config = nuclio.config.new_config()
        update_in(config, "spec.handler", handler)
        update_in(config, "spec.runtime", runtime)
        update_in(config, "spec.build.path", source)
        update_in(config, "spec.build.codeEntryType", code_entry_type)
        update_in(config, "spec.build.codeEntryAttributes", code_entry_attributes)
        self.spec.base_spec = config

        return self

    def with_v3io(self, local="", remote=""):
        """Add v3io volume to the function

        :param local: local path (mount path inside the function container)
        :param remote: v3io path
        """
        if local and remote:
            self.apply(mount_v3io(remote=remote, mount_path=local))
        else:
            self.apply(v3io_cred())
        return self

    def with_http(
        self,
        workers=8,
        port=0,
        host=None,
        paths=None,
        canary=None,
        secret=None,
        worker_timeout: int = None,
        gateway_timeout: int = None,
        trigger_name=None,
        annotations=None,
        extra_attributes=None,
    ):
        """update/add nuclio HTTP trigger settings

        Note: gateway timeout is the maximum request time before an error is returned, while the worker timeout
        if the max time a request will wait for until it will start processing, gateway_timeout must be greater than
        the worker_timeout.

        :param workers:    number of worker processes (default=8)
        :param port:       TCP port
        :param host:       hostname
        :param paths:      list of sub paths
        :param canary:     k8s ingress canary (% traffic value between 0 to 100)
        :param secret:     k8s secret name for SSL certificate
        :param worker_timeout:  worker wait timeout in sec (how long a message should wait in the worker queue
                                before an error is returned)
        :param gateway_timeout: nginx ingress timeout in sec (request timeout, when will the gateway return an error)
        :param trigger_name:    alternative nuclio trigger name
        :param annotations:     key/value dict of ingress annotations
        :param extra_attributes: key/value dict of extra nuclio trigger attributes
        :return: function object (self)
        """
        annotations = annotations or {}
        if worker_timeout:
            gateway_timeout = gateway_timeout or (worker_timeout + 60)
        if gateway_timeout:
            if worker_timeout and worker_timeout >= gateway_timeout:
                raise ValueError(
                    "gateway timeout must be greater than the worker timeout"
                )
            annotations[
                "nginx.ingress.kubernetes.io/proxy-connect-timeout"
            ] = f"{gateway_timeout}"
            annotations[
                "nginx.ingress.kubernetes.io/proxy-read-timeout"
            ] = f"{gateway_timeout}"
            annotations[
                "nginx.ingress.kubernetes.io/proxy-send-timeout"
            ] = f"{gateway_timeout}"

        trigger = nuclio.HttpTrigger(
            workers,
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

    def add_model(self, name, model_path, **kw):
        warnings.warn(
            'This method is deprecated and will be removed in 0.10.0. Use the "serving" runtime instead',
            # TODO: remove in 0.10.0
            DeprecationWarning,
        )
        if model_path.startswith("v3io://"):
            model = "/User/" + "/".join(model_path.split("/")[5:])
        else:
            model = model_path
        self.set_env(f"SERVING_MODEL_{name}", model)
        return self

    def from_image(self, image):
        config = nuclio.config.new_config()
        update_in(
            config, "spec.handler", self.spec.function_handler or "main:handler",
        )
        update_in(config, "spec.image", image)
        update_in(config, "spec.build.codeEntryType", "image")
        self.spec.base_spec = config

    def serving(
        self,
        models: dict = None,
        model_class="",
        protocol="",
        image="",
        endpoint="",
        explainer=False,
        workers=8,
        canary=None,
    ):
        warnings.warn(
            'This method is deprecated and will be removed in 0.10.0. Use the "serving" runtime instead',
            # TODO: remove in 0.10.0
            DeprecationWarning,
        )

        if models:
            for k, v in models.items():
                self.set_env(f"SERVING_MODEL_{k}", v)

        if protocol:
            self.set_env("TRANSPORT_PROTOCOL", protocol)
        if model_class:
            self.set_env("MODEL_CLASS", model_class)
        self.set_env("ENABLE_EXPLAINER", str(explainer))
        self.with_http(workers, host=endpoint, canary=canary)
        self.spec.function_kind = "serving"

        if image:
            self.from_image(image)

        return self

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

        :param stream_path:    v3io stream path (e.g. 'v3io:///projects/myproj/stream1'
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
            endpoint, stream_path = parse_v3io_path(stream_path, suffix="")
        container, path = split_path(stream_path)
        shards = shards or 1
        extra_attributes = extra_attributes or {}
        if ack_window_size:
            extra_attributes["ackWindowSize"] = ack_window_size
        self.add_trigger(
            name,
            V3IOStreamTrigger(
                name=name,
                container=container,
                path=path[1:],
                consumerGroup=group,
                seekTo=seek_to,
                webapi=endpoint or "http://v3io-webapi:8081",
                extra_attributes=extra_attributes,
                **kwargs,
            ),
        )
        self.spec.min_replicas = shards
        self.spec.max_replicas = shards

    def add_secrets_config_to_spec(self):
        # For nuclio functions, we just add the project secrets as env variables. Since there's no MLRun code
        # to decode the secrets and special env variable names in the function, we just use the same env variable as
        # the key name (encode_key_names=False)
        self._add_project_k8s_secrets_to_spec(
            None, project=self.metadata.project, encode_key_names=False
        )

    def deploy(
        self,
        dashboard="",
        project="",
        tag="",
        verbose=False,
        auth_info: AuthInfo = None,
    ):
        """Deploy the nuclio function to the cluster

        :param dashboard:  address of the nuclio dashboard service (keep blank for current cluster)
        :param project:    project name
        :param tag:        function tag
        :param verbose:    set True for verbose logging
        :param auth_info:  service AuthInfo
        """
        # todo: verify that the function name is normalized

        verbose = verbose or self.verbose
        if verbose:
            self.set_env("MLRUN_LOG_LEVEL", "DEBUG")
        if project:
            self.metadata.project = project
        if tag:
            self.metadata.tag = tag

        save_record = False
        if not dashboard:
            # Attempt auto-mounting, before sending to remote build
            self.try_auto_mount_based_on_config()
            self.fill_credentials()
            db = self._get_db()
            logger.info("Starting remote function deploy")
            data = db.remote_builder(self, False)
            self.status = data["data"].get("status")
            self._wait_for_function_deployment(db, verbose=verbose)

            # NOTE: on older mlrun versions & nuclio versions, function are exposed via NodePort
            #       now, functions can be not exposed (using service type ClusterIP) and hence
            #       for BC we first try to populate the external invocation url, and then
            #       if not exists, take the internal invocation url
            if self.status.external_invocation_urls:
                self.spec.command = f"http://{self.status.external_invocation_urls[0]}"
                save_record = True
            elif self.status.internal_invocation_urls:
                self.spec.command = f"http://{self.status.internal_invocation_urls[0]}"
                save_record = True
            elif self.status.address:
                self.spec.command = f"http://{self.status.address}"
                save_record = True

        else:
            self.save(versioned=False)
            self._ensure_run_db()
            internal_invocation_urls, external_invocation_urls = deploy_nuclio_function(
                self, dashboard=dashboard, watch=True, auth_info=auth_info,
            )
            self.status.internal_invocation_urls = internal_invocation_urls
            self.status.external_invocation_urls = external_invocation_urls

            # save the (first) function external invocation url
            # this is made for backwards compatability because the user, at this point, may
            # work remotely and need the external invocation url on the spec.command
            # TODO: when using `ClusterIP`, this block might not fulfilled
            #       as long as function doesnt have ingresses
            if self.status.external_invocation_urls:
                address = self.status.external_invocation_urls[0]
                self.spec.command = f"http://{address}"
                self.status.state = "ready"
                self.status.address = address
                save_record = True

        logger.info(
            "successfully deployed function",
            internal_invocation_urls=self.status.internal_invocation_urls,
            external_invocation_urls=self.status.external_invocation_urls,
        )

        if save_record:
            self.save(versioned=False)

        return self.spec.command

    def _wait_for_function_deployment(self, db, verbose=False):
        text = ""
        state = ""
        last_log_timestamp = 1
        while state not in ["ready", "error", "unhealthy"]:
            sleep(1)
            try:
                text, last_log_timestamp = db.get_builder_status(
                    self, last_log_timestamp=last_log_timestamp, verbose=verbose
                )
            except RunDBError:
                raise ValueError("function or deploy process not found")
            state = self.status.state
            if text:
                print(text)

        if state != "ready":
            logger.error("Nuclio function failed to deploy", function_state=state)
            raise RunError(f"function {self.metadata.name} deployment failed")

    @min_nuclio_versions("1.5.20", "1.6.10")
    def with_node_selection(
        self,
        node_name: typing.Optional[str] = None,
        node_selector: typing.Optional[typing.Dict[str, str]] = None,
        affinity: typing.Optional[client.V1Affinity] = None,
    ):
        """k8s node selection attributes"""
        super().with_node_selection(node_name, node_selector, affinity)

    @min_nuclio_versions("1.6.18")
    def with_priority_class(self, name: typing.Optional[str] = None):
        """k8s priority class"""
        super().with_priority_class(name)

    def _get_state(
        self,
        dashboard="",
        last_log_timestamp=0,
        verbose=False,
        raise_on_exception=True,
        resolve_address=True,
        auth_info: AuthInfo = None,
    ) -> typing.Tuple[str, str, typing.Optional[float]]:
        if dashboard:
            (
                state,
                address,
                name,
                last_log_timestamp,
                text,
                function_status,
            ) = get_nuclio_deploy_status(
                self.metadata.name,
                self.metadata.project,
                self.metadata.tag,
                dashboard,
                last_log_timestamp=last_log_timestamp,
                verbose=verbose,
                resolve_address=resolve_address,
                auth_info=auth_info,
            )
            self.status.internal_invocation_urls = function_status.get(
                "internalInvocationUrls", []
            )
            self.status.external_invocation_urls = function_status.get(
                "externalInvocationUrls", []
            )
            self.status.state = state
            self.status.nuclio_name = name
            if address:
                self.status.address = address
                self.spec.command = f"http://{address}"
            return state, text, last_log_timestamp

        try:
            text, last_log_timestamp = self._get_db().get_builder_status(
                self, last_log_timestamp=last_log_timestamp, verbose=verbose
            )
        except RunDBError:
            if raise_on_exception:
                return "", "", None
            raise ValueError("function or deploy process not found")
        return self.status.state, text, last_log_timestamp

    @staticmethod
    def _resolve_git_reference_from_source(source):
        split_source = source.split("#")

        # no reference was passed
        if len(split_source) != 2:
            return source

        reference = split_source[1]
        if reference.startswith("refs"):
            return split_source, reference

        return split_source[0], f"refs/heads/{reference}"

    def _resolve_work_dir_and_handler(self, handler):
        """
        Resolves a nuclio function working dir and handler inside an archive/git repo
        :param handler: a path describing working dir and handler of a nuclio function
        :return: (working_dir, handler) tuple, as nuclio expects to get it

        Example: ("a/b/c#main:Handler) -> ("a/b/c", "main:Handler")
        """
        if handler == "":
            return "", self.spec.function_handler or "main:handler"

        split_handler = handler.split("#")
        if len(split_handler) == 1:
            return "", handler

        return "/".join(split_handler[:-1]), split_handler[-1]

    @staticmethod
    def _resolve_code_entry_type(source):
        if source.startswith("s3://"):
            return "s3"
        if source.startswith("git://"):
            return "git"

        for archive_prefix in ["http://", "https://", "v3io://", "v3ios://"]:
            if source.startswith(archive_prefix):
                return "archive"
        return ""

    def _get_runtime_env(self):
        # for runtime specific env var enrichment (before deploy)
        runtime_env = {
            "MLRUN_DEFAULT_PROJECT": self.metadata.project or mlconf.default_project,
        }
        if self.spec.rundb or mlconf.httpdb.api_url:
            runtime_env["MLRUN_DBPATH"] = self.spec.rundb or mlconf.httpdb.api_url
        if mlconf.namespace:
            runtime_env["MLRUN_NAMESPACE"] = mlconf.namespace
        if self.metadata.credentials.access_key:
            runtime_env["MLRUN_AUTH_SESSION"] = self.metadata.credentials.access_key
        return runtime_env

    def get_nuclio_config_spec_env(self):
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
        dashboard="",
        project="",
        models=None,
        env=None,
        tag=None,
        verbose=None,
        use_function_from_db=True,
    ):
        """return as a Kubeflow pipeline step (ContainerOp), recommended to use mlrun.deploy_function() instead"""
        models = {} if models is None else models
        function_name = self.metadata.name or "function"
        name = f"deploy_{function_name}"
        project = project or self.metadata.project
        if models and isinstance(models, dict):
            models = [{"key": k, "model_path": v} for k, v in models.items()]

        if use_function_from_db:
            url = self.save(versioned=True, refresh=True)
        else:
            url = None

        return deploy_op(
            name,
            self,
            func_url=url,
            dashboard=dashboard,
            project=project,
            models=models,
            env=env,
            tag=tag,
            verbose=verbose,
        )

    def invoke(
        self,
        path: str,
        body: typing.Union[str, bytes, dict] = None,
        method: str = None,
        headers: dict = None,
        dashboard: str = "",
        force_external_address: bool = False,
        auth_info: AuthInfo = None,
    ):
        """Invoke the remote (live) function and return the results

        example::

            function.invoke("/api", body={"inputs": x})

        :param path:     request sub path (e.g. /images)
        :param body:     request body (str, bytes or a dict for json requests)
        :param method:   HTTP method (GET, PUT, ..)
        :param headers:  key/value dict with http headers
        :param dashboard: nuclio dashboard address
        :param force_external_address:   use the external ingress URL
        :param auth_info: service AuthInfo
        """
        if not method:
            method = "POST" if body else "GET"
        if "://" not in path:
            if not self.status.address:
                state, _, _ = self._get_state(dashboard, auth_info=auth_info)
                if state != "ready" or not self.status.address:
                    raise ValueError(
                        "no function address or not ready, first run .deploy()"
                    )

            path = self._resolve_invocation_url(path, force_external_address)

        if headers is None:
            headers = {}

        # if function is scaled to zero, let the DLX know we want to wake it up
        full_function_name = get_fullname(
            self.metadata.name, self.metadata.project, self.metadata.tag
        )
        headers.setdefault("x-nuclio-target", full_function_name)
        kwargs = {}
        if body:
            if isinstance(body, (str, bytes)):
                kwargs["data"] = body
            else:
                kwargs["json"] = body
        try:
            logger.info("invoking function", method=method, path=path)
            resp = requests.request(method, path, headers=headers, **kwargs)
        except OSError as err:
            raise OSError(f"error: cannot run function at url {path}, {err}")
        if not resp.ok:
            raise RuntimeError(f"bad function response {resp.status_code}: {resp.text}")

        data = resp.content
        if resp.headers["content-type"] == "application/json":
            data = json.loads(data)
        return data

    def _pre_run_validations(self):
        if self.spec.function_kind != "mlrun":
            raise RunError(
                '.run() can only be execute on "mlrun" kind'
                ', recreate with function kind "mlrun"'
            )

        state = self.status.state
        if state != "ready":
            if state:
                raise RunError(f"cannot run, function in state {state}")
            state, _, _ = self._get_state(raise_on_exception=True)
            if state != "ready":
                logger.info("starting nuclio build!")
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
            logger.error(f"error invoking function: {err}")
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
                runs.append(asyncio.ensure_future(resp,))

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
                            f"reached early stop condition ({generator.options.stop_condition}), stopping iterations!"
                        )
                        break

                if num_errors > generator.max_errors:
                    logger.error("max errors reached, stopping iterations!")
                    stop = True
                    break

        if stop:
            for task in runs:
                task.cancel()
        return results

    def _resolve_invocation_url(self, path, force_external_address):

        if path.startswith("/"):
            path = path[1:]

        # internal / external invocation urls is a nuclio >= 1.6.x feature
        # try to infer the invocation url from the internal and if not exists, use external.
        # $$$$ we do not want to use the external invocation url (e.g.: ingress, nodePort, etc)
        if (
            not force_external_address
            and self.status.internal_invocation_urls
            and get_k8s_helper(
                silent=True, log=False
            ).is_running_inside_kubernetes_cluster()
        ):
            return f"http://{self.status.internal_invocation_urls[0]}/{path}"

        if self.status.external_invocation_urls:
            return f"http://{self.status.external_invocation_urls[0]}/{path}"
        else:
            return f"http://{self.status.address}/{path}"


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


def _fullname(project, name):
    if project:
        return f"{project}-{name}"
    return name


def get_fullname(name, project, tag):
    if project:
        name = f"{project}-{name}"
    if tag:
        name = f"{name}-{tag}"
    return name


def deploy_nuclio_function(
    function: RemoteRuntime, dashboard="", watch=False, auth_info: AuthInfo = None
):
    dashboard = dashboard or mlconf.nuclio_dashboard_url
    function_name, project_name, function_config = compile_function_config(function)

    # if mode allows it, enrich function http trigger with an ingress
    enrich_function_with_ingress(
        function_config,
        mlconf.httpdb.nuclio.add_templated_ingress_host_mode,
        mlconf.httpdb.nuclio.default_service_type,
    )

    return nuclio.deploy.deploy_config(
        function_config,
        dashboard_url=dashboard,
        name=function_name,
        project=project_name,
        tag=function.metadata.tag,
        verbose=function.verbose,
        create_new=True,
        watch=watch,
        return_address_mode=nuclio.deploy.ReturnAddressModes.all,
        auth_info=auth_info.to_nuclio_auth_info() if auth_info else None,
    )


def resolve_function_ingresses(function_spec):
    http_trigger = resolve_function_http_trigger(function_spec)
    if not http_trigger:
        return []

    ingresses = []
    for _, ingress_config in (
        http_trigger.get("attributes", {}).get("ingresses", {}).items()
    ):
        ingresses.append(ingress_config)
    return ingresses


def resolve_function_http_trigger(function_spec):
    for trigger_name, trigger_config in function_spec.get("triggers", {}).items():
        if trigger_config.get("kind") != "http":
            continue
        return trigger_config


def compile_function_config(function: RemoteRuntime):
    labels = function.metadata.labels or {}
    labels.update({"mlrun/class": function.kind})
    for key, value in labels.items():
        function.set_config(f"metadata.labels.{key}", value)

    # Add vault configurations to function's pod spec, if vault secret source was added.
    # Needs to be here, since it adds env params, which are handled in the next lines.
    function.add_secrets_config_to_spec()

    env_dict, external_source_env_dict = function.get_nuclio_config_spec_env()
    spec = nuclio.ConfigSpec(
        env=env_dict,
        external_source_env=external_source_env_dict,
        config=function.spec.config,
    )
    spec.cmd = function.spec.build.commands or []
    project = function.metadata.project or "default"
    tag = function.metadata.tag
    handler = function.spec.function_handler

    # In Nuclio >= 1.6.x default serviceType has changed to "ClusterIP".
    spec.set_config("spec.serviceType", mlconf.httpdb.nuclio.default_service_type)
    if function.spec.readiness_timeout:
        spec.set_config("spec.readinessTimeoutSeconds", function.spec.readiness_timeout)
    if function.spec.resources:
        spec.set_config("spec.resources", function.spec.resources)
    if function.spec.no_cache:
        spec.set_config("spec.build.noCache", True)
    if function.spec.build.functionSourceCode:
        spec.set_config(
            "spec.build.functionSourceCode", function.spec.build.functionSourceCode
        )
    # don't send node selections if nuclio is not compatible
    if validate_nuclio_version_compatibility("1.5.20", "1.6.10"):
        if function.spec.node_selector:
            spec.set_config("spec.nodeSelector", function.spec.node_selector)
        if function.spec.node_name:
            spec.set_config("spec.nodeName", function.spec.node_name)
        if function.spec.affinity:
            spec.set_config("spec.affinity", function.spec._get_sanitized_affinity())
    # don't send default or any priority class name if nuclio is not compatible
    if (
        function.spec.priority_class_name
        and validate_nuclio_version_compatibility("1.6.18")
        and len(mlconf.get_valid_function_priority_class_names())
    ):
        spec.set_config("spec.priorityClassName", function.spec.priority_class_name)

    if function.spec.replicas:
        spec.set_config("spec.minReplicas", function.spec.replicas)
        spec.set_config("spec.maxReplicas", function.spec.replicas)
    else:
        spec.set_config("spec.minReplicas", function.spec.min_replicas)
        spec.set_config("spec.maxReplicas", function.spec.max_replicas)

    if function.spec.service_account:
        spec.set_config("spec.serviceAccount", function.spec.service_account)

    if function.spec.base_spec or function.spec.build.functionSourceCode:
        config = function.spec.base_spec
        if not config:
            # if base_spec was not set (when not using code_to_function) and we have base64 code
            # we create the base spec with essential attributes
            config = nuclio.config.new_config()
            update_in(config, "spec.handler", handler or "main:handler")

        config = nuclio.config.extend_config(
            config, spec, tag, function.spec.build.code_origin
        )
        update_in(config, "metadata.name", function.metadata.name)
        update_in(config, "spec.volumes", function.spec.generate_nuclio_volumes())
        base_image = (
            get_in(config, "spec.build.baseImage")
            or function.spec.image
            or function.spec.build.base_image
        )
        if base_image:
            update_in(config, "spec.build.baseImage", enrich_image_url(base_image))

        logger.info("deploy started")
        name = get_fullname(function.metadata.name, project, tag)
        function.status.nuclio_name = name
        update_in(config, "metadata.name", name)
    else:

        name, config, code = nuclio.build_file(
            function.spec.source,
            name=function.metadata.name,
            project=project,
            handler=handler,
            tag=tag,
            spec=spec,
            kind=function.spec.function_kind,
            verbose=function.verbose,
        )

        update_in(config, "spec.volumes", function.spec.generate_nuclio_volumes())
        base_image = function.spec.image or function.spec.build.base_image
        if base_image:
            update_in(
                config, "spec.build.baseImage", enrich_image_url(base_image),
            )

        name = get_fullname(name, project, tag)
        function.status.nuclio_name = name

        update_in(config, "metadata.name", name)

    return name, project, config


def enrich_function_with_ingress(config, mode, service_type):
    # do not enrich with an ingress
    if mode == NuclioIngressAddTemplatedIngressModes.never:
        return

    ingresses = resolve_function_ingresses(config["spec"])

    # function has ingresses already, nothing to add / enrich
    if ingresses:
        return

    # if exists, get the http trigger the function has
    # we would enrich it with an ingress
    http_trigger = resolve_function_http_trigger(config["spec"])
    if not http_trigger:
        # function has an HTTP trigger without an ingress
        # TODO: read from nuclio-api frontend-spec
        http_trigger = {
            "kind": "http",
            "name": "http",
            "maxWorkers": 1,
            "workerAvailabilityTimeoutMilliseconds": 10000,  # 10 seconds
            "attributes": {},
        }

    def enrich():
        http_trigger.setdefault("attributes", {}).setdefault("ingresses", {})["0"] = {
            "paths": ["/"],
            # this would tell Nuclio to use its default ingress host template
            # and would auto assign a host for the ingress
            "hostTemplate": "@nuclio.fromDefault",
        }
        http_trigger["attributes"]["serviceType"] = service_type
        config["spec"].setdefault("triggers", {})[http_trigger["name"]] = http_trigger

    if mode == NuclioIngressAddTemplatedIngressModes.always:
        enrich()
    elif mode == NuclioIngressAddTemplatedIngressModes.on_cluster_ip:

        # service type is not cluster ip, bail out
        if service_type and service_type.lower() != "clusterip":
            return

        enrich()


def get_nuclio_deploy_status(
    name,
    project,
    tag,
    dashboard="",
    last_log_timestamp=0,
    verbose=False,
    resolve_address=True,
    auth_info: AuthInfo = None,
):
    api_address = find_dashboard_url(dashboard or mlconf.nuclio_dashboard_url)
    name = get_fullname(name, project, tag)

    state, address, last_log_timestamp, outputs, function_status = get_deploy_status(
        api_address,
        name,
        last_log_timestamp,
        verbose,
        resolve_address,
        return_function_status=True,
        auth_info=auth_info.to_nuclio_auth_info() if auth_info else None,
    )

    text = "\n".join(outputs) if outputs else ""
    return state, address, name, last_log_timestamp, text, function_status
