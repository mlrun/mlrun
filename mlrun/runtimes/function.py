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
from datetime import datetime
from os import environ, getenv
from time import sleep

import nuclio
import requests
from aiohttp.client import ClientSession
from kubernetes import client
from nuclio.deploy import deploy_config, find_dashboard_url, get_deploy_status
from nuclio.triggers import V3IOStreamTrigger

import mlrun.errors
from mlrun.datastore import parse_s3_bucket_and_key
from mlrun.db import RunDBError

from ..config import config as mlconf
from ..kfpops import deploy_op
from ..lists import RunList
from ..model import RunObject
from ..platforms.iguazio import mount_v3io, parse_v3io_path, split_path
from ..utils import enrich_image_url, get_in, logger, update_in
from .base import FunctionStatus, RunError
from .pod import KubeResource, KubeResourceSpec
from .utils import get_item_name, log_std

default_max_replicas = 4


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
    def __init__(self, state=None, nuclio_name=None, address=None):
        super().__init__(state)

        self.nuclio_name = nuclio_name
        self.address = address


class RemoteRuntime(KubeResource):
    kind = "remote"

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
        if hasattr(spec, "to_dict"):
            spec = spec.to_dict()
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
        for key in ["V3IO_FRAMESD", "V3IO_USERNAME", "V3IO_ACCESS_KEY", "V3IO_API"]:
            if key in environ:
                self.set_env(key, environ[key])
        if local and remote:
            self.apply(mount_v3io(remote=remote, mount_path=local))
        return self

    def with_http(
        self, workers=8, port=0, host=None, paths=None, canary=None, secret=None
    ):
        self.add_trigger(
            "http",
            nuclio.HttpTrigger(
                workers, port=port, host=host, paths=paths, canary=canary, secret=secret
            ),
        )
        return self

    def add_model(self, name, model_path, **kw):
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
        self, stream_path, name="stream", group="serving", seek_to="earliest", shards=1,
    ):
        """add v3io stream trigger to the function"""
        endpoint = None
        if "://" in stream_path:
            endpoint, stream_path = parse_v3io_path(stream_path, suffix="")
        container, path = split_path(stream_path)
        shards = shards or 1
        self.add_trigger(
            name,
            V3IOStreamTrigger(
                name=name,
                container=container,
                path=path[1:],
                consumerGroup=group,
                seekTo=seek_to,
                webapi=endpoint or "http://v3io-webapi:8081",
            ),
        )
        self.spec.min_replicas = shards
        self.spec.max_replicas = shards

    def add_secrets_config_to_spec(self):
        # Currently secrets are only handled in Serving runtime.
        pass

    def deploy(
        self, dashboard="", project="", tag="", verbose=False,
    ):
        # todo: verify that the function name is normalized

        verbose = verbose or self.verbose
        if verbose:
            self.set_env("MLRUN_LOG_LEVEL", "DEBUG")
        if project:
            self.metadata.project = project
        if tag:
            self.metadata.tag = tag
        state = ""
        last_log_timestamp = 1

        if not dashboard:
            db = self._get_db()
            logger.info("Starting remote function deploy")
            data = db.remote_builder(self, False)
            self.status = data["data"].get("status")
            # ready = data.get("ready", False)

            while state not in ["ready", "error"]:
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
                logger.error("Nuclio function failed to deploy")
                raise RunError(f"cannot deploy {text}")

            if self.status.address:
                self.spec.command = f"http://{self.status.address}"
                self.save(versioned=False)

        else:
            self.save(versioned=False)
            self._ensure_run_db()
            address = deploy_nuclio_function(self, dashboard=dashboard, watch=True)
            if address:
                self.spec.command = f"http://{address}"
                self.status.state = "ready"
                self.status.address = address
                self.save(versioned=False)

        logger.info(f"function deployed, address={self.status.address}")
        return self.spec.command

    def with_node_selection(
        self,
        node_name: typing.Optional[str] = None,
        node_selector: typing.Optional[typing.Dict[str, str]] = None,
        affinity: typing.Optional[client.V1Affinity] = None,
    ):
        raise NotImplementedError("Node selection is not supported for nuclio runtime")

    def _get_state(
        self,
        dashboard="",
        last_log_timestamp=0,
        verbose=False,
        raise_on_exception=True,
    ):
        if dashboard:
            state, address, name, last_log_timestamp, text = get_nuclio_deploy_status(
                self.metadata.name,
                self.metadata.project,
                self.metadata.tag,
                dashboard,
                last_log_timestamp=last_log_timestamp,
                verbose=verbose,
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
        return runtime_env

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

    def invoke(self, path, body=None, method=None, headers=None, dashboard=""):
        if not method:
            method = "POST" if body else "GET"
        if "://" not in path:
            if not self.status.address:
                state, _, _ = self._get_state(dashboard)
                if state != "ready" or not self.status.address:
                    raise ValueError(
                        "no function address or not ready, first run .deploy()"
                    )
            if path.startswith("/"):
                path = path[1:]
            path = f"http://{self.status.address}/{path}"

        kwargs = {}
        if body:
            if isinstance(body, (str, bytes)):
                kwargs["data"] = body
            else:
                kwargs["json"] = body
        try:
            resp = requests.request(method, path, headers=headers, **kwargs)
        except OSError as err:
            raise OSError(f"error: cannot run function at url {path}, {err}")
        if not resp.ok:
            raise RuntimeError(f"bad function response {resp.text}")

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
            state = self._get_state(raise_on_exception=True)
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
            self._invoke_async(tasks, command, headers, secrets)
        )

        loop.run_until_complete(future)
        return future.result()

    def _update_state(self, rundict: dict):
        last_state = get_in(rundict, "status.state", "")
        if last_state != "error":
            update_in(rundict, "status.state", "completed")
        self._store_run_dict(rundict)
        return rundict

    async def _invoke_async(self, runs, url, headers, secrets):
        results = RunList()
        tasks = []

        async with ClientSession() as session:
            for run in runs:
                self.store_run(run)
                run.spec.secret_sources = secrets or []
                tasks.append(asyncio.ensure_future(submit(session, url, run, headers),))

            for status, resp, logs, run in await asyncio.gather(*tasks):

                if status != 200:
                    logger.error(f"failed to access {url} - {resp}")
                else:
                    results.append(self._update_state(json.loads(resp)))

                if logs:
                    log_std(self._db_conn, run, parse_logs(logs))

        return results


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


async def submit(session, url, run, headers=None):
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


def deploy_nuclio_function(function: RemoteRuntime, dashboard="", watch=False):
    function.set_config("metadata.labels.mlrun/class", function.kind)

    # Add vault configurations to function's pod spec, if vault secret source was added.
    # Needs to be here, since it adds env params, which are handled in the next lines.
    function.add_secrets_config_to_spec()

    env_dict = {get_item_name(v): get_item_name(v, "value") for v in function.spec.env}
    for key, value in function._get_runtime_env().items():
        env_dict[key] = value
    spec = nuclio.ConfigSpec(env=env_dict, config=function.spec.config)
    spec.cmd = function.spec.build.commands or []
    project = function.metadata.project or "default"
    tag = function.metadata.tag
    handler = function.spec.function_handler

    # In Nuclio 1.6.0 default serviceType changed to "ClusterIP", make sure we're using NodePort
    spec.set_config("spec.serviceType", "NodePort")
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

    if function.spec.replicas:
        spec.set_config("spec.minReplicas", function.spec.replicas)
        spec.set_config("spec.maxReplicas", function.spec.replicas)
    else:
        spec.set_config("spec.minReplicas", function.spec.min_replicas)
        spec.set_config("spec.maxReplicas", function.spec.max_replicas)

    dashboard = dashboard or mlconf.nuclio_dashboard_url
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
        base_image = get_in(config, "spec.build.baseImage") or function.spec.image
        if base_image:
            update_in(config, "spec.build.baseImage", enrich_image_url(base_image))

        logger.info("deploy started")
        name = get_fullname(function.metadata.name, project, tag)
        function.status.nuclio_name = name
        update_in(config, "metadata.name", name)
        return nuclio.deploy.deploy_config(
            config,
            dashboard,
            name=name,
            project=project,
            tag=tag,
            verbose=function.verbose,
            create_new=True,
            watch=watch,
        )
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
        if function.spec.image:
            update_in(
                config, "spec.build.baseImage", enrich_image_url(function.spec.image)
            )
        name = get_fullname(name, project, tag)
        function.status.nuclio_name = name

        update_in(config, "metadata.name", name)
        return deploy_config(
            config,
            dashboard_url=dashboard,
            name=name,
            project=project,
            tag=tag,
            verbose=function.verbose,
            create_new=True,
            watch=watch,
        )


def get_nuclio_deploy_status(
    name, project, tag, dashboard="", last_log_timestamp=0, verbose=False
):
    api_address = find_dashboard_url(dashboard or mlconf.nuclio_dashboard_url)
    name = get_fullname(name, project, tag)

    state, address, last_log_timestamp, outputs = get_deploy_status(
        api_address, name, last_log_timestamp, verbose
    )

    text = "\n".join(outputs) if outputs else ""
    return state, address, name, last_log_timestamp, text
