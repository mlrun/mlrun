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

import json
from os import environ
from time import sleep

import requests
from datetime import datetime
import asyncio
from aiohttp.client import ClientSession
from mlrun.db import RunDBError
from nuclio.deploy import deploy_config, get_deploy_status, find_dashboard_url
import nuclio

from .pod import KubeResourceSpec, KubeResource
from ..kfpops import deploy_op
from ..platforms.iguazio import mount_v3io
from .base import RunError, FunctionStatus
from .utils import log_std, set_named_item, get_item_name
from ..utils import logger, update_in, get_in, enrich_image_url
from ..lists import RunList
from ..model import RunObject
from ..config import config as mlconf

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

    @property
    def volumes(self) -> list:
        return list(self._volumes.values())

    @volumes.setter
    def volumes(self, volumes):
        self._volumes = {}
        if volumes:
            for vol in volumes:
                set_named_item(self._volumes, vol)

    @property
    def volume_mounts(self) -> list:
        return list(self._volume_mounts.values())

    @volume_mounts.setter
    def volume_mounts(self, volume_mounts):
        self._volume_mounts = {}
        if volume_mounts:
            for vol in volume_mounts:
                set_named_item(self._volume_mounts, vol)

    def update_vols_and_mounts(self, volumes, volume_mounts):
        if volumes:
            for vol in volumes:
                set_named_item(self._volumes, vol)

        if volume_mounts:
            for vol in volume_mounts:
                set_named_item(self._volume_mounts, vol)

    def to_nuclio_vol(self):
        vols = []
        for name, vol in self._volumes.items():
            if name not in self._volume_mounts:
                raise ValueError(
                    "found volume without a volume mount ({})".format(name)
                )
            vols.append({"volume": vol, "volumeMount": self._volume_mounts[name]})
        return vols


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
        self.spec.config["spec.triggers.{}".format(name)] = spec
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
        self.set_env("SERVING_MODEL_{}".format(name), model)
        return self

    def from_image(self, image):
        config = nuclio.config.new_config()
        update_in(
            config,
            "spec.handler",
            self.spec.function_handler or "main:{}".format("handler"),
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
                self.set_env("SERVING_MODEL_{}".format(k), v)

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

    def deploy(
        self, dashboard="", project="", tag="", verbose=False,
    ):
        verbose = verbose or self.verbose
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
                self.spec.command = "http://{}".format(self.status.address)
                self.save(versioned=False)

        else:
            self.save(versioned=False)
            self._ensure_run_db()
            address = deploy_nuclio_function(self, dashboard=dashboard, watch=True)
            if address:
                self.spec.command = "http://{}".format(address)
                self.status.state = "ready"
                self.status.address = address
                self.save(versioned=False)

        logger.info(f"function deployed, address={self.status.address}")
        return self.spec.command

    def _get_state(
        self,
        dashboard="",
        last_log_timestamp=None,
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
                self.spec.command = "http://{}".format(address)
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

    def _get_runtime_env(self):
        # for runtime specific env var enrichment (before deploy)
        runtime_env = {}
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
        name = "deploy_{}".format(self.metadata.name or "function")
        project = project or self.metadata.project
        if models and isinstance(models, dict):
            models = [{"key": k, "model_path": v} for k, v in models.items()]

        if use_function_from_db:
            hash_key = self.save(versioned=True, refresh=True)
            url = "db://" + self._function_uri(hash_key=hash_key)
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
            command = "{}/{}".format(command, runobj.spec.handler_name)
        headers = {"x-nuclio-log-level": log_level}
        try:
            resp = requests.put(command, json=runobj.to_dict(), headers=headers)
        except OSError as err:
            logger.error("error invoking function: {}".format(err))
            raise OSError("error: cannot run function at url {}".format(command))

        if not resp.ok:
            logger.error("bad function resp!!\n{}".format(resp.text))
            raise RunError("bad function response")

        logs = resp.headers.get("X-Nuclio-Logs")
        if logs:
            log_std(self._db_conn, runobj, parse_logs(logs))

        return self._update_state(resp.json())

    def _run_many(self, tasks, execution, runobj: RunObject):
        self._pre_run_validations()
        secrets = self._secrets.to_serial() if self._secrets else None
        log_level = execution.log_level
        headers = {"x-nuclio-log-level": log_level}

        command = self.spec.command
        if runobj.spec.handler:
            command = "{}/{}".format(command, runobj.spec.handler_name)
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
                    logger.error("failed to access {} - {}".format(url, resp))
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
        for k, v in line.items():
            if k not in ["time", "level", "name", "message"]:
                extra.append("{}={}".format(k, v))
        line["extra"] = ", ".join(extra)
        line["time"] = datetime.fromtimestamp(float(line["time"]) / 1000).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )
        lines += "{time}  {level:<6} {message}  {extra}\n".format(**line)

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
        return "{}-{}".format(project, name)
    return name


def get_fullname(name, project, tag):
    if project:
        name = "{}-{}".format(project, name)
    if tag:
        name = "{}-{}".format(name, tag)
    return name


def deploy_nuclio_function(function: RemoteRuntime, dashboard="", watch=False):
    function.set_config("metadata.labels.mlrun/class", function.kind)
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
    if function.spec.replicas:
        spec.set_config("spec.minReplicas", function.spec.replicas)
        spec.set_config("spec.maxReplicas", function.spec.replicas)
    else:
        spec.set_config("spec.minReplicas", function.spec.min_replicas)
        spec.set_config("spec.maxReplicas", function.spec.max_replicas)

    dashboard = dashboard or mlconf.nuclio_dashboard_url
    if function.spec.base_spec:
        config = nuclio.config.extend_config(
            function.spec.base_spec, spec, tag, function.spec.build.code_origin
        )
        update_in(config, "metadata.name", function.metadata.name)
        update_in(config, "spec.volumes", function.spec.to_nuclio_vol())
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

        update_in(config, "spec.volumes", function.spec.to_nuclio_vol())
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
    name, project, tag, dashboard="", last_log_timestamp=None, verbose=False
):
    api_address = find_dashboard_url(dashboard or mlconf.nuclio_dashboard_url)
    name = get_fullname(name, project, tag)

    state, address, last_log_timestamp, outputs = get_deploy_status(
        api_address, name, last_log_timestamp, verbose
    )

    text = "\n".join(outputs) if outputs else ""
    return state, address, name, last_log_timestamp, text
