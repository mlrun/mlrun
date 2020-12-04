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
from typing import List

from nuclio.triggers import NuclioTrigger

import mlrun
import os

import nuclio
from ..platforms.iguazio import split_path
from ..model import ModelObj, ObjectList
from .function import RemoteRuntime, NuclioSpec
from ..utils import logger, get_caller_globals
from ..serving.server import create_mock_server
from ..serving.states import (
    ServingRouterState,
    new_remote_endpoint,
    new_model_endpoint,
    StateKinds,
    ServingRootFlowState,
    ServingTaskState,
    ServingQueueState,
)

serving_subkind = "serving_v2"


def new_v2_model_server(
    name,
    model_class: str,
    models: dict = None,
    filename="",
    protocol="",
    image="",
    endpoint="",
    workers=8,
    canary=None,
):
    f = ServingRuntime()
    if not image:
        name, spec, code = nuclio.build_file(
            filename, name=name, handler="handler", kind=serving_subkind
        )
        f.spec.base_spec = spec

    f.metadata.name = name
    f.spec.default_class = model_class
    params = None
    if protocol:
        params = {"protocol": protocol}
    if models:
        for name, model_path in models.items():
            f.add_model(name, model_path=model_path, parameters=params)

    f.with_http(workers, host=endpoint, canary=canary)
    if image:
        f.from_image(image)

    return f


class FunctionRef(ModelObj):
    def __init__(self, url=None, image=None, requirements=None, kind=None, name=None):
        self.url = url
        self.kind = kind
        self.image = image
        self.requirements = requirements
        self.name = name
        self.spec = None

        self._triggers = {}
        self._function = None
        self._address = None

    @property
    def function_object(self):
        return self._function

    def to_function(self):
        if self.url and "://" not in self.url:
            if not os.path.isfile(self.url):
                raise OSError("{} not found".format(self.url))

        kind = self.kind or "serving"
        if self.spec:
            func = mlrun.new_function(self.name, runtime=self.spec)
        elif (
            self.url.endswith(".yaml")
            or self.url.startswith("db://")
            or self.url.startswith("hub://")
        ):
            func = mlrun.import_function(self.url)
            if self.image:
                func.spec.image = self.image
        elif self.url.endswith(".ipynb"):
            func = mlrun.code_to_function(
                self.name, filename=self.url, image=self.image, kind=kind
            )
        elif self.url.endswith(".py"):
            if not self.image:
                raise ValueError(
                    "image must be provided with py code files, "
                    "use function object for more control/settings"
                )
            func = mlrun.code_to_function(
                self.name, filename=self.url, image=self.image, kind=kind
            )
        else:
            raise ValueError("unsupported function url {} or no spec".format(self.url))
        if self.requirements:
            commands = func.spec.build.commands or []
            commands.append("python -m pip install " + " ".join(self.requirements))
            func.spec.build.commands = commands
        func.set_env("SERVING_CURRENT_FUNCTION", self.name)
        self._function = func
        return func

    def add_stream_trigger(
        self, stream_path, name="stream", group="serving", seek_to="earliest"
    ):
        container, path = split_path(stream_path)
        self._function.add_trigger(
            name,
            V3IOStreamTrigger(
                name=name,
                container=container,
                path=path[1:],
                consumerGroup=group,
                seekTo=seek_to,
            ),
        )

    def deploy(self):
        self._address = self._function.deploy()


class ServingSpec(NuclioSpec):
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
        models=None,
        graph=None,
        parameters=None,
        default_class=None,
        load_mode=None,
        build=None,
        function_refs=None,
    ):

        super().__init__(
            command=command,
            args=args,
            image=image,
            mode=mode,
            entry_points=entry_points,
            description=description,
            replicas=replicas,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            volumes=volumes,
            volume_mounts=volume_mounts,
            env=env,
            resources=resources,
            config=config,
            base_spec=base_spec,
            no_cache=no_cache,
            source=source,
            image_pull_policy=image_pull_policy,
            function_kind=serving_subkind,
            service_account=service_account,
            readiness_timeout=readiness_timeout,
            build=build,
        )

        self.models = models or {}
        self._graph = None
        self.graph: ServingRouterState = graph
        self.parameters = parameters or {}
        self.default_class = default_class
        self.load_mode = load_mode
        self._function_refs: ObjectList = None
        self.function_refs = function_refs or []

    @property
    def graph(self) -> ServingRouterState:
        return self._graph

    @graph.setter
    def graph(self, graph):
        if graph:
            if isinstance(graph, dict):
                kind = graph.get("kind")
            elif hasattr(graph, "kind"):
                kind = graph.kind
            else:
                raise ValueError("graph must be a dict or a valid object")
            if kind == StateKinds.router:
                self._graph = self._verify_dict(graph, "graph", ServingRouterState)
            else:
                self._graph = self._verify_dict(graph, "graph", ServingRootFlowState)

    @property
    def function_refs(self) -> List[FunctionRef]:
        return self._function_refs

    @function_refs.setter
    def function_refs(self, function_refs: List[FunctionRef]):
        self._function_refs = ObjectList.from_list(FunctionRef, function_refs)


class ServingRuntime(RemoteRuntime):
    kind = "serving"

    @property
    def spec(self) -> ServingSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", ServingSpec)

    def set_topology(
        self,
        topology=None,
        class_name=None,
        exist_ok=False,
        start_at=None,
        engine=None,
        **class_args,
    ):
        """set the serving graph topology (router/flow/endpoint) and root class"""
        topology = topology or StateKinds.router
        if self.spec.graph and not exist_ok:
            raise ValueError("graph topology is already set, cannot be overwritten")

        # currently we only support router topology
        if topology == StateKinds.router:
            self.spec.graph = ServingRouterState(
                class_name=class_name, class_args=class_args
            )
        elif topology == StateKinds.flow:
            self.spec.graph = ServingRootFlowState(start_at=start_at, engine=engine)
        else:
            raise ValueError(f"unsupported topology {topology}, use 'router' or 'flow'")
        return self.spec.graph

    def set_tracking(self, stream_path, batch=None, sample=None):
        """set tracking log stream parameters"""
        self.spec.parameters["log_stream"] = stream_path
        if batch:
            self.spec.parameters["log_stream_batch"] = batch
        if sample:
            self.spec.parameters["log_stream_sample"] = sample

    def add_model(
        self,
        key,
        model_path=None,
        class_name=None,
        model_url=None,
        handler=None,
        parent=None,
        **class_args,
    ):
        """add ml model and/or route to the function

        Example, create a function (from the notebook), add a model class, and deploy:

            fn = code_to_function(kind='serving')
            fn.add_model('boost', model_path, model_class='MyClass', my_arg=5)
            fn.deploy()

        :param key:         model api key (or name:version), will determine the relative url/path
        :param model_path:  path to mlrun model artifact or model directory file/object path
        :param class_name:  V2 Model python class name
                            (can also module.submodule.class and it will be imported automatically)
        :param model_url:   url of a remote model serving endpoint (cannot be used with model_path)
        :param handler:     for advanced users!, override default class handler name (do_event)
        :param parent:      in hierarchical topology, state name of the parent
        :param class_args:  extra kwargs to pass to the model serving class __init__
                            (can be read in the model using .get_param(key) method)
        """
        if not model_path and not model_url:
            raise ValueError("model_path or model_url must be provided")
        class_name = class_name or self.spec.default_class
        if not isinstance(class_name, str):
            raise ValueError(
                "class name must be a string (name ot module.submodule.name)"
            )
        if model_path and not class_name:
            raise ValueError("model_path must be provided with class_name")
        if model_path:
            model_path = str(model_path)

        if not self.spec.graph:
            self.set_topology()

        if model_url:
            state = new_remote_endpoint(model_url, **class_args)
        else:
            state = new_model_endpoint(class_name, model_path, handler, **class_args)

        root = self.spec.graph
        if parent:
            root = root.path_to_state(parent)

        if root.kind == StateKinds.router:
            root.add_route(key, state)
        else:
            raise ValueError("models can only be added under router state")
        return state

    def add_state(
        self,
        key: str,
        class_name: str = None,
        handler: str = None,
        after: str = None,
        parent: str = None,
        kind: str = None,
        end: bool = None,
        function=None,
        **class_args,
    ):
        """add compute class or model or route to the function

        Example, create a function (from the notebook), add a model class, and deploy:

            fn = code_to_function(kind='serving')
            fn.add_model('boost', model_path, model_class='MyClass', my_arg=5)
            fn.deploy()

        :param key:         model api key (or name:version), will determine the relative url/path
        :param class_name:  V2 Model python class name
                            (can also module.submodule.class and it will be imported automatically)
        :param handler:     for advanced users!, override default class handler name (do_event)
        :param after:       for flow topology, the step name this will come after
        :param parent:      in hierarchical topology, state the parent name
        :param kind:        state kind, task or router (default is task)
        :param end:         mark the state as final/result state, only one state can have end=True
        :param class_args:  extra kwargs to pass to the model serving class __init__
                            (can be read in the model using .get_param(key) method)
        """
        if class_name and not isinstance(class_name, str):
            raise ValueError(
                "class name must be a string (name ot module.submodule.name)"
            )

        if not self.spec.graph:
            self.set_topology()

        if kind == StateKinds.router:
            state = ServingRouterState(
                class_name=class_name, class_args=class_args, end=end, function=function
            )
        elif kind == StateKinds.queue:
            state = ServingQueueState(**class_args)
        else:
            state = ServingTaskState(
                class_name, class_args, handler=handler, end=end, function=function
            )

        root = self.spec.graph
        if parent:
            root = root.path_to_state(parent)

        if root.kind == StateKinds.router:
            root.add_route(key, state)
        else:
            root.add_state(key, state, after=after)
        return state

    def add_function(self, name, url=None, image=None, requirements=None, kind=None):
        function_ref = FunctionRef(
            url, image, requirements=requirements, kind=kind or "serving"
        )
        self._spec.function_refs.update(function_ref, name)
        return function_ref.to_function()

    def add_ref_triggers(self, group=None):
        for function, stream in self.spec.graph.get_queue_links().items():
            if stream:
                if function not in self._spec.function_refs.keys():
                    raise ValueError(f"function reference {function} not present")
                self._spec.function_refs[function].add_stream_trigger(
                    stream, group=group or "serving"
                )

    def _deploy_function_refs(self):
        for function in self._spec.function_refs.values():
            logger.info(f"deploy child function {function.name} ...")
            function_object = function.function_object
            function_object.metadata.name = f'{self.metadata.name}-{function.name}'
            function_object.metadata.project = self.metadata.project
            function_object.spec.graph = self.spec.graph
            function_object.apply(mlrun.v3io_cred())
            function_object.deploy()

    def remove_states(self, keys: list):
        """remove one, multiple, or all models from the spec (blank list for all)"""
        if self.spec.graph:
            self.spec.graph.clear_children(keys)

    def deploy(self, dashboard="", project="", tag="", stream_group=None):
        """deploy model serving function to a local/remote cluster

        :param dashboard: remote nuclio dashboard url (blank for local or auto detection)
        :param project:   optional, overide function specified project name
        :param tag:       specify unique function tag (a different function service is created for every tag)
        """
        load_mode = self.spec.load_mode
        if load_mode and load_mode not in ["sync", "async"]:
            raise ValueError(f"illegal model loading mode {load_mode}")
        if not self.spec.graph:
            raise ValueError("nothing to deploy, .spec.graph is none, use .add_model()")

        if self._spec.function_refs:
            self.add_ref_triggers(group=stream_group)
            self._deploy_function_refs()
            logger.info(f"deploy root function {self.metadata.name} ...")
        return super().deploy(dashboard, project, tag)

    def _get_runtime_env(self):
        # we currently support a minimal topology of one router + multiple child routes/models
        # in the future we will extend the support to a full graph, the spec is already built accordingly
        serving_spec = {
            "function_uri": self._function_uri(),
            "version": "v2",
            "parameters": self.spec.parameters,
            "graph": self.spec.graph.to_dict(),
            "load_mode": self.spec.load_mode,
            # "functions": self.spec.function_refs.to_dict(),
            "verbose": self.verbose,
        }
        return {"SERVING_SPEC_ENV": json.dumps(serving_spec)}

    def to_mock_server(self, namespace=None, log_level="debug", current_function=None):
        """create mock server object for local testing/emulation

        :param namespace: classes search namespace, use globals() for current notebook
        :param log_level: log level (error | info | debug)
        """
        return create_mock_server(
            parameters=self.spec.parameters,
            load_mode=self.spec.load_mode,
            graph=self.spec.graph,
            namespace=namespace or get_caller_globals(),
            logger=logger,
            level=log_level,
            current_function=current_function,
        )

    def plot(self, filename=None, format=None):
        if not self.spec.graph:
            logger.info("no graph topology to plot")
            return
        return self.spec.graph.plot(filename, format)


class V3IOStreamTrigger(NuclioTrigger):
    kind = "v3ioStream"

    def __init__(
        self,
        url: str = None,
        seekTo: str = "latest",
        partitions: list = None,
        pollingIntervalMS: int = 500,
        readBatchSize: int = 64,
        maxWorkers: int = 1,
        access_key: str = None,
        sessionTimeout: str = "10s",
        name: str = "streamtrigger",
        container: str = None,
        path: str = None,
        workerAllocationMode: str = "pool",
        webapi: str = "http://v3io-webapi:8081",
        consumerGroup: str = "default",
        sequenceNumberCommitInterval: str = "1s",
        heartbeatInterval: str = "3s",
    ):

        if url and not container and not path:
            self._struct = {"kind": self.kind, "url": url, "attributes": {}}
        else:
            self._struct = {
                "kind": self.kind,
                "url": webapi,
                "name": name,
                "attributes": {
                    "containerName": container,
                    "streamPath": path,
                    "consumerGroup": consumerGroup,
                    "sequenceNumberCommitInterval": sequenceNumberCommitInterval,
                    "workerAllocationMode": workerAllocationMode,
                    "sessionTimeout": sessionTimeout,
                    "heartbeatInterval": heartbeatInterval,
                },
            }

        if maxWorkers:
            self._struct["maxWorkers"] = maxWorkers
        if seekTo:
            self._struct["attributes"]["seekTo"] = seekTo
        if readBatchSize:
            self._struct["attributes"]["readBatchSize"] = readBatchSize
        if partitions:
            self._struct["attributes"]["partitions"] = partitions
        if pollingIntervalMS:
            self._struct["attributes"]["pollingIntervalMs"] = pollingIntervalMS
        access_key = access_key if access_key else os.environ["V3IO_ACCESS_KEY"]
        self._struct["password"] = access_key
