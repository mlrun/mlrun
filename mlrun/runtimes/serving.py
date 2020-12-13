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
from typing import List, Union
import mlrun
import nuclio

from ..model import ObjectList
from .function import RemoteRuntime, NuclioSpec, FunctionRef
from ..utils import logger, get_caller_globals
from ..serving.server import create_graph_server, GraphServer
from ..serving.states import (
    RouterState,
    StateKinds,
    RootFlowState,
    graph_root_setter,
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
        graph_initializer=None,
        error_stream=None,
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
        self.graph: RouterState = graph
        self.parameters = parameters or {}
        self.default_class = default_class
        self.load_mode = load_mode
        self._function_refs: ObjectList = None
        self.function_refs = function_refs or []
        self.graph_initializer = graph_initializer
        self.error_stream = error_stream

    @property
    def graph(self) -> RouterState:
        return self._graph

    @graph.setter
    def graph(self, graph):
        graph_root_setter(self, graph)

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
        start_at=None,
        engine=None,
        result_state=None,
        exist_ok=False,
        **class_args,
    ) -> Union[RootFlowState, RouterState]:
        """set the serving graph topology (router/flow) and root class or params

        e.g.: graph = fn.set_topology("flow", start_at="s1", engine="async", result_state="s5")

        topology can be:
          router - root router + multiple child route states/models
                   can specify special router class and router arguments
          flow   - workflow (DAG) with a chain of states, starting at "start_at" state
                   flow support "sync" and "async" engines, branches are note allowed in sync mode
                   when using async mode the optional result_state indicate the state which will
                   generate the (REST) call response

        :param topology:     - graph topology, router or flow
        :param class_name:   - optional for router, router class name/path
        :param start_at:     - for flow, starting state
        :param engine:       - optional for flow, sync or async engine (default to sync)
        :param result_state: - optional for flow, the state which returns the results
        :param exist_ok:     - allow overriding existing topology
        :param class_args:   - optional, router/flow class extra args

        :return graph object (fn.spec.graph)
        """
        topology = topology or StateKinds.router
        if self.spec.graph and not exist_ok:
            raise ValueError("graph topology is already set, cannot be overwritten")

        # currently we only support router topology
        if topology == StateKinds.router:
            self.spec.graph = RouterState(class_name=class_name, class_args=class_args)
        elif topology == StateKinds.flow:
            self.spec.graph = RootFlowState(
                start_at=start_at, engine=engine, result_state=result_state
            )
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
        :param class_args:  extra kwargs to pass to the model serving class __init__
                            (can be read in the model using .get_param(key) method)
        """
        graph = self.spec.graph
        if not graph:
            self.set_topology()

        if graph.kind == StateKinds.router:
            return graph.add_model(
                key,
                model_path,
                class_name,
                model_url=model_url,
                handler=handler,
                **class_args,
            )
        else:
            raise ValueError("models can only be added under router state")

    def add_child_function(
        self, name, url=None, image=None, requirements=None, kind=None
    ):
        """in a multi-function pipeline add child function"""
        function_ref = FunctionRef(
            url, image, requirements=requirements, kind=kind or "serving"
        )
        self._spec.function_refs.update(function_ref, name)
        func = function_ref.to_function()
        func.set_env("SERVING_CURRENT_FUNCTION", function_ref.name)
        return func

    def _add_ref_triggers(self):
        for function, stream in self.spec.graph.get_queue_links().items():
            if stream.path:
                if function not in self._spec.function_refs.keys():
                    raise ValueError(f"function reference {function} not present")
                child_function = self._spec.function_refs[function]
                group = stream.options.get("group", "serving")
                child_function.add_stream_trigger(
                    stream.path, group=group, shards=stream.shards or 1
                )

    def _deploy_function_refs(self):
        for function in self._spec.function_refs.values():
            logger.info(f"deploy child function {function.name} ...")
            function_object = function.function_object
            function_object.metadata.name = function.fullname(self)
            function_object.metadata.project = self.metadata.project
            function_object.metadata.tag = self.metadata.tag
            function_object.spec.graph = self.spec.graph
            function_object.apply(mlrun.v3io_cred())
            function.db_uri = function_object._function_uri()
            function_object.verbose = self.verbose
            function_object.deploy()

    def remove_states(self, keys: list):
        """remove one, multiple, or all models from the spec (blank list for all)"""
        if self.spec.graph:
            self.spec.graph.clear_children(keys)

    def deploy(self, dashboard="", project="", tag="", verbose=False):
        """deploy model serving function to a local/remote cluster

        :param dashboard: remote nuclio dashboard url (blank for local or auto detection)
        :param project:   optional, overide function specified project name
        :param tag:       specify unique function tag (a different function service is created for every tag)
        :param verbose:   verbose logging
        """
        load_mode = self.spec.load_mode
        if load_mode and load_mode not in ["sync", "async"]:
            raise ValueError(f"illegal model loading mode {load_mode}")
        if not self.spec.graph:
            raise ValueError("nothing to deploy, .spec.graph is none, use .add_model()")

        if self.spec.graph.kind != StateKinds.router:
            # initialize or create required streams/queues
            self.spec.graph.init_queues()
        if self._spec.function_refs:
            # deploy child functions
            self._add_ref_triggers()
            self._deploy_function_refs()
            logger.info(f"deploy root function {self.metadata.name} ...")
        return super().deploy(dashboard, project, tag, verbose=verbose)

    def _get_runtime_env(self):

        functions = {f.name: f.uri(self) for f in self.spec.function_refs}
        serving_spec = {
            "function_uri": self._function_uri(),
            "version": "v2",
            "parameters": self.spec.parameters,
            "graph": self.spec.graph.to_dict(),
            "load_mode": self.spec.load_mode,
            "functions": functions,
            "graph_initializer": self.spec.graph_initializer,
            "error_stream": self.spec.error_stream,
        }
        return {"SERVING_SPEC_ENV": json.dumps(serving_spec)}

    def to_mock_server(
        self, namespace=None, log_level="debug", current_function=None
    ) -> GraphServer:
        """create mock server object for local testing/emulation

        :param namespace: classes search namespace, use globals() for current notebook
        :param log_level: log level (error | info | debug)
        """
        return create_graph_server(
            parameters=self.spec.parameters,
            load_mode=self.spec.load_mode,
            graph=self.spec.graph,
            namespace=namespace or get_caller_globals(),
            logger=logger,
            level=log_level,
            current_function=current_function,
        )
