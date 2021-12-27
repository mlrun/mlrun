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
from copy import deepcopy
from typing import List, Union

import nuclio

import mlrun
import mlrun.api.schemas

from ..model import ObjectList
from ..secrets import SecretsStore
from ..serving.server import GraphServer, create_graph_server
from ..serving.states import (
    RootFlowStep,
    RouterStep,
    StepKinds,
    TaskStep,
    graph_root_setter,
    new_remote_endpoint,
    params_to_step,
)
from ..utils import get_caller_globals, logger
from .function import NuclioSpec, RemoteRuntime
from .function_reference import FunctionReference

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
        track_models=None,
        secret_sources=None,
        default_content_type=None,
        node_name=None,
        node_selector=None,
        affinity=None,
        disable_auto_mount=False,
        priority_class_name=None,
        default_handler=None,
        pythonpath=None,
        workdir=None,
        image_pull_secret=None,
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
            node_name=node_name,
            node_selector=node_selector,
            affinity=affinity,
            disable_auto_mount=disable_auto_mount,
            priority_class_name=priority_class_name,
            default_handler=default_handler,
            pythonpath=pythonpath,
            workdir=workdir,
            image_pull_secret=image_pull_secret,
        )

        self.models = models or {}
        self._graph = None
        self.graph: Union[RouterStep, RootFlowStep] = graph
        self.parameters = parameters or {}
        self.default_class = default_class
        self.load_mode = load_mode
        self._function_refs: ObjectList = None
        self.function_refs = function_refs or []
        self.graph_initializer = graph_initializer
        self.error_stream = error_stream
        self.track_models = track_models
        self.secret_sources = secret_sources or []
        self.default_content_type = default_content_type

    @property
    def graph(self) -> Union[RouterStep, RootFlowStep]:
        """states graph, holding the serving workflow/DAG topology"""
        return self._graph

    @graph.setter
    def graph(self, graph):
        graph_root_setter(self, graph)

    @property
    def function_refs(self) -> List[FunctionReference]:
        """function references, list of optional child function refs"""
        return self._function_refs

    @function_refs.setter
    def function_refs(self, function_refs: List[FunctionReference]):
        self._function_refs = ObjectList.from_list(FunctionReference, function_refs)


class ServingRuntime(RemoteRuntime):
    """MLRun Serving Runtime"""

    kind = "serving"

    @property
    def spec(self) -> ServingSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", ServingSpec)

    def set_topology(
        self, topology=None, class_name=None, engine=None, exist_ok=False, **class_args,
    ) -> Union[RootFlowStep, RouterStep]:
        """set the serving graph topology (router/flow) and root class or params

        examples::

            # simple model router topology
            graph = fn.set_topology("router")
            fn.add_model(name, class_name="ClassifierModel", model_path=model_uri)

            # async flow topology
            graph = fn.set_topology("flow", engine="async")
            graph.to("MyClass").to(name="to_json", handler="json.dumps").respond()

        topology options are::

          router - root router + multiple child route states/models
                   route is usually determined by the path (route key/name)
                   can specify special router class and router arguments

          flow   - workflow (DAG) with a chain of states
                   flow support "sync" and "async" engines, branches are not allowed in sync mode
                   when using async mode calling state.respond() will mark the state as the
                   one which generates the (REST) call response

        :param topology:     - graph topology, router or flow
        :param class_name:   - optional for router, router class name/path or router object
        :param engine:       - optional for flow, sync or async engine (default to async)
        :param exist_ok:     - allow overriding existing topology
        :param class_args:   - optional, router/flow class init args

        :return graph object (fn.spec.graph)
        """
        topology = topology or StepKinds.router
        if self.spec.graph and not exist_ok:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "graph topology is already set, cannot be overwritten"
            )

        if topology == StepKinds.router:
            if class_name and hasattr(class_name, "to_dict"):
                _, step = params_to_step(class_name, None)
                if step.kind != StepKinds.router:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        "provided class is not a router step, must provide a router class in router topology"
                    )
            else:
                step = RouterStep(class_name=class_name, class_args=class_args)
            self.spec.graph = step
        elif topology == StepKinds.flow:
            self.spec.graph = RootFlowStep(engine=engine)
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"unsupported topology {topology}, use 'router' or 'flow'"
            )
        return self.spec.graph

    def set_tracking(
        self,
        stream_path: str = None,
        batch: int = None,
        sample: int = None,
        stream_args: dict = None,
    ):
        """set tracking stream parameters:

        :param stream_path:  path/url of the tracking stream e.g. v3io:///users/mike/mystream
                             you can use the "dummy://" path for test/simulation
        :param batch:        micro batch size (send micro batches of N records at a time)
        :param sample:       sample size (send only one of N records)
        :param stream_args:  stream initialization parameters, e.g. shards, retention_in_hours, ..
        """
        self.spec.track_models = True
        if stream_path:
            self.spec.parameters["log_stream"] = stream_path
        if batch:
            self.spec.parameters["log_stream_batch"] = batch
        if sample:
            self.spec.parameters["log_stream_sample"] = sample
        if stream_args:
            self.spec.parameters["stream_args"] = stream_args

    def add_model(
        self,
        key,
        model_path=None,
        class_name=None,
        model_url=None,
        handler=None,
        router_step=None,
        child_function=None,
        **class_args,
    ):
        """add ml model and/or route to the function.

        Example, create a function (from the notebook), add a model class, and deploy::

            fn = code_to_function(kind='serving')
            fn.add_model('boost', model_path, model_class='MyClass', my_arg=5)
            fn.deploy()

        only works with router topology, for nested topologies (model under router under flow)
        need to add router to flow and use router.add_route()

        :param key:         model api key (or name:version), will determine the relative url/path
        :param model_path:  path to mlrun model artifact or model directory file/object path
        :param class_name:  V2 Model python class name or a model class instance
                            (can also module.submodule.class and it will be imported automatically)
        :param model_url:   url of a remote model serving endpoint (cannot be used with model_path)
        :param handler:     for advanced users!, override default class handler name (do_event)
        :param router_step: router step name (to determine which router we add the model to in graphs
                            with multiple router steps)
        :param child_function: child function name, when the model runs in a child function
        :param class_args:  extra kwargs to pass to the model serving class __init__
                            (can be read in the model using .get_param(key) method)
        """
        graph = self.spec.graph
        if not graph:
            graph = self.set_topology()

        if graph.kind != StepKinds.router:
            if router_step:
                if router_step not in graph:
                    raise ValueError(
                        f"router step {router_step} not present in the graph"
                    )
                graph = graph[router_step]
            else:
                routers = [
                    step
                    for step in graph.steps.values()
                    if step.kind == StepKinds.router
                ]
                if len(routers) == 0:
                    raise ValueError(
                        "graph does not contain any router, add_model can only be "
                        "used when there is a router step"
                    )
                if len(routers) > 1:
                    raise ValueError(
                        f"found {len(routers)} routers, please specify the router_step"
                        " you would like to add this model to"
                    )
                graph = routers[0]

        if class_name and hasattr(class_name, "to_dict"):
            if model_path:
                class_name.model_path = model_path
            key, state = params_to_step(class_name, key)
        else:
            if not model_path and not model_url:
                raise ValueError("model_path or model_url must be provided")
            class_name = class_name or self.spec.default_class
            if class_name and not isinstance(class_name, str):
                raise ValueError(
                    "class name must be a string (name of module.submodule.name)"
                )
            if model_path and not class_name:
                raise ValueError("model_path must be provided with class_name")
            if model_path:
                model_path = str(model_path)

            if model_url:
                state = new_remote_endpoint(model_url, **class_args)
            else:
                class_args = deepcopy(class_args)
                class_args["model_path"] = model_path
                state = TaskStep(
                    class_name, class_args, handler=handler, function=child_function
                )

        return graph.add_route(key, state)

    def add_child_function(
        self, name, url=None, image=None, requirements=None, kind=None
    ):
        """in a multi-function pipeline add child function

        example::

            fn.add_child_function('enrich', './enrich.ipynb', 'mlrun/mlrun')

        :param name:   child function name
        :param url:    function/code url, support .py, .ipynb, .yaml extensions
        :param image:  base docker image for the function
        :param requirements: py package requirements file path OR list of packages
        :param kind:   mlrun function/runtime kind

        :return function object
        """
        function_reference = FunctionReference(
            url, image, requirements=requirements, kind=kind or "serving"
        )
        self._spec.function_refs.update(function_reference, name)
        func = function_reference.to_function(self.kind)
        return func

    def _add_ref_triggers(self):
        """add stream trigger to downstream child functions"""
        for function_name, stream in self.spec.graph.get_queue_links().items():
            if stream.path:
                if function_name not in self._spec.function_refs.keys():
                    raise ValueError(f"function reference {function_name} not present")
                group = stream.options.get("group", "serving")

                child_function = self._spec.function_refs[function_name]
                trigger_args = stream.trigger_args or {}
                child_function.function_object.add_v3io_stream_trigger(
                    stream.path, group=group, shards=stream.shards, **trigger_args
                )

    def _deploy_function_refs(self):
        """set metadata and deploy child functions"""
        for function_ref in self._spec.function_refs.values():
            logger.info(f"deploy child function {function_ref.name} ...")
            function_object = function_ref.function_object
            if not function_object:
                function_object = function_ref.to_function(self.kind)
            function_object.metadata.name = function_ref.fullname(self)
            function_object.metadata.project = self.metadata.project
            function_object.metadata.tag = self.metadata.tag

            function_object.metadata.labels = function_object.metadata.labels or {}
            function_object.metadata.labels[
                "mlrun/parent-function"
            ] = self.metadata.name
            if not function_object.spec.graph:
                # copy the current graph only if the child doesnt have a graph of his own
                function_object.set_env("SERVING_CURRENT_FUNCTION", function_ref.name)
                function_object.spec.graph = self.spec.graph

            function_object.verbose = self.verbose
            function_object.spec.secret_sources = self.spec.secret_sources
            function_object.deploy()

    def remove_states(self, keys: list):
        """remove one, multiple, or all states/models from the spec (blank list for all)"""
        if self.spec.graph:
            self.spec.graph.clear_children(keys)

    def with_secrets(self, kind, source):
        """register a secrets source (file, env or dict)

        read secrets from a source provider to be used in workflows, example::

            task.with_secrets('file', 'file.txt')
            task.with_secrets('inline', {'key': 'val'})
            task.with_secrets('env', 'ENV1,ENV2')
            task.with_secrets('vault', ['secret1', 'secret2'...])

            # If using an empty secrets list [] then all accessible secrets will be available.
            task.with_secrets('vault', [])

            # To use with Azure key vault, a k8s secret must be created with the following keys:
            # kubectl -n <namespace> create secret generic azure-key-vault-secret \\
            #     --from-literal=tenant_id=<service principal tenant ID> \\
            #     --from-literal=client_id=<service principal client ID> \\
            #     --from-literal=secret=<service principal secret key>

            task.with_secrets('azure_vault', {
                'name': 'my-vault-name',
                'k8s_secret': 'azure-key-vault-secret',
                # An empty secrets list may be passed ('secrets': []) to access all vault secrets.
                'secrets': ['secret1', 'secret2'...]
            })

        :param kind:   secret type (file, inline, env)
        :param source: secret data or link (see example)

        :returns: The Runtime (function) object
        """

        if kind == "vault" and isinstance(source, list):
            source = {"project": self.metadata.project, "secrets": source}

        self.spec.secret_sources.append({"kind": kind, "source": source})
        return self

    def add_secrets_config_to_spec(self):
        if self.spec.secret_sources:
            self._secrets = SecretsStore.from_list(self.spec.secret_sources)
            if self._secrets.has_vault_source():
                self._add_vault_params_to_spec(project=self.metadata.project)
            if self._secrets.has_azure_vault_source():
                self._add_azure_vault_params_to_spec(
                    self._secrets.get_azure_vault_k8s_secret()
                )
            self._add_project_k8s_secrets_to_spec(
                self._secrets.get_k8s_secrets(), project=self.metadata.project
            )
        else:
            self._add_project_k8s_secrets_to_spec(None, project=self.metadata.project)

    def deploy(
        self,
        dashboard="",
        project="",
        tag="",
        verbose=False,
        auth_info: mlrun.api.schemas.AuthInfo = None,
    ):
        """deploy model serving function to a local/remote cluster

        :param dashboard: remote nuclio dashboard url (blank for local or auto detection)
        :param project:   optional, override function specified project name
        :param tag:       specify unique function tag (a different function service is created for every tag)
        :param verbose:   verbose logging
        :param auth_info: The auth info to use to communicate with the Nuclio dashboard, required only when providing
                          dashboard
        """
        load_mode = self.spec.load_mode
        if load_mode and load_mode not in ["sync", "async"]:
            raise ValueError(f"illegal model loading mode {load_mode}")
        if not self.spec.graph:
            raise ValueError("nothing to deploy, .spec.graph is none, use .add_model()")

        if self.spec.graph.kind != StepKinds.router:
            # initialize or create required streams/queues
            self.spec.graph.check_and_process_graph()
            self.spec.graph.init_queues()

        # Handle secret processing before handling child functions, since secrets are transferred to them
        if self.spec.secret_sources:
            # Before passing to remote builder, secrets values must be retrieved (for example from ENV)
            # and stored as inline secrets. Otherwise, they will not be available to the builder.
            self._secrets = SecretsStore.from_list(self.spec.secret_sources)
            self.spec.secret_sources = self._secrets.to_serial()

        if self._spec.function_refs:
            # deploy child functions
            self._add_ref_triggers()
            self._deploy_function_refs()
            logger.info(f"deploy root function {self.metadata.name} ...")

        return super().deploy(dashboard, project, tag, verbose, auth_info)

    def _get_runtime_env(self):
        env = super()._get_runtime_env()
        function_name_uri_map = {f.name: f.uri(self) for f in self.spec.function_refs}

        serving_spec = {
            "function_uri": self._function_uri(),
            "version": "v2",
            "parameters": self.spec.parameters,
            "graph": self.spec.graph.to_dict() if self.spec.graph else {},
            "load_mode": self.spec.load_mode,
            "functions": function_name_uri_map,
            "graph_initializer": self.spec.graph_initializer,
            "error_stream": self.spec.error_stream,
            "track_models": self.spec.track_models,
            "default_content_type": self.spec.default_content_type,
        }

        if self.spec.secret_sources:
            self._secrets = SecretsStore.from_list(self.spec.secret_sources)
            serving_spec["secret_sources"] = self._secrets.to_serial()

        env["SERVING_SPEC_ENV"] = json.dumps(serving_spec)
        return env

    def to_mock_server(
        self, namespace=None, current_function="*", track_models=False, **kwargs
    ) -> GraphServer:
        """create mock server object for local testing/emulation

        :param namespace: classes search namespace, use globals() for current notebook
        :param log_level: log level (error | info | debug)
        :param current_function: specify if you want to simulate a child function, * for all functions
        :param track_models: allow model tracking (disabled by default in the mock server)
        """

        server = create_graph_server(
            parameters=self.spec.parameters,
            load_mode=self.spec.load_mode,
            graph=self.spec.graph,
            verbose=self.verbose,
            current_function=current_function,
            graph_initializer=self.spec.graph_initializer,
            track_models=track_models and self.spec.track_models,
            function_uri=self._function_uri(),
            secret_sources=self.spec.secret_sources,
            default_content_type=self.spec.default_content_type,
            **kwargs,
        )
        server.init_states(
            context=None,
            namespace=namespace or get_caller_globals(),
            logger=logger,
            is_mock=True,
        )
        server.init_object(namespace or get_caller_globals())
        return server
