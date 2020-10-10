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
import nuclio

from .function import RemoteRuntime, NuclioSpec

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
        router=None,
        router_args=None,
        parameters=None,
        default_class=None,
        load_mode=None,
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
        )

        self.models = models or {}
        self.router = router
        self.router_args = router_args
        self.parameters = parameters or {}
        self.default_class = default_class
        self.load_mode = load_mode


class ServingRuntime(RemoteRuntime):
    kind = "serving"

    @property
    def spec(self) -> ServingSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", ServingSpec)

    def set_router(self, router, **router_args):
        """set the routing class and router arguments"""
        self.spec.router = router
        self.spec.router_args = router_args

    def set_tracking(self, stream_path, batch=None, sample=None):
        """set tracking log stream parameters"""
        self.spec.parameters["log_stream"] = stream_path
        if batch:
            self.spec.parameters["log_stream_batch"] = batch
        if sample:
            self.spec.parameters["log_stream_sample"] = sample

    def add_model(
        self,
        name,
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

        :param name:        model api name (or name:version), will determine the relative url/path
        :param model_path:  path to mlrun model artifact or model directory file/object path
        :param class_name: V2 Model python class name
                            (can also module.submodule.class and it will be imported automatically)
        :param model_url:   url of a remote model serving endpoint (cannot be used with model_path)
        :param handler:     for advanced users!, override default class handler name (do_event)
        :param class_args:  extra kwargs to pass to the model serving class __init__
                            (can be read in the model using .get_param(key) method)
        """
        if not model_path and not model_url:
            raise ValueError("model_path or model_url must be provided")
        if model_path and not class_name and not self.spec.default_class:
            raise ValueError("model_path must be provided with class_name")
        if model_path:
            model_path = str(model_path)

        class_args = deepcopy(class_args)
        if model_path:
            class_args["model_path"] = model_path
        if model_url:
            class_name = "$remote"
            class_args["url"] = model_url

        model = {
            "class_name": class_name or self.spec.default_class,
            "class_args": class_args,
        }
        if handler:
            model["handler"] = handler
        self.spec.models[name] = model

    def remove_models(self, *names):
        """remove one, multiple, or all models from the spec (blank list for all)"""
        if not names:
            names = self.spec.models.keys()
        for name in names:
            del self.spec.models[name]

    def deploy(self, dashboard="", project="", tag=""):
        """deploy model serving function to a local/remote cluster

        :param dashboard: remote nuclio dashboard url (blank for local or auto detection)
        :param project:   optional, overide function specified project name
        :param tag:       specify unique function tag (a different function service is created for every tag)
        """
        load_mode = self.spec.load_mode
        if load_mode and load_mode not in ["sync", "async"]:
            raise ValueError(f"illegal model loading mode {load_mode}")
        kind = None
        if not self.spec.base_spec:
            kind = serving_subkind

        # we currently support a minimal topology of one router + multiple child routes/models
        # in the future we will extend the support to a full graph, the spec is already built accordingly
        serving_spec = {
            "version": "v2",
            "parameters": self.spec.parameters,
            "router": {
                "class_name": self.spec.router,
                "class_args": self.spec.router_args,
                "routes": self.spec.models,
            },
            "load_mode": load_mode,
            "verbose": self.verbose,
        }
        env = {"SERVING_SPEC_ENV": json.dumps(serving_spec)}
        return super().deploy(dashboard, project, tag, kind, env)
