import json
from .function import RemoteRuntime, NuclioSpec


serving_subkind = "v2serving"


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
            function_kind=function_kind or serving_subkind,
            service_account=service_account,
            readiness_timeout=readiness_timeout,
        )

        self.models = models or {}
        self.router = router
        self.router_args = router_args
        self.parameters = parameters or {}


class ServingRuntime(RemoteRuntime):
    kind = "serving"

    @property
    def spec(self) -> ServingSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", ServingSpec)

    def add_model(
        self,
        name,
        model_class=None,
        model_path=None,
        model_url=None,
        parameters=None,
        load_mode=None,
        handler=None,
    ):
        """add ml model to the function

        :param name:        model api name (or name:version), will determine the relative url/path
        :param model_class: V2 Model python class name
                            (can also module.submodule.class and it will be imported automatically)
        :param model_path:  path to mlrun model artifact or model directory path
        :param model_url:   url of a remote url serving that model (cannot be used with model_path)
        :param parameters:  extra kwargs to pass to the model serving class __init__
                            (can be read in the model using .get_param(key) method)
        :param load_mode:   model loading mode: sync - during init, async - in the background
        :param handler:     for advanced users!, override default class handler name (do_event)
        """
        if not model_path and not model_url:
            raise ValueError("model_path or model_url must be provided")
        if model_path and not model_class:
            raise ValueError("model_path must be provided with model_class")
        if load_mode and load_mode not in ["sync", "async"]:
            raise ValueError(f"illegal model loading mode {load_mode}")

        model = {
            "model_class": model_class,
            "model_path": model_path,
            "model_url": model_url,
            "params": parameters,
            "load_mode": load_mode,
            "handler": handler,
        }
        model = {k: v for k, v in model.items() if v is not None}
        self.spec.models[name] = model

    def deploy(self, dashboard="", project="", tag=""):
        """deploy model serving function to a local/remote cluster

        :param dashboard: remote nuclio dashboard url (blank for local or auto detection)
        :param project:   optional, overide function specified project name
        :param tag:       specify unique function tag (a different function service is created for every tag)
        """
        kind = None
        if not self.spec.base_spec:
            kind = serving_subkind
        serving_spec = {
            "router_class": self.spec.router,
            "router_args": self.spec.router_args,
            "models": self.spec.models,
            "parameters": self.spec.parameters,
        }
        env = {"MODELSRV_SPEC_ENV": json.dumps(serving_spec)}
        return super().deploy(dashboard, project, tag, kind, env)
