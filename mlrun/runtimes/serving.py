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

        self.models = models
        self.router = router
        self.router_args = router_args
        self.parameters = parameters


class ServingRuntime(RemoteRuntime):
    kind = "serving"

    @property
    def spec(self) -> ServingSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", ServingSpec)

    def add_model(
        self, name, model_class, model_path=None, model_url=None, parameters=None
    ):
        if not model_path and not model_url:
            raise ValueError("model_path or model_url must be provided")

        model = {
            "model_class": model_class,
            "model_path": model_path,
            "model_url": model_url,
            "params": parameters,
        }
        self.spec.models[name] = model

    def deploy(self, dashboard="", project="", tag="", kind=None):
        if not kind and not self.spec.base_spec:
            kind = serving_subkind
        serving_spec = {
            "router_class": self.spec.router,
            "router_args": self.spec.router_args,
            "models": self.spec.models,
            "parameters": self.spec.parameters,
        }
        env = {"MODELSRV_SPEC_ENV": json.dumps(serving_spec)}
        return super().deploy(dashboard, project, tag, kind, env)
