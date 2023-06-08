import abc
import os
import typing

import mlrun

from ..execution import MLClientCtx
from ..features import Feature


class TrackingProvider(abc.ABC):
    def is_enabled(self, mode):
        return True

    @abc.abstractmethod
    def init_tracking(
        self, context: MLClientCtx, env: dict, args: dict
    ) -> (dict, dict):
        pass

    @abc.abstractmethod
    def post_run(self, context: typing.Union[MLClientCtx, dict], args: dict):
        pass


class MLflowTracker(TrackingProvider):
    def __init__(self):
        self._utils = None

        try:
            import mlflow

            self._mlflow = mlflow
            self._client = mlflow.MlflowClient()
        except ImportError:
            self._mlflow = None

    def utils(self):
        if self._utils:
            return self._utils

        from mlrun.frameworks._common import CommonUtils

        self._utils = CommonUtils
        return self._utils

    def is_enabled(self, mode):
        mlflow_mode = mode and mode == "mlflow"
        return self._mlflow is not None and (
            mlflow_mode or mlrun.mlconf.mlflow_tracking
        )

    def init_tracking(self, context: MLClientCtx, env: dict, args: dict):
        env = {}
        experiment = self._mlflow.get_experiment_by_name(context.name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = self._mlflow.create_experiment(context.name)
        env["MLFLOW_RUN_CONTEXT"] = '{"mlrun_runid": "%s", "mlrun_project": "%s"}' % (
            context._uid,
            context.project,
        )
        env["MLFLOW_EXPERIMENT_ID"] = experiment_id
        args["mlflow_experiment"] = experiment_id
        return env, args

    def _update_run(self, context, run):
        print(f"run: {run.info.run_id}")
        for key, val in run.data.params.items():
            context._parameters[key] = val
        context.log_results(run.data.metrics)
        context.set_label("mlflow-runid", run.info.run_id)
        context.set_label("mlflow-experiment", run.info.experiment_id)
        for artifact in self._client.list_artifacts(run.info.run_id):
            print(artifact.path, artifact.is_dir)
            full_path = self._mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id, artifact_path=artifact.path
            )
            if artifact.is_dir and os.path.exists(os.path.join(full_path, "MLmodel")):
                self._log_model(full_path, context)
            else:
                context.log_artifact(artifact.path, local_path=full_path)
        print()

    def _log_model(self, model_uri, context):
        model_info = self._mlflow.models.get_model_info(model_uri=model_uri)
        key = model_info.artifact_path
        extra_data = {
            f: f
            for f in os.listdir(model_uri)
            if f != "MLmodel" and not f.startswith(".")
        }
        inputs = outputs = None

        def schema_to_feature(schema):
            is_tensor = schema.is_tensor_spec()
            features = []
            for i, item in enumerate(schema.inputs):
                name = item.name or str(i)
                shape = None
                if is_tensor:
                    value_type = item.type
                    shape = list(item.shape) if item.shape else None
                else:
                    value_type = item.type.to_numpy()
                features.append(
                    Feature(
                        self.utils().convert_np_dtype_to_value_type(value_type),
                        shape,
                        name=name,
                    )
                )
            return features

        if model_info.signature is not None:
            if model_info.signature.inputs is not None:
                inputs = schema_to_feature(model_info.signature.inputs)
            if model_info.signature.outputs is not None:
                outputs = schema_to_feature(model_info.signature.outputs)

        context.log_model(
            key,
            framework="mlflow",
            model_file="MLmodel",
            model_dir=model_uri,
            metrics=context.results,
            parameters=model_info.flavors,
            labels={
                "mlflow_run_id": model_info.run_id,
                "mlflow_version": model_info.mlflow_version,
                "model_uuid": model_info.model_uuid,
            },
            extra_data=extra_data,
            inputs=inputs,
            outputs=outputs,
        )

    def post_run(self, context: typing.Union[MLClientCtx, dict], args: dict):
        experiment_id = args.get("mlflow_experiment")
        runs = self._client.search_runs(
            experiment_id, filter_string=f'tags.mlrun_runid="{context._uid}"'
        )
        if not runs:
            experiments = [
                experiment.experiment_id
                for experiment in self._client.search_experiments()
            ]
            runs = self._client.search_runs(
                experiments, filter_string=f'tags.mlrun_runid="{context._uid}"'
            )

        # todo: handle multiple child runs
        if runs:
            self._update_run(context, runs[0])


class RunTrackingServices:
    def __init__(self):
        self._trackers = []

    def register(self, tracker_class):
        self._trackers.append(tracker_class())

    def init_tracking(
        self, context: MLClientCtx, mode: str = None, env: dict = None
    ) -> (dict, dict):
        env = env or {}
        args = {"mode": mode}
        for tracker in self._trackers:
            if tracker.is_enabled(mode):
                env, args = tracker.init_tracking(context, env, args)
        return env, args

    def post_run(self, context: typing.Union[MLClientCtx, dict], args: dict) -> dict:
        for tracker in self._trackers:
            if tracker.is_enabled(args.get("mode")):
                if isinstance(context, dict):
                    context = MLClientCtx.from_dict(context)
                tracker.post_run(context, args)

        if isinstance(context, dict):
            return context
        context.commit()
        return context.to_dict()


tracking_services = RunTrackingServices()
tracking_services.register(MLflowTracker)