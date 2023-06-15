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

import os
import zipfile
import typing
import pathlib
from mlrun.track.base_tracker import BaseTracker
from mlrun.config import config as mlconf
from mlrun.execution import MLClientCtx
from mlrun.features import Feature


class MLFlowTracker(BaseTracker):
    """
    specific tracker to log artifacts, parameters and metrics collected by MLFlow
    """

    MODULE_NAME = "mlflow"

    @classmethod
    def is_relevant(cls) -> bool:
        """
        class method used to check if module is being used before creating MLFlowTracker
        """
        try:
            import mlflow

            return True
        except:
            return False

    def __init__(self):
        super().__init__()
        import mlflow

        self._tracked_platform = mlflow
        self._client = mlflow.MlflowClient()
        self.kwargs = {}

    def is_enabled(self) -> bool:
        """
        validates that mlflow should be tracked, both in user config and in env imports
        """
        return (
            self._tracked_platform is not None
            and mlconf.tracking.mlflow.enable_tracking
        )  # TODO check about more config enables

    def pre_run(self, context: MLClientCtx, env: dict, kwargs: dict) -> (dict, dict):
        """
        Initializes the tracking system for a 3rd party module.

        This function sets up the necessary components and resources to enable tracking of params, artifacts,
        or metrics within the module.
        Returns:
            two dictionaries, env and kwargs containing environment and tracking data
        """
        env = {}
        self.kwargs = kwargs
        experiment = self._tracked_platform.get_experiment_by_name(context.name)
        if experiment:  # check if exists, if not create
            experiment_id = experiment.experiment_id
        else:
            experiment_id = self._tracked_platform.create_experiment(context.name)
        env[
            f"{self.MODULE_NAME.upper()}_RUN_CONTEXT"
        ] = '{"mlrun_runid": "%s", "mlrun_project": "%s"}' % (
            context._uid,
            context.project,
        )
        env[f"{self.MODULE_NAME.upper()}_EXPERIMENT_ID"] = experiment_id
        self.kwargs[f"{self.MODULE_NAME}_experiment"] = experiment_id
        return env

    def _post_run(
        self,
        context: typing.Union[MLClientCtx, dict],
        experiment: str = None,
    ):
        """
        !!! not to confuse with post_run!!!
        Performs post-run tasks of logging 3rd party artifacts generated during the run.
        """
        experiment_id = self.kwargs.get(experiment)
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

        if runs:
            for run in runs:  # each run gets logged in a different child process
                self._log_run(context, run)

    def _log_run(self, context, run):
        """
        after mlrun function's run is done, copy all data logged by third party app tracker

        """
        self._artifacts = {}
        model_path = None
        for key, val in run.data.params.items():
            context._parameters[key] = val
        context.log_results(run.data.metrics)
        context.set_label(f"{self.MODULE_NAME}-runid", run.info.run_id)
        context.set_label(f"{self.MODULE_NAME}-experiment", run.info.experiment_id)
        for artifact in self._client.list_artifacts(run.info.run_id):
            full_path = self._tracked_platform.artifacts.download_artifacts(
                run_id=run.info.run_id, artifact_path=artifact.path
            )
            if artifact.is_dir and os.path.exists(os.path.join(full_path, "MLmodel")):
                model_path = full_path  # this is the model folder, we log it after logging all artifacts
            else:
                artifact = context.log_artifact(
                    item=pathlib.Path(artifact.path).stem, local_path=full_path
                )
                self._artifacts[artifact.key] = artifact
        if model_path:
            self.log_model(model_path, context)

    def log_model(
        self,
        model_uri: str,
        context: typing.Union[MLClientCtx, dict],
    ):
        """
        zips model dir and logs it and all artifacts
        """
        model_info = self._tracked_platform.models.get_model_info(model_uri=model_uri)
        model_zip = model_uri + ".zip"
        MLFlowTracker.zip_folder(model_uri, model_zip)
        key = model_info.artifact_path
        inputs = outputs = None

        if model_info.signature is not None:
            if model_info.signature.inputs is not None:
                inputs = MLFlowTracker.schema_to_feature(
                    model_info.signature.inputs, self.utils()
                )
            if model_info.signature.outputs is not None:
                outputs = MLFlowTracker.schema_to_feature(
                    model_info.signature.outputs, self.utils()
                )

        context.log_model(
            key,
            framework="mlflow",
            model_file=model_zip,
            metrics=context.results,
            parameters=model_info.flavors,
            labels={
                "mlflow_run_id": model_info.run_id,
                "mlflow_version": model_info.mlflow_version,
                "model_uuid": model_info.model_uuid,
            },
            extra_data=self._artifacts,
            inputs=inputs,
            outputs=outputs,
        )

    @staticmethod
    def zip_folder(folder_path: str, output_path: str):
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))

    @staticmethod
    def schema_to_feature(schema, utils) -> list:
        """
        changes the features from a scheme (usually tensor) to a list
        :param schema: features as made by mlflow
        :param utils: CommonUtils.convert_np_dtype_to_value_type, can't import here
        :return: list of features to log
        """
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
                    utils(value_type),
                    shape,
                    name=name,
                )
            )
        return features

    def post_run(self, context: typing.Union[MLClientCtx, dict], db=None):
        experiment = "mlflow_experiment"
        self._post_run(context=context, experiment=experiment)
