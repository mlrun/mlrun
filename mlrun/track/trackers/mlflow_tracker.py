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
import pathlib
import tempfile

from mlrun.execution import MLClientCtx
from mlrun.features import Feature
from mlrun.track.base_tracker import BaseTracker
from mlrun.track.utils import convert_np_dtype_to_value_type, zip_folder
from mlrun.utils import logger


class MLFlowTracker(BaseTracker):
    """
    specific tracker to log artifacts, parameters and metrics collected by MLFlow
    """

    TRACKED_MODULE_NAME = "mlflow"

    def __init__(self):
        super().__init__()
        self._mlflow_experiment = None
        self._track_uri = 0
        self._client = None

    def pre_run(self, context: MLClientCtx) -> dict:
        env = {}
        experiment = self._tracked_platform.get_experiment_by_name(context.name)
        if experiment:  # check if exists, if not create
            experiment_id = experiment.experiment_id
        else:
            experiment_id = self._tracked_platform.create_experiment(context.name)
        env["MLFLOW_RUN_CONTEXT"] = '{"mlrun_runid": "%s", "mlrun_project": "%s"}' % (
            context.uid,
            context.project,
        )
        env["MLFLOW_EXPERIMENT_ID"] = experiment_id
        self._mlflow_experiment = experiment_id
        return env

    def _apply_post_run_tasks(
        self,
        context: MLClientCtx,
    ):
        """
        Performs post-run tasks of logging 3rd party artifacts generated during the run.
        :param context: current mlrun context
        """
        experiment_id = self._mlflow_experiment
        runs = self._client.search_runs(
            experiment_id, filter_string=f'tags.mlrun_runid="{context.uid}"'
        )
        if not runs:
            logger.debug("No runs from mlflow client, searching in experiments")
            experiments = [
                experiment.experiment_id
                for experiment in self._client.search_experiments()
            ]
            runs = self._client.search_runs(
                experiments, filter_string=f'tags.mlrun_runid="{context.uid}"'
            )

        if runs:
            for run in runs:  # each run gets logged separately
                logger.debug("Logging run", run=run)
                self._log_run(context, run)
        else:
            logger.debug("No runs found to log")

    def _log_run(self, context: MLClientCtx, run):
        """
        after mlrun function's run is done, copy all data logged by third party app tracker
        :param context: current mlrun context
        :param run: mlflow run to log
        """
        model_paths = []
        logger.debug("Logging MLFlow params")
        for key, val in run.data.params.items():
            context._parameters[key] = val
        logger.debug("Finished to log MLFlow params")
        context.log_results({"mlflow_run_metrics": run.data.metrics})
        context.set_label("mlflow-runid", run.info.run_id)
        context.set_label("mlflow-experiment", run.info.experiment_id)
        logger.debug("Starting to log MLFlow artifacts")
        for artifact in self._client.list_artifacts(run.info.run_id):
            full_path = self._tracked_platform.artifacts.download_artifacts(
                run_id=run.info.run_id, artifact_path=artifact.path
            )
            if artifact.is_dir and os.path.exists(os.path.join(full_path, "MLmodel")):
                model_paths.append(
                    full_path
                )  # this is the model folder, we log it after logging all artifacts
            else:
                self.log_artifact(context, full_path, artifact)
        logger.debug("Finished to log MLFlow artifacts")
        if model_paths:
            logger.debug("Starting to log MLFlow model")
            for path in model_paths:
                self.log_model(path, context)
            logger.debug("Finished to log MLFlow model")

    def log_model(
        self,
        model_uri: str,
        context: MLClientCtx,
    ):

        model_info = self._tracked_platform.models.get_model_info(model_uri=model_uri)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_zip = f"{tmp_dir}/model.zip"
            zip_folder(model_uri, model_zip)
            key = model_info.artifact_path
            inputs = outputs = None

            if model_info.signature is not None:
                if model_info.signature.inputs is not None:
                    inputs = schema_to_feature(model_info.signature.inputs)
                if model_info.signature.outputs is not None:
                    outputs = schema_to_feature(model_info.signature.outputs)
            context.log_model(
                key,
                framework="mlflow",
                model_file=model_zip,
                metrics=context.results.get("mlflow_run_metrics"),
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

    # todo add mlflow.artifact hint
    def log_artifact(self, context: MLClientCtx, local_path: str, artifact):
        artifact = context.log_artifact(
            item=pathlib.Path(artifact.path).name.replace(".", "_"),
            local_path=local_path,
        )
        self._artifacts[artifact.key] = artifact

    def post_run(self, context: MLClientCtx):
        self._client = self._tracked_platform.MlflowClient()
        self._apply_post_run_tasks(context=context)

    # todo actually implement this
    def log_dataset(self, dataset_path, context):
        pass


def schema_to_feature(schema) -> list:  # todo add hints here
    """
    changes the features from a scheme (usually tensor) to a list
    :param schema: features as made by mlflow

    :return: list of features to log
    """
    converter = convert_np_dtype_to_value_type()
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
                converter(value_type),
                shape,
                name=name,
            )
        )
    return features
