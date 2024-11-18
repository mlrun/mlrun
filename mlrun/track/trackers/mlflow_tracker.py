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
import zipfile
from typing import Optional

import mlflow
import mlflow.entities
import mlflow.environment_variables
import mlflow.types

import mlrun
from mlrun import MLClientCtx, mlconf
from mlrun.artifacts import Artifact, ModelArtifact
from mlrun.features import Feature
from mlrun.launcher.client import ClientBaseLauncher
from mlrun.model import RunObject
from mlrun.projects import MlrunProject
from mlrun.track.tracker import Tracker
from mlrun.utils import logger, now_date


class MLFlowTracker(Tracker):
    """
    A class for detecting and logging MLFlow runs into MLRun. Notice, only the last active MLFlow run is logged.
    """

    def __init__(self):
        super().__init__()
        self._pre_runs = None
        self._experiment_name = None
        self._run_id = None

    @staticmethod
    def is_enabled() -> bool:
        """
        Checks if tracker is enabled.

        :return: True if the tracking configuration is enabled, False otherwise.
        """
        return mlconf.external_platform_tracking.mlflow.enabled

    def pre_run(self, context: MLClientCtx):
        """
        Initializes the MLFlow tracker, setting the experiment name as configured by the user.

        :param context: Current mlrun context
        """
        # Check for a user set experiment name via the environment variable:
        self._experiment_name = (
            mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.get()
        )

        # Check if the user configured for matching the experiment name with the context name:
        if mlconf.external_platform_tracking.mlflow.match_experiment_to_runtime:
            if self._experiment_name is not None:
                context.logger.warn(
                    f"`mlconf.external_platform_tracking.mlflow.match_experiment_to_runtime` is set to True but the "
                    f"MLFlow experiment name environment variable ('MLFLOW_EXPERIMENT_NAME') is set for using the "
                    f"name: '{self._experiment_name}'. This name will be overriden "
                    f"with MLRun's runtime name as set in the MLRun configuration: '{context.name}'."
                )
            self._experiment_name = context.name
            mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.set(
                self._experiment_name
            )

        # We have 3 options to track our run:
        # 1. Set the run id.
        # 2. Use the experiment, and check for added runs in it.
        # 3. Look for added runs across all experiments.
        # In case we want to determine the run id we have to create a new mlflow run, save the run id and use it
        # to override the run later (due to mlflow limitations)
        if mlrun.mlconf.external_platform_tracking.mlflow.control_run:
            with mlflow.start_run() as run:
                self._run_id = run.info.run_id
            mlflow.environment_variables.MLFLOW_RUN_ID.set(self._run_id)

        elif self._experiment_name:
            # We try to load an existing experiment first
            experiment = mlflow.get_experiment_by_name(name=self._experiment_name)
            # If no experiment with corresponding name exists, we create a new one
            if not experiment:
                experiment_id = mlflow.create_experiment(name=self._experiment_name)
                experiment = mlflow.get_experiment(experiment_id=experiment_id)

            # Save all runs logged in experiment prior of our run for later comparison
            self._pre_runs = set(
                [
                    run.info.run_id
                    for run in mlflow.search_runs(
                        experiment_ids=experiment.experiment_id,
                        output_format="list",
                    )
                ]
            )

        else:  # TODO Talk about removing/replacing with more exact methods
            # We warn user that when no experiment name or run id is given we look at all runs and might track the wrong
            # run in case where number of runs are running simultaneously
            context.logger.warn(
                "MLFlow Experiment Name and MLFlow Run ID Not Found. "
                "In this case, MLRun encountered an issue as it was unable to locate the MLFlow"
                " experiment name and run ID. When this occurs, MLRun will attempt to search through "
                "all experiments and runs to identify the newly added run. However, in scenarios where"
                " a substantial number of runs are active simultaneously, MLRun may not accurately log"
                " the correct run. To resolve this issue, please consider the following steps: "
                "1. Set MLFlow Environment Variables: Ensure that you have set the necessary"
                " MLFlow environment variables, specifically 'MLFLOW_EXPERIMENT_NAME' and"
                " 'MLFLOW_RUN_ID', to provide explicit identification for the experiment and run. "
                "2. Adjust mlconf Configuration (mlconf): "
                "Alternatively, you can configure MLRun by setting "
                "'mlconf.external_platform_tracking.mlflow.match_experiment_to_runtime` to True within"
                " the mlrun configuration (mlconf). This setting enables MLRun to match the "
                "experiment with the runtime more effectively. "
            )
            # Find all experiments to save all their runs for later comparison
            all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
            self._pre_runs = set(
                [
                    run.info.run_id
                    for run in mlflow.search_runs(
                        experiment_ids=all_experiments,
                        output_format="list",
                    )
                ]
            )

    def post_run(self, context: MLClientCtx):
        """
        Performs post-run tasks for logging MLFlow artifacts generated during the run.

        :param context: Current mlrun context
        """
        # We try to find the correct run to log via available data
        if self._run_id:
            run = [self._run_id]

        elif self._experiment_name:
            # If we now the experiment's name we can look at all runs and find the new one
            experiment = mlflow.get_experiment_by_name(self._experiment_name)
            run = (
                set(
                    [
                        run.info.run_id
                        for run in mlflow.search_runs(
                            experiment_ids=experiment.experiment_id,
                            output_format="list",
                        )
                    ]
                )
                ^ self._pre_runs  # xor to find the run added to the list
            )

        else:
            # In case we don't have the experiment name or run id we need to look at all new runs to found the one added
            all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
            run = (
                set(
                    [
                        run.info.run_id
                        for run in mlflow.search_runs(
                            experiment_ids=all_experiments,
                            output_format="list",
                        )
                    ]
                )
                ^ self._pre_runs  # xor to find the run added to the list
            )
        # should be only one run id in run
        run = mlflow.get_run(run_id=run.pop())

        # Log the run:
        MLFlowTracker._log_run(context=context, run=run)

    def import_run(
        self,
        project: MlrunProject,
        reference_id: str,
        function_name: str,
        handler: Optional[str] = None,
        **kwargs,
    ) -> RunObject:
        """
        Import a previous MLFlow experiment run to MLRun.

        :param project:       The MLRun project to import the run to.
        :param reference_id:  The MLFlow `run_id` to import.
        :param function_name: The MLRun function to assign this run to.
        :param handler:       The handler for MLRun's RunObject

        :return: The newly imported RunObject.
        """

        run = mlflow.get_run(reference_id)
        if not run:
            raise ValueError(
                "The provided run id was not found in the set MLFlow registry. Try to set the registry using "
                "MLFlow's environment variables."
            )

        # Create the run object to hold the mlflow run:
        run_object = self._create_run_object(
            handler=handler,
            run_name=run.info.run_name,
            project_name=project.name,
            uid=run.info.run_uuid,
        )

        # Create a context from the run object:
        ctx = mlrun.get_or_create_ctx(
            name=run.info.run_name,
            spec=run_object,
        )
        # Store the run in the MLRun DB, then import the MLFlow data to it:
        self._log_run(context=ctx, run=run, is_offline=True)

        # Create a rundb in order to update the run's state as completed (can't be done using context)
        rundb = mlrun.get_run_db()
        project = run_object.metadata.project
        uid = run_object.metadata.uid
        updates = {
            "status.last_update": now_date().isoformat(),
            "status.state": "completed",
        }
        run_object.status.state = "completed"
        rundb.update_run(updates, uid, project)

        # Print a summary message after importing the run:
        result = ctx.to_dict()
        run_object = RunObject.from_dict(result)
        ClientBaseLauncher._log_track_results(
            is_child=False, result=result, run=run_object
        )

        return run_object

    def import_model(
        self,
        project: MlrunProject,
        reference_id: str,
        key: Optional[str] = None,
        metrics: Optional[dict] = None,
        extra_data: Optional[dict] = None,
    ) -> ModelArtifact:
        """
        Import a model from MLFlow to MLRun.

        :param project:         The MLRun project to import the model to.
        :param reference_id:    The MLFlow model uri to import.
        :param key:             The model key to be used in MLRun. Mandatory for importing.
        :param metrics:         The model's metrics.
        :param extra_data:      Extra artifacts and files to log with the model.

        :return: The newly imported ModelArtifact.
        """
        # Validate key is given:
        if key is None:
            raise ValueError(
                "MLFlow models require a key to import into MLRun - the key of the model artifact to be created."
            )

        # Setup defaults:
        metrics = metrics or {}
        extra_data = extra_data or {}
        with tempfile.TemporaryDirectory() as temp_dir:
            # Log the model:
            model = self._log_model(
                project=project,
                model_uri=reference_id,
                key=key,
                metrics=metrics,
                extra_data=extra_data,
                tmp_path=temp_dir,
            )

            logger.info("model imported successfully", key=key)
            return model

    def import_artifact(
        self, project: MlrunProject, reference_id: str, key: Optional[str] = None
    ) -> Artifact:
        """
        Import an artifact from MLFlow to MLRun.

        :param project:      The MLRun project to import the artifact to.
        :param reference_id: The MLFlow artifact uri to import.
        :param key:          The artifact key to be used in MLRun. Mandatory for importing.

        :return: The newly imported artifact.
        """
        # Validate key is given:
        if key is None:
            raise ValueError(
                "MLFlow artifacts require a key to import into MLRun - the key of the artifact to be created."
            )

        # Import the artifact:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download the artifact to local temp:
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=reference_id, dst_path=tmp_dir
            )

            # Log and return the artifact:
            artifact = self._log_artifact(
                project=project,
                key=key,
                local_path=local_path,
                tmp_path=tmp_dir,
            )

            logger.info("Artifact imported successfully", key=key)
            return artifact

    @staticmethod
    def _log_run(
        context: MLClientCtx, run: mlflow.entities.Run, is_offline: bool = False
    ):
        """
        Log the given MLFlow run to MLRun.

        :param context:     Current MLRun context or project.
        :param run:         MLFlow Run to log. Can be given as a `Run` object
        :param is_offline:  True if logging an offline run (importing), False if online run (tracking)
        """
        client = mlflow.MlflowClient()

        # Create a temporary directory to log all data temporarily:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # When running in offline mode (manually importing) we want to make sure there is an `artifact_path` set in
            # the context, if not we will set it to this temporary directory so the user won't have garbage in his
            # local work dir:
            if is_offline and not context.artifact_path:
                context.artifact_path = tmp_dir

            # Set new MLRun related tags to the MLFlow run:
            client.set_tag(
                run_id=run.info.run_id, key="mlrun-context-id", value=context.uid
            )
            client.set_tag(
                run_id=run.info.run_id, key="mlrun-project-name", value=context.project
            )

            # Get the MLFlow run's tags and save them as labels:
            context.set_label(key="mlflow-user", value=run.data.tags.get("mlflow.user"))
            context.set_label(
                key="mlflow-run-name", value=run.data.tags.get("mlflow.runName")
            )
            context.set_label(key="mlflow-run-id", value=run.info.run_id)
            context.set_label(key="mlflow-experiment-id", value=run.info.experiment_id)

            # Get the MLFlow run's parameters and save as job parameters:
            context._parameters.update(run.data.params)

            # Get the MLFlow run's metrics and log them as results:
            results = run.data.metrics
            context.log_results(results=results)

            # Import the MLFlow run's artifacts to MLRun (model are logged after the rest of artifacts
            # so the artifacts can be registered as extra data in the models):
            artifacts = {}
            model_paths = []
            for artifact in client.list_artifacts(run_id=run.info.run_id):
                # Get the artifact's local path (MLFlow suggests that if the artifact is already in the local filesystem
                # its local path will be returned:
                artifact_local_path = mlflow.artifacts.download_artifacts(
                    run_id=run.info.run_id,
                    artifact_path=artifact.path,
                )
                # Check if the artifact is a model (will be logged after the artifacts):
                if artifact.is_dir and os.path.exists(
                    os.path.join(
                        artifact_local_path, "MLmodel"
                    )  # Add tag to show model dir
                ):
                    model_paths.append(artifact_local_path)
                else:
                    # Log the artifact:
                    artifact = MLFlowTracker._log_artifact(
                        context=context,
                        key=pathlib.Path(artifact.path).name.replace(".", "_"),
                        # Mlflow has the same name for files but with different extensions, so we add extension to name
                        local_path=artifact_local_path,
                        tmp_path=tmp_dir,
                    )
                    artifacts[artifact.key] = artifact

            for model_path in model_paths:
                MLFlowTracker._log_model(
                    context=context,
                    model_uri=model_path,
                    key=pathlib.Path(model_path).stem,
                    metrics=results,
                    extra_data=artifacts,
                    tmp_path=tmp_dir,
                )

    @staticmethod
    def _log_model(
        model_uri: str,
        key: str,
        metrics: dict,
        extra_data: dict,
        tmp_path: os.path,
        context: MLClientCtx = None,
        project: MlrunProject = None,
    ):
        """
        Log the given produced model from MLFlow as a model artifact in MLRun.

        :param model_uri:   The local path to the model (an MLFlow model directory locally downloaded).
        :param key:         The model artifact's key.
        :param metrics:     The key/value dict of model metrics
        :param extra_data:  The extra data to log in addition to the model (training data for example)
        :param tmp_path:    The path to the dir where we temporarily save model and artifacts
        :param context:     The MLRun context to log to, needed one of context or project.
        :param project:     The MLRun project to log to, needed one of context or project.
        """
        # Check that either project or context is provided:
        if not project and not context:
            logger.error(
                "One of context or project must be given in order to log model"
            )
            return

        # Get the model info from MLFlow:
        model_info = mlflow.models.get_model_info(model_uri=model_uri)

        # Prepare the archive path:
        model_uri = pathlib.Path(model_uri)
        archive_path = pathlib.Path(tmp_path) / f"{model_uri.stem}.zip"
        if not os.path.exists(model_uri):
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=str(model_uri)
            )
            model_uri = pathlib.Path(local_path)

        # TODO add progress bar for the case of large files
        # Zip the artifact:
        with zipfile.ZipFile(archive_path, "w") as zip_file:
            for path in model_uri.rglob("*"):
                zip_file.write(filename=path, arcname=path.relative_to(model_uri))

        # Get inputs and outputs info:
        inputs = outputs = None
        if model_info.signature is not None:
            if model_info.signature.inputs is not None:
                inputs = MLFlowTracker._schema_to_feature(
                    schema=model_info.signature.inputs
                )
            if model_info.signature.outputs is not None:
                outputs = MLFlowTracker._schema_to_feature(
                    schema=model_info.signature.outputs
                )
        # Log the model:

        kwargs = {
            "key": key,
            "framework": "mlflow",
            "model_file": str(archive_path),
            "metrics": metrics,
            "labels": {
                "mlflow_flavors": str(list(model_info.flavors.keys())),
                "mlflow_run_id": model_info.run_id,
                "mlflow_version": model_info.mlflow_version,
                "mlflow_model_uuid": model_info.model_uuid,
            },
            "extra_data": extra_data,
            "inputs": inputs,
            "outputs": outputs,
        }
        return context.log_model(**kwargs) if context else project.log_model(**kwargs)

    @staticmethod
    def _log_artifact(
        key: str,
        local_path: str,
        tmp_path: str,
        context: MLClientCtx = None,
        project: MlrunProject = None,
    ) -> Artifact:
        """
        Log the given produced file from MLFlow as a run artifact in MLRun.

        :param key:         The artifact's key.
        :param local_path:  The local path to the artifact.
        :param tmp_path:    The path to the dir where we temporarily save artifacts
        :param context:     The MLRun context to log to, needed one of context or project.
        :param project:     The MLRun project to log to, needed one of context or project.
        """
        # Check that either project or context is provided:
        if not project and not context:
            logger.error(
                "One of context or project must be given in order to log artifact"
            )
            return
        # Check if the artifact is a directory for archiving it:
        if pathlib.Path(local_path).is_dir():
            # Prepare the archive path:
            archive_path = pathlib.Path(tmp_path) / f"{key}.zip"
            local_path = pathlib.Path(local_path)
            # Zip the artifact:
            with zipfile.ZipFile(archive_path, "w") as zip_file:
                for path in local_path.rglob("*"):
                    zip_file.write(filename=path, arcname=path.relative_to(local_path))
            # Set the local path to the archive file:
            local_path = str(archive_path)
        kwargs = {
            "item": key,
            "local_path": local_path,
        }
        # Log and return the artifact in the local path:
        return (
            context.log_artifact(**kwargs)
            if context
            else project.log_artifact(**kwargs)
        )

    @staticmethod
    def _schema_to_feature(schema: mlflow.types.Schema) -> list[Feature]:
        """
        Cast MLFlow schema to MLRun features.

        :param schema: The MLFlow schema.

        :return: List of MLRun features representing the schema.
        """
        # Import here due to circular dependencies:
        from mlrun.frameworks._common import CommonUtils

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
                    CommonUtils.convert_np_dtype_to_value_type(value_type),
                    shape,
                    name=name,
                )
            )

        return features

    @staticmethod
    def _create_run_object(handler, run_name, project_name, uid):
        """
        A util function for creating a RunObject
        :param handler:         The handler for MLRun's RunObject
        :param run_name:        The name of the run
        :param project_name:    The name of the project
        :param uid:             The uid of the run
        :return: RunObject
        """
        task = mlrun.new_task(handler=handler, name=run_name, project=project_name)
        task.metadata.uid = uid
        return RunObject.from_template(task)
