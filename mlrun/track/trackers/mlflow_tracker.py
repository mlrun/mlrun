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
from typing import List, Union

import mlflow
import mlflow.entities
import mlflow.environment_variables
import mlflow.types

import mlrun
from mlrun import MLClientCtx, mlconf
from mlrun.artifacts import Artifact, ModelArtifact
from mlrun.features import Feature
from mlrun.launcher.client import ClientBaseLauncher
from mlrun.model import RunMetadata, RunObject, RunSpec
from mlrun.projects import MlrunProject
from mlrun.track.tracker import Tracker
from mlrun.utils import now_date


class MLFlowTracker(Tracker):
    """
    A class for detecting and logging MLFlow runs into MLRun. Notice, only the last active MLFlow run is logged.
    """

    @classmethod
    def is_enabled(cls) -> bool:
        """
        Checks if tracker is enabled.

        :return: True if the tracking configuration is enabled, False otherwise.
        """
        return getattr(mlconf.external_platform_tracking, mlflow.__name__).enabled

    @classmethod
    def pre_run(cls, context: MLClientCtx):
        """
        Initializes the MLFlow tracker, setting the experiment name as configured by the user.

        :param context: Current mlrun context
        """
        # Check for a user set experiment name via the environment variable:
        experiment_name = mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.get()

        # Check if the user configured for matching the experiment name with the context name:
        if getattr(
            mlconf.external_platform_tracking, mlflow.__name__
        ).match_experiment_to_runtime:
            if experiment_name is not None:
                context.logger.warn(
                    f"`mlconf.external_platform_tracking.mlflow.match_experiment_to_runtime` is set to True but the "
                    f"MLFlow experiment name environment variable ('MLFLOW_EXPERIMENT_NAME') is set for using the "
                    f"name: '{experiment_name}'. This name will be overriden "
                    f"with MLRun's runtime name as set in the MLRun configuration: '{context.name}'."
                )
            experiment_name = context.name
            mlflow.set_experiment(experiment_name=experiment_name)

    @classmethod
    def post_run(cls, context: MLClientCtx):
        """
        Performs post-run tasks for logging MLFlow artifacts generated during the run.

        :param context: Current mlrun context
        """
        # Get the last run (None means no run was created or found):
        run = mlflow.last_active_run()
        if run is None:
            return

        # Log the run:
        MLFlowTracker._log_run(context=context, run=run)

    @classmethod
    def import_run(
        cls,
        project: MlrunProject,
        pointer: str,
        function_name: str = None,
        handler: str = None,
    ) -> RunObject:
        """
        Import a previous MLFlow experiment run to MLRun.

        :param project:       The MLRun project to import the run to.
        :param pointer:       The MLFlow `run_id` to import.
        :param function_name: The MLRun function to assign this run to.
        :param handler:       The handler for MLRun's RunObject

        :return: The newly imported RunObject.
        """
        # Validate function name was given:
        if function_name is None:
            raise ValueError(
                "For importing a MLFlow experiment run, a MLRun function name must be provided."
            )

        # Get the MLFlow run object:
        run = mlflow.search_runs(
            search_all_experiments=True,
            filter_string=f"attributes.run_id = '{pointer}'",
            output_format="list",
        )
        if not run:
            raise ValueError(
                "The provided run id was not found in the set MLFlow registry. Try to set the registry using "
                "MLFlow's environment variables."
            )
        run = run[0]  # We are using a run id, so only one will be returned in the list.
        # Create a run spec and metadata for creating the run object to hold the mlflow run:
        run_spec = RunSpec(function=function_name, handler=handler)
        run_metadata = RunMetadata(
            uid=run.info.run_uuid, name=run.info.run_name, project=project.name
        )
        run_object = RunObject(spec=run_spec, metadata=run_metadata)
        # Create a context from the run object:
        ctx = mlrun.get_or_create_ctx(
            name=run.info.run_name,
            spec=run_object,
        )
        # Store the run in the MLRun DB, then import the MLFlow data to it:
        ctx.store_run()
        cls._log_run(context=ctx, run=run, is_offline=True)

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
        # RuntimeMock needed for mocking runtime parameter in `_log_track_results`
        class RuntimeMock:
            is_child = False

        result = ctx.to_dict()
        run_object = RunObject.from_dict(result)
        ClientBaseLauncher._log_track_results(
            runtime=RuntimeMock, result=result, run=run_object
        )

        return run_object

    @classmethod
    def import_model(
        cls,
        project: MlrunProject,
        pointer: str,
        key: str = None,
        metrics: dict = None,
        extra_data: dict = None,
    ) -> ModelArtifact:
        """
        Import a model from MLFlow to MLRun.

        :param project:    The MLRun project to import the model to.
        :param pointer:    The MLFlow model uri to import.
        :param key:        The model key to be used in MLRun. Mandatory for importing.
        :param metrics:    The model's metrics.
        :param extra_data: Extra artifacts and files to log with the model.

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
            model = cls._log_model(
                context_or_project=project,
                model_uri=pointer,
                key=key,
                metrics=metrics,
                extra_data=extra_data,
                tmp_path=temp_dir,
            )

            print(f"[Info]: model {key} imported successfully")
            return model

    @classmethod
    def import_artifact(
        cls, project: MlrunProject, pointer: str, key: str = None
    ) -> Artifact:
        """
        Import an artifact from MLFlow to MLRun.

        :param project: The MLRun project to import the artifact to.
        :param pointer: The MLFlow artifact uri to import.
        :param key:     The artifact key to be used in MLRun. Mandatory for importing.

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
                artifact_uri=pointer, dst_path=tmp_dir
            )

            # Log and return the artifact:
            artifact = cls._log_artifact(
                context_or_project=project,
                key=key,
                local_path=local_path,
                tmp_path=tmp_dir,
            )

            print(f"[Info]: model {key} imported successfully")
            return artifact

    @staticmethod
    def _log_run(
        context: MLClientCtx, run: mlflow.entities.Run, is_offline: bool = False
    ):
        """
        Log the given MLFlow run to MLRun.

        :param context:    Current MLRun context or project.
        :param run: MLFlow Run to log. Can be given as a `Run` object or as a run id.
        :param is_offline: True if logging an offline run (importing), False if online run (tracking)
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
                    os.path.join(artifact_local_path, "MLmodel")
                ):
                    model_paths.append(artifact_local_path)
                else:
                    # Log the artifact:
                    artifact = MLFlowTracker._log_artifact(
                        context_or_project=context,
                        key=pathlib.Path(artifact.path).name.replace(".", "_"),
                        local_path=artifact_local_path,
                        tmp_path=tmp_dir,
                    )
                    artifacts[artifact.key] = artifact
            if model_paths:
                for model_path in model_paths:
                    MLFlowTracker._log_model(
                        context_or_project=context,
                        model_uri=model_path,
                        key=pathlib.Path(model_path).stem,
                        metrics=results,
                        extra_data=artifacts,
                        tmp_path=tmp_dir,
                    )

    @staticmethod
    def _log_model(
        context_or_project: Union[MLClientCtx, MlrunProject],
        model_uri: str,
        key: str,
        metrics: dict,
        extra_data: dict,
        tmp_path: os.path,
    ):
        """
        Log the given produced model from MLFlow as a model artifact in MLRun.

        :param context_or_project: The MLRun context or project to log to.
        :param model_uri:          The local path to the model (an MLFlow model directory locally downloaded).
        :param key:                The model artifact's key.
        :param metrics:            The key/value dict of model metrics
        :param extra_data:         The extra data to log in addition to the model (training data for example)
        :param tmp_path:           The path to the dir where we temporarily save model and artifacts
        """
        # Get the model info from MLFlow:
        model_info = mlflow.models.get_model_info(model_uri=model_uri)

        # Prepare the archive path:
        model_uri = pathlib.Path(model_uri)
        archive_path = pathlib.Path(tmp_path) / f"{model_uri.stem}.zip"

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
        return context_or_project.log_model(
            key=key,
            framework="mlflow",
            model_file=str(archive_path),
            metrics=metrics,
            labels={
                "mlflow_run_id": model_info.run_id,
                "mlflow_version": model_info.mlflow_version,
                "mlflow_model_uuid": model_info.model_uuid,
            },
            extra_data=extra_data,
            inputs=inputs,
            outputs=outputs,
        )

    @staticmethod
    def _log_artifact(
        context_or_project: Union[MLClientCtx, MlrunProject],
        key: str,
        local_path: str,
        tmp_path: str,
    ) -> Artifact:
        """
        Log the given produced file from MLFlow as a run artifact in MLRun.

        :param context_or_project: The MLRun context or project to log to.
        :param key:                The artifact's key.
        :param local_path:         The local path to the artifact.
        :param tmp_path:           The path to the dir where we temporarily save artifacts
        """
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

        # Log and return the artifact in the local path:
        return context_or_project.log_artifact(
            item=key,
            local_path=local_path,
        )

    @staticmethod
    def _schema_to_feature(schema: mlflow.types.Schema) -> List[Feature]:
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
