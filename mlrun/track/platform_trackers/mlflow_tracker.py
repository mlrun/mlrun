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
from mlrun.track.base_tracker import BaseTracker
from mlrun.config import config as mlconf
from mlrun.execution import MLClientCtx
from mlrun.features import Feature


class MLFlowTracker(BaseTracker):
    '''
    specific tracker to log artifacts, parameters and metrics collected by MLFlow
    '''
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
        try:
            import mlflow
            self._tracked_platform = mlflow
            self._client = mlflow.MlflowClient()
        except ImportError:
            self._tracked_platform = None

    def is_enabled(self) -> bool:
        """
        validates that mlflow should be tracked, both in user config and in env imports
        """
        return (
            self._tracked_platform is not None and mlconf.mlflow_tracking.is_enabled
        )  # TODO check about more config enables

    def log_model(
        self, model_uri: str, context: typing.Union[MLClientCtx, dict],
    ):
        """
        zips model dir and logs it and all artifacts
        """

        def zip_folder(folder_path: str, output_path: str):
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, folder_path))

        def schema_to_feature(schema) -> list:
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

        model_info = self._tracked_platform.models.get_model_info(model_uri=model_uri)
        model_zip = model_uri + ".zip"
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

    def post_run(self, context: typing.Union[MLClientCtx, dict], args: dict, db=None):
        experiment = "mlflow_experiment"
        self._post_run(context=context, args=args, experiment=experiment)










