# Copyright 2023 Iguazio
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
#
import importlib.util
import json
import os
import shutil
import sys
import zipfile
from abc import ABC, abstractmethod
from types import MethodType
from typing import Any, Generic, Optional, Union

import numpy as np

import mlrun
from mlrun.artifacts import Artifact
from mlrun.execution import MLClientCtx
from mlrun.features import Feature

from .mlrun_interface import MLRunInterface
from .utils import CommonTypes, CommonUtils


class ModelHandler(ABC, Generic[CommonTypes.ModelType, CommonTypes.IOSampleType]):
    """
    An abstract interface for handling a model of the supported frameworks. The handler will support loading, saving
    and logging a model with all the required modules, custom objects and collected information about it.
    """

    # Framework name (Must be set when inheriting the class):
    FRAMEWORK_NAME = None  # type: str

    # Constant artifact names:
    _MODEL_FILE_ARTIFACT_NAME = "{}_model_file"
    _MODULES_MAP_ARTIFACT_NAME = "{}_modules_map.json"
    _CUSTOM_OBJECTS_MAP_ARTIFACT_NAME = "{}_custom_objects_map.json"
    _CUSTOM_OBJECTS_DIRECTORY_ARTIFACT_NAME = "{}_custom_objects.zip"

    # Constant defaults:
    _DEFAULT_ONNX_MODEL_NAME = "onnx_{}"

    def __init__(
        self,
        model: CommonTypes.ModelType = None,
        model_path: CommonTypes.PathType = None,
        model_name: Optional[str] = None,
        modules_map: Union[
            dict[str, Union[None, str, list[str]]], CommonTypes.PathType
        ] = None,
        custom_objects_map: Union[
            dict[str, Union[str, list[str]]], CommonTypes.PathType
        ] = None,
        custom_objects_directory: CommonTypes.PathType = None,
        context: MLClientCtx = None,
        **kwargs,
    ):
        """
        Initialize the handler. The model can be set here, so it won't require loading. Note you must provide at least
        one of `model` and `model_path`. If a model is not given, the files in the model path will be collected
        automatically to be ready for loading.

        :param model:                    Model to handle or None in case a loading parameters were supplied.
        :param model_path:               Path to the directory with the model files. Can be passed as a model object
                                         path in the following format:
                                         'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'
        :param model_name:               The model name for saving and logging the model:

                                         * Mandatory for loading the model from a local path.
                                         * If given a logged model (store model path) it will be read from the artifact.
                                         * If given a loaded model object and the model name is None, the name will be
                                           set to the model's object name / class.

        :param modules_map:              A dictionary of all the modules required for loading the model. Each key
                                         is a path to a module and its value is the object name to import from it. All
                                         the modules will be imported globally. If multiple objects needed to be
                                         imported from the same module a list can be given. The map can be passed as a
                                         path to a json file as well. For example:

                                         {
                                             "module1": None,  # => import module1
                                             "module2": ["func1", "func2"],  # => from module2 import func1, func2
                                             "module3.sub_module": "func3",  # => from module3.sub_module import func3
                                         }

                                         If the model path given is of a store object, the modules map will be read from
                                         the logged modules map artifact of the model.
        :param custom_objects_map:       A dictionary of all the custom objects required for loading the model. Each key
                                         is a path to a python file and its value is the custom object name to import
                                         from it. If multiple objects needed to be imported from the same py file a list
                                         can be given. The map can be passed as a path to a json file as well. For
                                         example:

                                         {
                                             "/.../custom_optimizer.py": "optimizer",
                                             "/.../custom_layers.py": ["layer1", "layer2"]
                                         }

                                         All the paths will be accessed from the given 'custom_objects_directory',
                                         meaning each py file will be read from 'custom_objects_directory/<MAP VALUE>'.
                                         If the model path given is of a store object, the custom objects map will be
                                         read from the logged custom object map artifact of the model.
                                         Notice: The custom objects will be imported in the order they came in this
                                         dictionary (or json). If a custom object is depended on another, make sure to
                                         put it below the one it relies on.
        :param custom_objects_directory: Path to the directory with all the python files required for the custom
                                         objects. Can be passed as a zip file as well (will be extracted during the run
                                         before loading the model). If the model path given is of a store object, the
                                         custom objects files will be read from the logged custom object artifact of the
                                         model.
        :param context:                  MLRun context to work with for logging the model.

        :raise MLRunInvalidArgumentError: In case one of the given parameters are invalid.
        """
        # Validate input:
        self._validate_model_parameters(model=model, model_path=model_path)
        self._validate_modules_parameter(modules_map=modules_map)
        self._validate_custom_objects_parameters(
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
        )

        # Set a default model name if needed - the class name of the given model:
        if model_name is None and model is not None:
            model_name = type(model).__name__

        # Store parameters:
        self._model_name = model_name
        self._model_path = model_path
        self._model = model
        self._modules_map = modules_map
        self._custom_objects_map = custom_objects_map
        self._custom_objects_directory = custom_objects_directory
        self._context = context

        # The imported modules from the map. None until the '_import_modules' method is called.
        self._modules = None  # type: Dict[str, Any]

        # The imported custom objects from the map. None until the '_import_custom_objects' method is called.
        self._custom_objects = None  # type: Dict[str, Any]

        # Local path to the model's primary file:
        self._model_file = kwargs.get("model_file", None)  # type: str

        # If the model path is of a store model object, this will be the 'ModelArtifact' object.
        self._model_artifact = kwargs.get("model_artifact", None)  # type: ModelArtifact

        # If the model path is of a store model object, this will be the extra data as DataItems ready to be downloaded.
        self._extra_data = kwargs.get("extra_data", {})  # type: Dict[str, CommonTypes.ExtraDataType]

        # If the model key is passed, override the default:
        self._model_key = kwargs.get("model_key", "model")

        # Setup additional properties for logging the model into a ModelArtifact:
        self._tag = ""
        self._inputs = None  # type: List[Feature]
        self._outputs = None  # type: List[Feature]
        self._labels = {}  # type: Dict[str, Union[str, int, float]]
        self._parameters = {}  # type: Dict[str, Union[str, int, float]]
        self._metrics = {}  # type: Dict[str, float]
        self._registered_artifacts = {}  # type: Dict[str, Artifact]

        # Set a flag to know if the user logged the model so its artifact is cached:
        self._is_logged = False

        # Collect the relevant files of the model into the handler (only in case the model was not provided):
        if model is None:
            self._collect_files()

    @property
    def model_name(self) -> str:
        """
        Get the handled model's name.

        :return: The handled model's name.
        """
        return self._model_name

    @property
    def model(self) -> CommonTypes.ModelType:
        """
        Get the handled model. Will return None in case the model is not initialized.

        :return: The handled model.
        """
        return self._model

    @property
    def model_file(self) -> str:
        """
        Get the model file path given to / located by this handler.

        :return: The model file path.
        """
        return self._model_file

    @property
    def context(self) -> mlrun.MLClientCtx:
        """
        Get the handler's MLRun context.

        :return: The handler's MLRun context.
        """
        return self._context

    @property
    def tag(self) -> str:
        """
        Get the model's tag.

        :return: The model's tag.
        """
        return self._tag

    @property
    def inputs(self) -> Union[list[Feature], None]:
        """
        Get the input ports features list of this model's artifact. If the inputs are not set, None will be returned.

        :return: The input ports features list if its set, otherwise None.
        """
        return self._inputs

    @property
    def outputs(self) -> Union[list[Feature], None]:
        """
        Get the output ports features list of this model's artifact. If the outputs are not set, None will be returned.

        :return: The output ports features list if its set, otherwise None.
        """
        return self._outputs

    @property
    def labels(self) -> dict[str, str]:
        """
        Get the labels dictionary of this model's artifact. These will be the labels that will be logged with the model.

        :return: The model's artifact labels.
        """
        return self._labels

    @property
    def parameters(self) -> dict[str, str]:
        """
        Get the parameters dictionary of this model's artifact. These will be the parameters that will be logged with
        the model.

        :return: The model's artifact parameters.
        """
        return self._parameters

    def get_artifacts(
        self, committed_only: bool = False
    ) -> dict[str, CommonTypes.ExtraDataType]:
        """
        Get the registered artifacts of this model's artifact. By default all the artifacts (logged and to be logged -
        committed only) will be returned. To get only the artifacts registered in the current run whom are committed and
        not logged yet, set the 'committed_only' flag to True.

        :param committed_only: Whether to return only the artifacts in queue to be logged or all the artifacts
                               registered to this model (logged and not logged).

        :return: The artifacts registered to this model.
        """
        if committed_only:
            return self._registered_artifacts
        return {**self._extra_data, **self._registered_artifacts}

    def set_model_name(self, model_name: str):
        """
        Set the handled model name. The 'save' and 'log' methods will use the new name for the files and logs. Keep in
        mind that changing the model's name before calling 'load' will fail as now the handler won't look for the
        correct files.

        :param model_name: The new model name to use.
        """
        self._model_name = model_name

    def set_context(self, context: MLClientCtx):
        """
        Set this handler MLRun context.

        :param context: The context to set to.
        """
        self._context = context

    def set_tag(self, tag: str):
        """
        Set the tag this model will be logged with.

        :param tag: The model tag to set.
        """
        self._tag = tag

    def set_inputs(
        self,
        from_sample: CommonTypes.IOSampleType = None,
        features: Optional[list[Feature]] = None,
        **kwargs,
    ):
        """
        Read the inputs property of this model to be logged along with it. The inputs can be set directly by passing the
        input features or to be read by a given input sample.

        :param from_sample: Read the inputs properties from a given input sample to the model.
        :param features:    List of MLRun.features.Feature to set.

        :raise MLRunInvalidArgumentError: If both parameters were passed.
        """
        # Validate parameters:
        if from_sample is not None and features is not None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "The inputs can either be read from a given sample or from a features list. Both parameters cannot be "
                "passed."
            )

        # Set the inputs:
        self._inputs = (
            features
            if features is not None
            else self._read_io_samples(samples=from_sample)
        )

    def set_outputs(
        self,
        from_sample: CommonTypes.IOSampleType = None,
        features: Optional[list[Feature]] = None,
        **kwargs,
    ):
        """
        Read the outputs property of this model to be logged along with it. The outputs can be set directly by passing
        the output features or to be read by a given output sample.

        :param from_sample: Read the outputs properties from a given output sample from the model.
        :param features:    List of MLRun.features.Feature to set.

        :raise MLRunInvalidArgumentError: If both parameters were passed.
        """
        # Validate parameters:
        if from_sample is not None and features is not None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "The outputs can either be read from a given sample or from a features list. Both parameters cannot be "
                "passed."
            )

        # Set the outputs:
        self._outputs = (
            features
            if features is not None
            else self._read_io_samples(samples=from_sample)
        )

    def set_labels(
        self,
        to_add: Optional[dict[str, Union[str, int, float]]] = None,
        to_remove: Optional[list[str]] = None,
    ):
        """
        Update the labels dictionary of this model artifact.

        :param to_add:    The labels to add.
        :param to_remove: A list of labels keys to remove.
        """
        # Update the labels:
        if to_add is not None:
            self._labels.update(to_add)

        # Remove labels:
        if to_remove is not None:
            for label in to_remove:
                self._labels.pop(label)

    def set_parameters(
        self,
        to_add: Optional[dict[str, Union[str, int, float]]] = None,
        to_remove: Optional[list[str]] = None,
    ):
        """
        Update the parameters dictionary of this model artifact.

        :param to_add:    The parameters to add.
        :param to_remove: A list of parameters keys to remove.
        """
        # Update the parameters:
        if to_add is not None:
            self._parameters.update(to_add)

        # Remove parameters:
        if to_remove is not None:
            for label in to_remove:
                self._parameters.pop(label)

    def set_metrics(
        self,
        to_add: Optional[dict[str, CommonTypes.ExtraDataType]] = None,
        to_remove: Optional[list[str]] = None,
    ):
        """
        Update the metrics dictionary of this model artifact.

        :param to_add:    The metrics to add.
        :param to_remove: A list of metrics keys to remove.
        """
        # Update the extra data:
        if to_add is not None:
            self._metrics.update(to_add)

        # Remove extra data:
        if to_remove is not None:
            for label in to_remove:
                self._metrics.pop(label)

    def set_extra_data(
        self,
        to_add: Optional[dict[str, CommonTypes.ExtraDataType]] = None,
        to_remove: Optional[list[str]] = None,
    ):
        """
        Update the extra data dictionary of this model artifact.

        :param to_add:    The extra data to add.
        :param to_remove: A list of extra data keys to remove.
        """
        # Update the extra data:
        if to_add is not None:
            self._extra_data.update(to_add)

        # Remove extra data:
        if to_remove is not None:
            for label in to_remove:
                self._extra_data.pop(label)

    def register_artifacts(
        self, artifacts: Union[Artifact, list[Artifact], dict[str, Artifact]]
    ):
        """
        Register the given artifacts, so they will be logged as extra data with the model of this handler. Notice: The
        artifacts will be logged only when either 'log' or 'update' are called.

        :param artifacts: The artifacts to register. Can be passed as a single artifact, a list of artifacts or an
                          artifacts dictionary. In case of single artifact or a list of artifacts, the artifacts key
                          will be used as its name in the extra data dictionary.
        """
        # If a single artifact is given, wrap in a list:
        if isinstance(artifacts, Artifact):
            artifacts = [artifacts]

        # If a list is given, prepare the artifacts dictionary:
        if isinstance(artifacts, list):
            artifacts = {artifact.key: artifact for artifact in artifacts}

        # Register the artifacts:
        self._registered_artifacts = {**self._registered_artifacts, **artifacts}

    @abstractmethod
    def save(
        self, output_path: CommonTypes.PathType = None, **kwargs
    ) -> Union[dict[str, Artifact], None]:
        """
        Save the handled model at the given output path.

        :param output_path:  The full path to the directory to save the handled model at. If not given, the context
                             stored will be used to save the model in the default artifacts location.

        :return The saved model artifacts dictionary if context is available and None otherwise.

        :raise MLRunRuntimeError:         In case there is no model initialized in this handler.
        :raise MLRunInvalidArgumentError: If an output path was not given, yet a context was not provided in
                                          initialization.
        """
        if self._model is None:
            raise mlrun.errors.MLRunRuntimeError(
                "Model cannot be save as it was not given in initialization or loaded during this run."
            )
        if output_path is None and self._context is None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "An output path was not given and a context was not provided during the initialization of "
                "this model handler. To save the model, one of the two parameters must be supplied."
            )
        return None

    @abstractmethod
    def load(self, **kwargs):
        """
        Load the specified model in this handler. The custom objects will be imported right before loading the model. To
        access the model, call the 'model' property.
        """
        # If a model instance is already loaded, delete it from memory:
        if self._model:
            del self._model

        # Import the modules if needed (will be only imported once):
        if self._modules is None:
            self._import_modules()

        # Import the custom objects if needed (will be only imported once):
        if self._custom_objects is None:
            self._import_custom_objects()

    @abstractmethod
    def to_onnx(
        self, model_name: Optional[str] = None, optimize: bool = True, **kwargs
    ):
        """
        Convert the model in this handler to an ONNX model.

        :param model_name: The name to give to the converted ONNX model. If not given the default name will be the
                           stored model name with the suffix '_onnx'.
        :param optimize:   Whether to optimize the ONNX model using 'onnxoptimizer' before saving the model. Default:
                           True.

        :return: The converted ONNX model (onnx.ModelProto).
        """
        pass

    def log(
        self,
        tag: str = "",
        labels: Optional[dict[str, Union[str, int, float]]] = None,
        parameters: Optional[dict[str, Union[str, int, float]]] = None,
        inputs: Optional[list[Feature]] = None,
        outputs: Optional[list[Feature]] = None,
        metrics: Optional[dict[str, Union[int, float]]] = None,
        artifacts: Optional[dict[str, Artifact]] = None,
        extra_data: Optional[dict[str, CommonTypes.ExtraDataType]] = None,
        **kwargs,
    ):
        """
        Log the model held by this handler into the MLRun context provided. The stored values such as labels, parameters
        and artifacts will be used and updated where tag, inputs and outputs will be overridden if given here.

        :param tag:        Tag of a version to give to the logged model. Will override the stored tag in this handler.
        :param labels:     Labels to log the model with. Will be joined to the labels set.
        :param parameters: Parameters to log with the model. Will be joined to the parameters set.
        :param inputs:     A list of features this model expects to receive - the model's input ports. If already set,
                           will be overridden by the inputs given here.
        :param outputs:    A list of features this model expects to return - the model's output ports. If already set,
                           will be overridden by the outputs given here.
        :param metrics:    Metrics results to log with the model.
        :param artifacts:  Artifacts to log the model with. Will be joined to the registered artifacts and added to the
                           extra data.
        :param extra_data: Extra data to log with the model.

        :raise MLRunRuntimeError:         In case is no model in this handler.
        :raise MLRunInvalidArgumentError: If a context is missing.
        """
        # Validate there is a model and context:
        if self._model is None:
            raise mlrun.errors.MLRunRuntimeError(
                "Model cannot be logged as it was not given in initialization or loaded during this run."
            )
        if self._context is None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Cannot log model if a context was not provided during initialization."
            )

        # Save the model:
        model_artifacts = self.save()

        # Log the imported modules:
        modules_artifacts = self._log_modules() if self._modules_map is not None else {}

        # Log the custom objects:
        custom_objects_artifacts = (
            self._log_custom_objects() if self._custom_objects_map is not None else {}
        )

        # Read inputs and outputs ports:
        if inputs is not None:
            self.set_inputs(features=inputs)
        if outputs is not None:
            self.set_outputs(features=outputs)

        # Update the tag:
        if tag != "":
            self.set_tag(tag=tag)

        # Update labels, parameters and metrics:
        self.set_labels(to_add=labels)
        self.set_parameters(to_add=parameters)
        self.set_metrics(to_add=metrics)

        # Update the extra data:
        self._extra_data = {
            **self._extra_data,
            **(model_artifacts if model_artifacts is not None else {}),
            **self._registered_artifacts,
            **(artifacts if artifacts is not None else {}),
            **(extra_data if extra_data is not None else {}),
            **modules_artifacts,
            **custom_objects_artifacts,
        }
        self._registered_artifacts = {}

        # Log the model:
        self._model_artifact = self._context.log_model(
            key=self._model_key,
            db_key=self._model_name,
            model_file=self._model_file,
            tag=self._tag,
            inputs=self._inputs,
            outputs=self._outputs,
            framework=self.FRAMEWORK_NAME,
            labels=self._labels,
            parameters=self._parameters,
            metrics=self._metrics,
            extra_data={
                k: v
                for k, v in self._extra_data.items()
                if not isinstance(v, mlrun.DataItem)
            },
            algorithm=kwargs.get("algorithm", None),
            training_set=kwargs.get("sample_set", None),
            label_column=kwargs.get("target_columns", None),
            feature_vector=kwargs.get("feature_vector", None),
            feature_weights=kwargs.get("feature_weights", None),
        )

        # Mark the model is logged:
        self._is_logged = True

    def update(
        self,
        labels: Optional[dict[str, Union[str, int, float]]] = None,
        parameters: Optional[dict[str, Union[str, int, float]]] = None,
        inputs: Optional[list[Feature]] = None,
        outputs: Optional[list[Feature]] = None,
        metrics: Optional[dict[str, Union[int, float]]] = None,
        artifacts: Optional[dict[str, Artifact]] = None,
        extra_data: Optional[dict[str, CommonTypes.ExtraDataType]] = None,
        **kwargs,
    ):
        """
        Update the model held by this handler into the MLRun context provided, updating the model's artifact properties
        in the same model path provided.

        :param labels:     Labels to update or add to the model.
        :param parameters: Parameters to update or add to the model.
        :param inputs:     A list of features this model expects to receive - the model's input ports. If already set,
                           will be overridden by the inputs given here.
        :param outputs:    A list of features this model expects to return - the model's output ports. If already set,
                           will be overridden by the outputs given here.
        :param metrics:    Metrics results to log with the model.
        :param artifacts:  Artifacts to update or add to the model. Will be joined to the registered artifacts and added
                           to the extra data.
        :param extra_data: Extra data to update or add to the model.

        :raise MLRunInvalidArgumentError: In case a context is missing or the model path in this handler is missing or
                                          not of a store object.
        """
        # Validate model path:
        if self._model_artifact is None:
            if self._model_path is None:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Cannot update model if 'model_path' is not provided or if the model was never logged with this "
                    "handler."
                )
            elif not mlrun.datastore.is_store_uri(self._model_path):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "To update a model artifact the 'model_path' must be a store object."
                )

        # Read inputs and outputs ports:
        if inputs is not None:
            self.set_inputs(features=inputs)
        if outputs is not None:
            self.set_outputs(features=outputs)

        # Update labels and parameters:
        self.set_labels(to_add=labels)
        self.set_parameters(to_add=parameters)

        # Update the extra data:
        self._extra_data = {
            **self._extra_data,
            **self._registered_artifacts,
            **(artifacts if artifacts is not None else {}),
            **(extra_data if extra_data is not None else {}),
        }
        self._registered_artifacts = {}

        # Get the model artifact. If the model was logged during this run, use the artifact, otherwise use the
        # user's given model path:
        model_artifact = (
            self._context.get_artifact(self._model_name)
            if self._is_logged
            else self._model_path
        )

        # Update the model artifact:
        self._model_artifact = mlrun.artifacts.update_model(
            model_artifact=model_artifact,
            labels=self._labels,
            parameters=self._parameters,
            inputs=self._inputs,
            outputs=self._outputs,
            metrics=metrics,
            extra_data={
                k: v
                for k, v in self._extra_data.items()
                if not isinstance(v, mlrun.DataItem)
            },
            feature_vector=kwargs.get("feature_vector", None),
            feature_weights=kwargs.get("feature_weights", None),
            store_object=not self._is_logged,  # If the model was not logged, store the updated model in the database.
        )
        if self._is_logged:
            self._context.update_artifact(
                self._model_artifact
            )  # Update the cached model to the database.

    def _collect_files_from_store_object(self):
        """
        If the model path given is of a store object, collect the needed model files into this handler for later loading
        the model.
        """
        # Read the model artifact information:
        if len(self._model_artifact.inputs) > 0:
            self.set_inputs(features=list(self._model_artifact.inputs))
        if len(self._model_artifact.outputs) > 0:
            self.set_outputs(features=list(self._model_artifact.outputs))
        self.set_labels(to_add=self._model_artifact.labels)
        self.set_parameters(to_add=self._model_artifact.parameters)

        # Read the modules:
        if self._get_modules_map_artifact_name() in self._extra_data:
            self._modules_map = self._extra_data[
                self._get_modules_map_artifact_name()
            ].local()
        else:
            self._modules_map = None

        # Read the custom objects:
        if self._get_custom_objects_map_artifact_name() in self._extra_data:
            self._custom_objects_map = self._extra_data[
                self._get_custom_objects_map_artifact_name()
            ].local()
            self._custom_objects_directory = self._extra_data[
                self._get_custom_objects_directory_artifact_name()
            ].local()
        else:
            self._custom_objects_map = None
            self._custom_objects_directory = None

    @abstractmethod
    def _collect_files_from_local_path(self):
        """
        If the model path given is of a local path, search for the needed model files and collect them into this handler
        for later loading the model.
        """
        pass

    def _get_model_file_artifact_name(self) -> str:
        """
        Get the standard name for the model file artifact.

        :return: The model file artifact name.
        """
        return self._MODEL_FILE_ARTIFACT_NAME.format(self._model_name)

    def _get_modules_map_artifact_name(self) -> str:
        """
        Get the standard name for the modules map json artifact.

        :return: The modules map json artifact name.
        """
        return self._MODULES_MAP_ARTIFACT_NAME.format(self._model_name)

    def _get_custom_objects_map_artifact_name(self) -> str:
        """
        Get the standard name for the custom objects map json artifact.

        :return: The custom objects map json artifact name.
        """
        return self._CUSTOM_OBJECTS_MAP_ARTIFACT_NAME.format(self._model_name)

    def _get_custom_objects_directory_artifact_name(self) -> str:
        """
        Get the standard name for the custom objects directory zip artifact.

        :return: The custom objects directory zip artifact name.
        """
        return self._CUSTOM_OBJECTS_DIRECTORY_ARTIFACT_NAME.format(self._model_name)

    def _get_default_onnx_model_name(self, model_name: Union[str, None]):
        """
        Check if the given model name is None and if so will generate the default ONNX model name: 'onnx_<MODEL_NAME>'.

        :param model_name: The model name to check.

        :return: The given model name if its not None or the default ONNX model name.
        """
        return (
            self._DEFAULT_ONNX_MODEL_NAME.format(self._model_name)
            if model_name is None
            else model_name
        )

    def _collect_files(self):
        """
        Collect the files from the given model path.

        :raise MLRunInvalidArgumentError: In case the model path was not provided.
        """
        # Validate model path is set:
        if self._model_path is None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "In order to collect the model's files a model path must be provided."
            )

        # Collect by the path's type:
        if mlrun.datastore.is_store_uri(self._model_path):
            # Check if the model object was already downloaded:
            if (
                self._model_file is None
                and self._model_artifact is None
                and self._extra_data == {}
            ):
                # Get the artifact and model file along with its extra data:
                (
                    self._model_file,
                    self._model_artifact,
                    self._extra_data,
                ) = mlrun.artifacts.get_model(self._model_path)
            # Check if the model name was not provided:
            if self._model_name is None:
                self._model_name = self._model_artifact.db_key
            # Continue to collect the files from the store object each framework requires:
            self._collect_files_from_store_object()
        else:
            if self._model_name is None:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "The model name must be provided in the handler's initialization in order to collect the required "
                    "model files from a local path."
                )
            self._collect_files_from_local_path()

    def _import_modules(self):
        """
        Import the modules from the map provided.
        """
        # Initialize the custom objects dictionary:
        self._modules = {}

        # Check if modules parameters were provided:
        if self._modules_map is None:
            return

        # Read the modules map if given as a json:
        if isinstance(self._modules_map, str):
            with open(self._modules_map) as map_json_file:
                self._modules_map = json.loads(map_json_file.read())

        # Start importing the modules according to the map:
        for module_path, objects_names in self._modules_map.items():
            self._modules = {
                **self._modules,
                **self._import_module(
                    module_path=module_path,
                    objects_names=(
                        objects_names
                        if isinstance(objects_names, list) or objects_names is None
                        else [objects_names]
                    ),
                ),
            }

    def _import_custom_objects(self):
        """
        Import the custom objects from the map and directory provided.
        """
        # Initialize the custom objects dictionary:
        self._custom_objects = {}

        # Check if custom objects parameters were provided:
        if self._custom_objects_map is None:
            return

        # Read the custom objects map if given as a json:
        if isinstance(self._custom_objects_map, str):
            with open(self._custom_objects_map) as map_json_file:
                self._custom_objects_map = json.loads(map_json_file.read())

        # Unzip the custom objects files if the directory was given as a zip:
        if not os.path.isdir(self._custom_objects_directory):
            with zipfile.ZipFile(self._custom_objects_directory, "r") as zip_file:
                # Update the root directory of all the custom objects py files:
                self._custom_objects_directory = os.path.join(
                    os.path.dirname(self._custom_objects_directory),
                    os.path.basename(self._custom_objects_directory).rsplit(".", 1)[0],
                )
                # Extract the zip files into it:
                zip_file.extractall(self._custom_objects_directory)

        # Start importing the custom objects according to the map:
        for py_file, custom_objects_names in self._custom_objects_map.items():
            self._custom_objects = {
                **self._custom_objects,
                **self._import_custom_object(
                    py_file_path=os.path.abspath(
                        os.path.join(self._custom_objects_directory, py_file)
                    ),
                    objects_names=(
                        custom_objects_names
                        if isinstance(custom_objects_names, list)
                        else [custom_objects_names]
                    ),
                ),
            }

    def _log_modules(self) -> dict[str, Artifact]:
        """
        Log the modules, returning the modules map json file logged as an artifact.

        :return: The logged artifact in an 'extra data' style to be logged with the model.
        """
        # Initialize the returning artifacts dictionary:
        artifacts = {}

        # Create the custom objects map json file:
        modules_map_json = self._get_modules_map_artifact_name()
        if isinstance(self._modules_map, str):
            # The modules map is still a json path (model was not loaded but given as a live object):
            shutil.copy(self._modules_map, modules_map_json)
        else:
            # Dump the dictionary to json:
            with open(modules_map_json, "w") as json_file:
                json.dump(self._modules_map, json_file, indent=4)

        # Log the json file artifact:
        artifacts[modules_map_json] = self._context.log_artifact(
            modules_map_json,
            local_path=modules_map_json,
            artifact_path=self._context.artifact_path,
            db_key=False,
        )

        return artifacts

    def _log_custom_objects(self) -> dict[str, Artifact]:
        """
        Log the custom objects, returning their artifacts:

        * custom objects map json file - creating a json file from the custom objects map dictionary and logging it as
          an artifact.
        * custom objects directory zip - zipping the model's given custom objects directory along all of its content.

        :return: The logged artifacts in an 'extra data' style to be logged with the model.
        """
        # Initialize the returning artifacts dictionary:
        artifacts = {}

        # Create the custom objects map json file:
        custom_objects_map_json = self._get_custom_objects_map_artifact_name()
        if isinstance(self._custom_objects_map, str):
            # The custom objects map is still a json path (model was not loaded but given as a live object):
            shutil.copy(self._custom_objects_map, custom_objects_map_json)
        else:
            # Dump the dictionary to json:
            with open(custom_objects_map_json, "w") as json_file:
                json.dump(self._custom_objects_map, json_file, indent=4)

        # Log the json file artifact:
        artifacts[custom_objects_map_json] = self._context.log_artifact(
            custom_objects_map_json,
            local_path=custom_objects_map_json,
            artifact_path=self._context.artifact_path,
        )

        # Zip the custom objects directory:
        custom_objects_zip = self._get_custom_objects_directory_artifact_name()
        if self._custom_objects_directory.endswith(".zip"):
            # The custom objects are still zipped (model was not loaded but given as a live object):
            shutil.copy(self._custom_objects_directory, custom_objects_zip)
        else:
            # Zip the custom objects contents:
            shutil.make_archive(
                base_name=custom_objects_zip.rsplit(".", 1)[0],
                format="zip",
                root_dir=os.path.abspath(self._custom_objects_directory),
            )

        # Log the zip file artifact:
        artifacts[custom_objects_zip] = self._context.log_artifact(
            custom_objects_zip,
            local_path=custom_objects_zip,
            artifact_path=self._context.artifact_path,
        )

        return artifacts

    def _read_io_samples(
        self,
        samples: Union[CommonTypes.IOSampleType, list[CommonTypes.IOSampleType]],
    ) -> list[Feature]:
        """
        Read the given inputs / output sample to / from the model into a list of MLRun Features (ports) to log in
        the model's artifact.

        :param samples: The given inputs / output sample to / from the model.

        :return: The generated ports list.
        """
        # If there is only one input, wrap in a list:
        if not (isinstance(samples, list) or isinstance(samples, tuple)):
            samples = [samples]

        return [self._read_sample(sample=sample) for sample in samples]

    def _read_sample(self, sample: CommonTypes.IOSampleType) -> Feature:
        """
        Read the sample into a MLRun Feature. This abstract class is reading samples of 'numpy.ndarray'. For further
        types of samples, please inherit this method.

        :param sample: The sample to read.

        :return: The created Feature.

        :raise MLRunInvalidArgumentError: In case the given sample type cannot be read.
        """
        # Supported types:
        if isinstance(sample, np.ndarray):
            return Feature(
                value_type=CommonUtils.convert_np_dtype_to_value_type(
                    np_dtype=sample.dtype
                ),
                dims=list(sample.shape),
            )

        # Unsupported type:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"The sample type given '{type(sample)}' is not supported. The input / output ports are readable from "
            f"samples of the following types: np.ndarray"
        )

    @staticmethod
    def _validate_model_parameters(model_path: str, model: CommonTypes.ModelType):
        """
        Validate the given model parameters.

        :param model_path: Path to the directory with the model files. Can be passed as a model object path in the
                           following format: 'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'
        :param model:      Model to handle or None in case a loading parameters were supplied.

        :raise MLRunInvalidArgumentError: If both parameters were None or both parameters were provided.
        """
        if model_path is None and model is None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "At least one of 'model' or 'model_path' must be provided to the model handler."
            )

    @staticmethod
    def _validate_modules_parameter(
        modules_map: Union[dict[str, Union[None, str, list[str]]], str],
    ):
        """
        Validate the given modules parameter.

        :param modules_map: A dictionary of all the modules required for loading the model. Each key is a path to a
                            module and its value is the object name to import from it. All the modules will be imported
                            globally. If multiple objects needed to be imported from the same module a list can be
                            given. The map can be passed as a path to a json file as well. For example:
                            {
                                "module1": None,  # => import module1
                                "module2": ["func1", "func2"],  # => from module2 import func1, func2
                                "module3.sub_module": "func3",  # => from module3.sub_module import func3
                            }
                            If the model path given is of a store object, the modules map will be read from the logged
                            modules map artifact of the model.

        :raise MLRunInvalidArgumentError: If the modules map is in incorrect file format or not exist.
        """
        if modules_map is not None:
            if isinstance(modules_map, str):
                if not (modules_map.endswith(".json") and os.path.exists(modules_map)):
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"The 'modules_map' is either not found or not a path to a json file. Received: '{modules_map}'"
                    )

    @staticmethod
    def _validate_custom_objects_parameters(
        custom_objects_map: Union[dict[str, Union[str, list[str]]], str],
        custom_objects_directory: str,
    ):
        """
        Validate the given custom objects parameters.

        :param custom_objects_map:       A dictionary of all the custom objects required for loading the model. Each key
                                         is a path to a python file and its value is the custom object name to import
                                         from it. If multiple objects needed to be imported from the same py file a list
                                         can be given. The map can be passed as a path to a json file as well. For
                                         example:
                                         {
                                             "/.../custom_optimizer.py": "optimizer",
                                             "/.../custom_layers.py": ["layer1", "layer2"]
                                         }
                                         All the paths will be accessed from the given 'custom_objects_directory',
                                         meaning each py file will be read from 'custom_objects_directory/<MAP VALUE>'.
                                         Notice: The custom objects will be imported in the order they came in this
                                         dictionary (or json). If a custom object is depended on another, make sure to
                                         put it below the one it relies on.
        :param custom_objects_directory: Path to the directory with all the python files required for the custom
                                         objects. Can be passed as a zip file as well (will be extracted during the run
                                         before loading the model).

        :raise MLRunInvalidArgumentError: If the custom objects directory is given without the map or if the paths were
                                          in incorrect file formats or not exist.
        """
        # Validate that if one is provided (not None), both are provided:
        if (custom_objects_map is not None and custom_objects_directory is None) or (
            custom_objects_map is None and custom_objects_directory is not None
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Either 'custom_objects_map' or 'custom_objects_directory' are None. Custom objects must be supplied "
                "with the custom object map dictionary (or json) and the directory (or zip) with all the python files."
            )

        # Validate that if the map is a path, it is a path to a json file:
        if custom_objects_map is not None:
            if isinstance(custom_objects_map, str):
                if not (
                    custom_objects_map.endswith(".json")
                    and os.path.exists(custom_objects_map)
                ):
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"The 'custom_objects_map' is either not found or not a path to a json file. "
                        f"received: '{custom_objects_map}'"
                    )

        # Validate that the path is of a directory or a zip file:
        if custom_objects_directory is not None:
            if not (
                os.path.isdir(custom_objects_directory)
                or (
                    custom_objects_directory.endswith(".zip")
                    and os.path.exists(custom_objects_directory)
                )
            ):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"The 'custom_objects_directory' is either not found or not a directory / zip file, "
                    f"received: '{custom_objects_directory}'"
                )

    @staticmethod
    def _import_module(
        module_path: str, objects_names: Union[list[str], None]
    ) -> dict[str, Any]:
        """
        Import the given objects by their names from the given module path by the following rules:

        * If 'objects_names' is None: import 'module_path'.
        * Otherwise: from 'module_path' import 'object_name'.

        If an object specified is already imported, a reference would simply be returned.

        :param module_path:   Path to the module with the objects to import.
        :param objects_names: The objects names to be imported from the given module.

        :return: The imported objects dictionary where the keys are the objects names and the values are their imported
                 objects.
        """
        # Initialize the imports dictionary:
        module_imports = {}

        # Check if the module is already imported:
        if module_path in sys.modules:
            # It is already imported:
            module = sys.modules[module_path]
        else:
            # Import the module:
            module = importlib.import_module(module_path)

        # Check what to import from the module:
        if objects_names is None:
            # Import the entire module (import X):
            module_imports[module_path] = module
        else:
            # Import multiple objects (from X import Y, Z, ...):
            module_imports = {
                object_name: getattr(module, object_name)
                for object_name in objects_names
            }

        # Update the globals dictionary with the module imports:
        globals().update(module_imports)

        return module_imports

    @staticmethod
    def _import_custom_object(
        py_file_path: str, objects_names: list[str]
    ) -> dict[str, Any]:
        """
        Import the given objects by their names from the given python file as: from 'py_file_path' import 'object_name'.
        If an object specified is already imported, a reference would simply be returned.

        :param py_file_path:  Path to the python file with the objects code.
        :param objects_names: The objects names to be imported from the given python file.

        :return: The imported objects dictionary where the keys are the objects names and the values are their imported
                 objects.
        """
        # Initialize the imports dictionary:
        objects_imports = {}

        # Go through the given classes:
        for object_name in objects_names:
            if object_name in sys.modules:
                # It is already imported:
                objects_imports[object_name] = sys.modules[object_name]
            else:
                # Import the custom object:
                spec = importlib.util.spec_from_file_location(
                    name=object_name, location=py_file_path
                )
                module = importlib.util.module_from_spec(spec=spec)
                spec.loader.exec_module(module)
                # Get the imported class and store it:
                objects_imports[object_name] = getattr(module, object_name)

        return objects_imports


def with_mlrun_interface(interface: type[MLRunInterface]):
    """
    Decorator configure for decorating a ModelHandler method (expecting 'self' to be the first argument) to add the
    given MLRun interface into the model before executing the method and remove it afterwards.

    :param interface: The MLRun interface to add.

    :return: The method decorator.
    """

    def decorator(model_handler_method: MethodType):
        def wrapper(model_handler: ModelHandler, *args, **kwargs):
            # Check if the interface is applied to the model inside the handler:
            is_applied = interface.is_applied(obj=model_handler.model)
            # If the interface is not applied, add it:
            if not is_applied:
                interface.add_interface(obj=model_handler.model)
            # Call the method:
            returned_value = model_handler_method(self=model_handler, *args, **kwargs)
            # If the interface was not applied, remove it:
            if not is_applied:
                interface.remove_interface(obj=model_handler.model)
            return returned_value

        return wrapper

    return decorator


def without_mlrun_interface(interface: type[MLRunInterface]):
    """
    Decorator configure for decorating a ModelHandler method (expecting 'self' to be the first argument) to remove the
    given MLRun interface from the model before executing the method and restore it afterwards.

    :param interface: The MLRun interface to remove.

    :return: The method decorator.
    """

    def decorator(model_handler_method: MethodType):
        def wrapper(model_handler: ModelHandler, *args, **kwargs):
            # Check if the interface is applied to the model inside the handler:
            is_applied = interface.is_applied(obj=model_handler.model)
            # If the interface is applied, remove it:
            restoration_information = None
            if is_applied:
                restoration_information = interface.remove_interface(
                    obj=model_handler.model
                )
            # Call the method:
            returned_value = model_handler_method(self=model_handler, *args, **kwargs)
            # If the interface was applied, add it:
            if is_applied:
                interface.add_interface(
                    obj=model_handler.model,
                    restoration=restoration_information,
                )
            return returned_value

        return wrapper

    return decorator
