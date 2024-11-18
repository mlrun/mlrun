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
import os
import shutil
import zipfile
from typing import Optional, Union

import numpy as np
import tensorflow as tf
from packaging import version
from tensorflow import keras

import mlrun
from mlrun.artifacts import Artifact
from mlrun.features import Feature

from .._common import without_mlrun_interface
from .._dl_common import DLModelHandler
from .mlrun_interface import TFKerasMLRunInterface
from .utils import TFKerasUtils


class TFKerasModelHandler(DLModelHandler):
    """
    Class for handling a tensorflow.keras model, enabling loading and saving it during runs.
    """

    # Framework name:
    FRAMEWORK_NAME = "tensorflow.keras"

    # Declare a type of an input sample:
    IOSample = Union[tf.Tensor, tf.TensorSpec, np.ndarray]

    class ModelFormats:
        """
        Model formats to pass to the 'TFKerasModelHandler' for loading and saving keras models.
        """

        SAVED_MODEL = "SavedModel"
        H5 = "h5"
        JSON_ARCHITECTURE_H5_WEIGHTS = "json_h5"

    class _LabelKeys:
        """
        Required labels keys to log with the model.
        """

        MODEL_FORMAT = "model-format"
        SAVE_TRACES = "save-traces"

    def __init__(
        self,
        model: keras.Model = None,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        model_format: str = ModelFormats.SAVED_MODEL,
        context: mlrun.MLClientCtx = None,
        modules_map: Optional[
            Union[dict[str, Union[None, str, list[str]]], str]
        ] = None,
        custom_objects_map: Optional[
            Union[dict[str, Union[str, list[str]]], str]
        ] = None,
        custom_objects_directory: Optional[str] = None,
        save_traces: bool = False,
        **kwargs,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading. Notice that if the model path
        given is of a previously logged model (store model object path), all of the other configurations will be loaded
        automatically as they were logged with the model, hence they are optional.

        :param model:                    Model to handle or None in case a loading parameters were supplied.
        :param model_path:               Path to the model's directory to load it from. The model files must start with
                                         the given model name and the directory must contain based on the given model
                                         formats:
                                         * SavedModel - A zip file 'model_name.zip' or a directory named 'model_name'.
                                         * H5 - A h5 file 'model_name.h5'.
                                         * Architecture and weights - The json file 'model_name.json' and h5 weight file
                                           'model_name.h5'.
                                         The model path can be also passed as a model object path in the following
                                         format: 'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'.
        :param model_name:               The model name for saving and logging the model:
                                         * Mandatory for loading the model from a local path.
                                         * If given a logged model (store model path) it will be read from the artifact.
                                         * If given a loaded model object and the model name is None, the name will be
                                           set to the model's object name / class.
        :param model_format:             The format to use for saving and loading the model. Should be passed as a
                                         member of the class 'ModelFormats'. Default: 'ModelFormats.SAVED_MODEL'.
        :param context:                  MLRun context to work with for logging the model.
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
        :param save_traces:              Whether to use functions saving (only available for the 'SavedModel' format)
                                         for loading the model later without the custom objects dictionary. Only from
                                         tensorflow version >= 2.4.0. Using this setting will increase the model saving
                                         size.

        :raise MLRunInvalidArgumentError: In case the input was incorrect:
                                          * Model format is unrecognized.
                                          * There was no model or model directory supplied.
                                          * 'save_traces' parameter was miss-used.
        """
        # Validate given format:
        if model_format not in [
            TFKerasModelHandler.ModelFormats.SAVED_MODEL,
            TFKerasModelHandler.ModelFormats.H5,
            TFKerasModelHandler.ModelFormats.JSON_ARCHITECTURE_H5_WEIGHTS,
        ]:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unrecognized model format: '{model_format}'. Please use one of the class members of "
                "'TFKerasModelHandler.ModelFormats'"
            )

        # Validate 'save_traces':
        if save_traces:
            if version.parse(tf.__version__) < version.parse("2.4.0"):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"The 'save_traces' parameter can be true only for tensorflow versions >= 2.4. Current "
                    f"version is {tf.__version__}"
                )
            if model_format != TFKerasModelHandler.ModelFormats.SAVED_MODEL:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "The 'save_traces' parameter is valid only for the 'SavedModel' format."
                )

        # If the model is given without a model name, set the model name:
        if model_name is None and model is not None:
            model_name = model.name

        # Store the configuration:
        self._model_format = model_format
        self._save_traces = save_traces

        # If the model format is architecture and weights, this will hold the weights file collected:
        self._weights_file = None  # type: str

        # Setup the base handler class:
        super().__init__(
            model=model,
            model_path=model_path,
            model_name=model_name,
            modules_map=modules_map,
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
            context=context,
            **kwargs,
        )

        # Set the required labels:
        self.set_labels()

    def set_labels(
        self,
        to_add: Optional[dict[str, Union[str, int, float]]] = None,
        to_remove: Optional[list[str]] = None,
    ):
        """
        Update the labels dictionary of this model artifact. There are required labels that cannot be edited or removed.

        :param to_add:    The labels to add.
        :param to_remove: A list of labels keys to remove.
        """
        # Update the user's labels:
        super().set_labels(to_add=to_add, to_remove=to_remove)

        # Set the required labels:
        self._labels[self._LabelKeys.MODEL_FORMAT] = self._model_format
        if self._model_format == self.ModelFormats.SAVED_MODEL:
            self._labels[self._LabelKeys.SAVE_TRACES] = self._save_traces

    # TODO: output_path won't work well with logging artifacts. Need to look into changing the logic of 'log_artifact'.
    @without_mlrun_interface(interface=TFKerasMLRunInterface)
    def save(
        self, output_path: Optional[str] = None, **kwargs
    ) -> Union[dict[str, Artifact], None]:
        """
        Save the handled model at the given output path. If a MLRun context is available, the saved model files will be
        logged and returned as artifacts.

        :param output_path: The full path to the directory to save the handled model at. If not given, the context
                            stored will be used to save the model in the default artifacts location.

        :return The saved model additional artifacts (if needed) dictionary if context is available and None otherwise.
        """
        super().save(output_path=output_path)

        # Setup the returning model artifacts list:
        artifacts = {}  # type: Dict[str, Artifact]

        # Set the output path:
        if output_path is None:
            output_path = os.path.join(self._context.artifact_path, self._model_name)

        # ModelFormats.H5 - Save as a h5 file:
        if self._model_format == TFKerasModelHandler.ModelFormats.H5:
            self._model_file = f"{self._model_name}.h5"
            self._model.save(self._model_file)

        # ModelFormats.SAVED_MODEL - Save as a SavedModel directory and zip its file:
        elif self._model_format == TFKerasModelHandler.ModelFormats.SAVED_MODEL:
            # Save it in a SavedModel format directory:
            if self._save_traces is True:
                # Save traces can only be used in versions >= 2.4, so only if its true we use it in the call:
                self._model.save(self._model_name, save_traces=self._save_traces)
            else:
                self._model.save(self._model_name)
            # Zip it:
            self._model_file = f"{self._model_name}.zip"
            shutil.make_archive(
                base_name=self._model_name, format="zip", base_dir=self._model_name
            )

        # ModelFormats.JSON_ARCHITECTURE_H5_WEIGHTS - Save as a json architecture and h5 weights files:
        else:
            # Save the model architecture (json):
            model_architecture = self._model.to_json()
            self._model_file = f"{self._model_name}.json"
            with open(self._model_file, "w") as json_file:
                json_file.write(model_architecture)
            # Save the model weights (h5):
            self._weights_file = f"{self._model_name}.h5"
            self._model.save_weights(self._weights_file)

        # Update the paths and log artifacts if context is available:
        if self._weights_file is not None:
            if self._context is not None:
                artifacts[self._get_weights_file_artifact_name()] = (
                    self._context.log_artifact(
                        self._weights_file,
                        local_path=self._weights_file,
                        artifact_path=output_path,
                        db_key=False,
                    )
                )

        return artifacts if self._context is not None else None

    def load(self, checkpoint: Optional[str] = None, **kwargs):
        """
        Load the specified model in this handler. If a checkpoint is required to be loaded, it can be given here
        according to the provided model path in the initialization of this handler. Additional parameters for the class
        initializer can be passed via the args list and kwargs dictionary.

        :param checkpoint: The checkpoint label to load the weights from. If the model path is of a store object, the
                           checkpoint will be taken from the logged checkpoints artifacts logged with the model. If the
                           model path is of a local directory, the checkpoint will be searched in it by the provided
                           name to this parameter.
        """
        # TODO: Add support for checkpoint loading after creating MLRun's checkpoint callback.
        if checkpoint is not None:
            raise NotImplementedError(
                "Loading a model using checkpoint is not yet implemented."
            )

        super().load()

        # ModelFormats.H5 - Load from a h5 file:
        if self._model_format == TFKerasModelHandler.ModelFormats.H5:
            self._model = keras.models.load_model(
                self._model_file, custom_objects=self._custom_objects
            )

        # ModelFormats.SAVED_MODEL - Load from a SavedModel directory:
        elif self._model_format == TFKerasModelHandler.ModelFormats.SAVED_MODEL:
            self._model = keras.models.load_model(
                self._model_file, custom_objects=self._custom_objects
            )

        # ModelFormats.JSON_ARCHITECTURE_H5_WEIGHTS - Load from a json architecture file and a h5 weights file:
        else:
            # Load the model architecture (json):
            with open(self._model_file) as json_file:
                model_architecture = json_file.read()
            self._model = keras.models.model_from_json(
                model_architecture, custom_objects=self._custom_objects
            )
            # Load the model weights (h5):
            self._model.load_weights(self._weights_file)

    def to_onnx(
        self,
        model_name: Optional[str] = None,
        optimize: bool = True,
        input_signature: Union[
            list[tf.TensorSpec], list[np.ndarray], tf.TensorSpec, np.ndarray
        ] = None,
        output_path: Optional[str] = None,
        log: Optional[bool] = None,
    ):
        """
        Convert the model in this handler to an ONNX model.
        :param model_name:      The name to give to the converted ONNX model. If not given the default name will be the
                                stored model name with the suffix '_onnx'.
        :param optimize:        Whether to optimize the ONNX model using 'onnxoptimizer' before saving the model.
                                Default: True.
        :param input_signature: An numpy.ndarray or tensorflow.TensorSpec that describe the input port (shape and data
                                type). If the model has multiple inputs, a list is expected in the order of the input
                                ports. If not provided, the method will try to extract the input signature of the model.
        :param output_path:     In order to save the ONNX model, pass here the output directory. The model file will be
                                named with the model name given. Default: None (not saving).
        :param log:             In order to log the ONNX model, pass True. If None, the model will be logged if this
                                handler has a MLRun context set. Default: None.

        :return: The converted ONNX model (onnx.ModelProto).

        :raise MLRunMissingDependencyError: If the onnx modules are missing in the interpreter.
        :raise MLRunInvalidArgumentError:   If the input signatures was not given and inputs are not set in the handler.
        """
        # Import onnx related modules:
        try:
            import tf2onnx

            from mlrun.frameworks.onnx import ONNXModelHandler
        except ModuleNotFoundError:
            raise mlrun.errors.MLRunMissingDependencyError(
                "ONNX conversion requires additional packages to be installed. "
                "Please run 'pip install mlrun[tensorflow]' to install MLRun's Tensorflow package."
            )

        # Set the onnx model name:
        model_name = self._get_default_onnx_model_name(model_name=model_name)

        # Set the input signature:
        if input_signature is None:
            if self._inputs is not None:
                # Parse the set input features:
                input_signature = [
                    tf.TensorSpec(
                        shape=input_feature.dims,
                        dtype=TFKerasUtils.convert_value_type_to_tf_dtype(
                            value_type=input_feature.value_type
                        ),
                    )
                    for input_feature in self._inputs
                ]
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "In order to convert the model to ONNX the 'input_signature' must be given or you can use one of "
                    "the 'set_inputs', 'read_inputs_from_model' methods."
                )
        elif not isinstance(input_signature, list):
            # Wrap it in a list:
            input_signature = [input_signature]

        # Set the output path:
        if output_path is not None:
            output_path = os.path.join(output_path, f"{model_name}.onnx")

        # Set the logging flag:
        log = self._context is not None if log is None else log

        # Convert to ONNX:
        model_proto, external_tensor_storage = tf2onnx.convert.from_keras(
            model=self._model, input_signature=input_signature, output_path=output_path
        )

        # Create a handler for the ONNX model:
        onnx_handler = ONNXModelHandler(
            model_name=model_name, model=model_proto, context=self._context
        )

        # Pass on the inputs and outputs properties:
        if self._inputs is not None:
            onnx_handler.set_inputs(features=self._inputs)
        if self._outputs is not None:
            onnx_handler.set_outputs(features=self._outputs)

        # Optimize the model if needed:
        if optimize:
            onnx_handler.optimize()
            # Save if logging is not required, as logging will save as well:
            if not log and output_path is not None:
                onnx_handler.save(output_path=output_path)

        # Log as a model object if needed:
        if log:
            onnx_handler.log()

        return onnx_handler.model

    def read_inputs_from_model(self):
        """
        Extract the inputs information out of the model and set it in the handler.

        :raise MLRunRuntimeError: If there is no model in this handler.
        """
        # Validate there is a model available:
        if self._model is None:
            raise mlrun.errors.MLRunRuntimeError(
                "The model in this handler was not loaded or given in initialization so the inputs cannot be read."
            )

        # Read the inputs:
        input_signature = [input_layer.type_spec for input_layer in self._model.inputs]

        # Set the inputs:
        self.set_inputs(from_sample=input_signature)

    def read_outputs_from_model(self):
        """
        Extract the outputs information out of the model and set it in the handler.

        :raise MLRunRuntimeError: If there is no model in this handler.
        """
        # Validate there is a model available:
        if self._model is None:
            raise mlrun.errors.MLRunRuntimeError(
                "The model in this handler was not loaded or given in initialization so the outputs cannot be read."
            )

        # Read the outputs:
        output_signature = [
            output_layer.type_spec for output_layer in self._model.outputs
        ]

        # Set the outputs:
        self.set_outputs(from_sample=output_signature)

    def _collect_files_from_store_object(self):
        """
        If the model path given is of a store object, collect the needed model files into this handler for later loading
        the model.
        """
        # Read the settings:
        self._model_format = self._model_artifact.labels[self._LabelKeys.MODEL_FORMAT]
        self._save_traces = self._model_artifact.labels.get(
            self._LabelKeys.SAVE_TRACES, None
        )

        # Read additional files according to the model format used:
        # # ModelFormats.SAVED_MODEL - Unzip the SavedModel archive:
        if self._model_format == TFKerasModelHandler.ModelFormats.SAVED_MODEL:
            # Unzip the SavedModel directory:
            with zipfile.ZipFile(self._model_file, "r") as zip_file:
                zip_file.extractall(os.path.dirname(self._model_file))
            # Set the model file to the unzipped directory:
            self._model_file = os.path.join(
                os.path.dirname(self._model_file), self._model_name
            )
        # # ModelFormats.JSON_ARCHITECTURE_H5_WEIGHTS - Get the weights file:
        elif (
            self._model_format
            == TFKerasModelHandler.ModelFormats.JSON_ARCHITECTURE_H5_WEIGHTS
        ):
            # Get the weights file:
            self._weights_file = self._extra_data[
                self._get_weights_file_artifact_name()
            ].local()

        # Continue collecting from abstract class:
        super()._collect_files_from_store_object()

    def _collect_files_from_local_path(self):
        """
        If the model path given is of a local path, search for the needed model files and collect them into this handler
        for later loading the model.

        :raise MLRunNotFoundError: If any of the required files are missing.
        """
        # ModelFormats.H5 - Get the h5 model file:
        if self._model_format == TFKerasModelHandler.ModelFormats.H5:
            self._model_file = os.path.join(self._model_path, f"{self._model_name}.h5")
            if not os.path.exists(self._model_file):
                raise mlrun.errors.MLRunNotFoundError(
                    f"The model file '{self._model_name}.h5' was not found within the given 'model_path': "
                    f"'{self._model_path}'"
                )

        # ModelFormats.SAVED_MODEL - Get the zip file and extract it, or simply locate the directory:
        elif self._model_format == TFKerasModelHandler.ModelFormats.SAVED_MODEL:
            self._model_file = os.path.join(self._model_path, f"{self._model_name}.zip")
            if os.path.exists(self._model_file):
                # Unzip it:
                with zipfile.ZipFile(self._model_file, "r") as zip_file:
                    zip_file.extractall(os.path.dirname(self._model_file))
                # Set the model file to the unzipped directory:
                self._model_file = os.path.join(
                    os.path.dirname(self._model_file), self._model_name
                )
            else:
                # Look for the SavedModel directory:
                self._model_file = os.path.join(self._model_path, self._model_name)
                if not os.path.exists(self._model_file):
                    raise mlrun.errors.MLRunNotFoundError(
                        f"There is no SavedModel zip archive '{self._model_name}.zip' or a SavedModel directory named "
                        f"'{self._model_name}' the given 'model_path': '{self._model_path}'"
                    )

        # ModelFormats.JSON_ARCHITECTURE_H5_WEIGHTS - Save as a json architecture and h5 weights files:
        else:
            # Locate the model architecture json file:
            self._model_file = f"{self._model_name}.json"
            if not os.path.exists(os.path.join(self._model_path, self._model_file)):
                raise mlrun.errors.MLRunNotFoundError(
                    f"The model architecture file '{self._model_file}' is missing in the given 'model_path': "
                    f"'{self._model_path}'"
                )
            # Locate the model weights h5 file:
            self._weights_file = f"{self._model_name}.h5"
            if not os.path.exists(os.path.join(self._model_path, self._weights_file)):
                raise mlrun.errors.MLRunNotFoundError(
                    f"The model weights file '{self._weights_file}' is missing in the given 'model_path': "
                    f"'{self._model_path}'"
                )

    def _read_sample(self, sample: IOSample) -> Feature:
        """
        Read the sample into a MLRun Feature.

        :param sample: The sample to read.

        :return: The created Feature.

        :raise MLRunInvalidArgumentError: In case the given sample type cannot be read.
        """
        # Supported types:
        if isinstance(sample, np.ndarray):
            return super()._read_sample(sample=sample)
        elif isinstance(sample, tf.TensorSpec):
            return Feature(
                name=sample.name,
                value_type=TFKerasUtils.convert_tf_dtype_to_value_type(
                    tf_dtype=sample.dtype
                ),
                dims=list(sample.shape),
            )
        elif isinstance(sample, tf.Tensor):
            return Feature(
                value_type=TFKerasUtils.convert_tf_dtype_to_value_type(
                    tf_dtype=sample.dtype
                ),
                dims=list(sample.shape),
            )

        # Unsupported type:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"The sample type given '{type(sample)}' is not supported. The input / output ports are readable from "
            f"samples of the following types: tf.Tensor, tf.TensorSpec, np.ndarray."
        )
