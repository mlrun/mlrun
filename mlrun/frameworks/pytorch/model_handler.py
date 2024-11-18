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
from typing import Optional, Union

import numpy as np
import torch
from torch.nn import Module

import mlrun
from mlrun.artifacts import Artifact
from mlrun.features import Feature

from .._dl_common import DLModelHandler
from .utils import PyTorchUtils


class PyTorchModelHandler(DLModelHandler):
    """
    Class for handling a PyTorch model, enabling loading and saving it during runs.
    """

    # Framework name:
    FRAMEWORK_NAME = "torch"

    # Declare a type of input sample:
    IOSample = Union[torch.Tensor, np.ndarray]

    class _LabelKeys:
        """
        Required labels keys to log with the model.
        """

        MODEL_CLASS_NAME = "model-class-name"

    def __init__(
        self,
        model: Module = None,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        model_class: Optional[Union[type[Module], str]] = None,
        modules_map: Optional[
            Union[dict[str, Union[None, str, list[str]]], str]
        ] = None,
        custom_objects_map: Optional[
            Union[dict[str, Union[str, list[str]]], str]
        ] = None,
        custom_objects_directory: Optional[str] = None,
        context: mlrun.MLClientCtx = None,
        **kwargs,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading.
        :param model:                    Model to handle or None in case a loading parameters were supplied.
        :param model_path:               Path to the model's directory with the saved '.pt' file. The file must start
                                         with the given model name. The model path can be also passed as a model object
                                         path in the following format:
                                         'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'.
        :param model_name:               The model name for saving and logging the model:
                                         * Mandatory for loading the model from a local path.
                                         * If given a logged model (store model path) it will be read from the artifact.
                                         * If given a loaded model object and the model name is None, the name will be
                                           set to the model's object name / class.
        :param model_class:              The model's class type object. Can be passed as the class's name (string) as
                                         well. The model class must appear in the custom objects / modules map
                                         dictionary / json. If the model path given is of a store object, this model
                                         class name will be read from the logged label of the model.
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

        :raise MLRunInvalidArgumentError: If the provided model path is of a local model files but the model class name
                                          and or the model name were not provided (= None).
        """
        # Validate a modules map or custom objects were provided:
        if (
            model_path is None
            and modules_map is None
            and (custom_objects_directory is None or custom_objects_directory is None)
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "At least 'modules_map' or both custom objects parameters: 'custom_objects_map' and "
                "'custom_objects_directory' are mandatory for the handler as the class must be located in a "
                "custom object python file or an installed module. Without one of them the model will not be able "
                "to be saved and logged"
            )

        # If the model is given try to set the model class:
        if model is not None:
            if model_class is None:
                model_class = type(model).__name__

        # Parse the class name (in case it was passed as a class type) and store it:
        if model_class is not None:
            self._model_class_name = (
                model_class if isinstance(model_class, str) else model_class.__name__
            )

        # Set up the base handler class:
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
        self._labels[self._LabelKeys.MODEL_CLASS_NAME] = self._model_class_name

    def save(
        self, output_path: Optional[str] = None, **kwargs
    ) -> Union[dict[str, Artifact], None]:
        """
        Save the handled model at the given output path.

        :param output_path: The full path to the directory to save the handled model at. If not given, the context
                            stored will be used to save the model in the default location.

        :return The saved model additional artifacts (if needed) dictionary if context is available and None otherwise.

        :raise MLRunRuntimeError:         In case there is no model initialized in this handler.
        :raise MLRunInvalidArgumentError: If an output path was not given, yet a context was not provided in
                                          initialization.
        """
        super().save(output_path=output_path)

        # Set the output path:
        if output_path is None:
            output_path = os.path.join(self._context.artifact_path, self._model_name)

        # Save the model:
        self._model_file = f"{self._model_name}.pt"
        torch.save(self._model.state_dict(), self._model_file)

        return None

    def load(self, checkpoint: Optional[str] = None, **kwargs):
        """
        Load the specified model in this handler. If a checkpoint is required to be loaded, it can be given here
        according to the provided model path in the initialization of this handler. Additional parameters for the class
        initializer can be passed via the args list and kwargs dictionary.

        :param checkpoint: The checkpoint label to load the weights from. If the model path is of a store object, the
                           checkpoint will be taken from the logged checkpoints artifacts logged with the model. If the
                           model path is of a local directory, the checkpoint will be searched in it by the provided
                           name to this parameter.

        :raise MLRunInvalidArgumentError: If the model's class is not in the custom objects map.
        """
        super().load()

        # Validate the model's class is in the custom objects map:
        if (
            self._model_class_name not in self._custom_objects
            and self._model_class_name not in self._modules
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The model class '{self._model_class_name}' was not found in the given custom objects map. The custom "
                f"objects map must include the model's class name in its values. Usually the model class should appear "
                f"last in the map dictionary as it is imported from the top to the bottom."
            )

        # Initialize the model:
        self._model = (
            self._custom_objects[self._model_class_name](**kwargs)
            if self._model_class_name in self._custom_objects
            else self._modules[self._model_class_name](**kwargs)
        )

        # Load the state dictionary into it:
        self._model.load_state_dict(torch.load(self._model_file))

    def to_onnx(
        self,
        model_name: Optional[str] = None,
        input_sample: Union[torch.Tensor, tuple[torch.Tensor, ...]] = None,
        input_layers_names: Optional[list[str]] = None,
        output_layers_names: Optional[list[str]] = None,
        dynamic_axes: Optional[dict[str, dict[int, str]]] = None,
        is_batched: bool = True,
        optimize: bool = True,
        output_path: Optional[str] = None,
        log: Optional[bool] = None,
    ):
        """
        Convert the model in this handler to an ONNX model. The layer names are optional, they do not change the
        semantics of the model, it is only for readability.

        :param model_name:          The name to give to the converted ONNX model. If not given the default name will be
                                    the stored model name with the suffix '_onnx'.
        :param optimize:            Whether to optimize the ONNX model using 'onnxoptimizer' before saving the model.
                                    Default: True.
        :param input_sample:        A torch.Tensor with the shape and data type of the expected input to the model. Can
                                    be passed as a tuple if the model expects multiple input tensors.
        :param input_layers_names:  List of names to assign to the input nodes of the graph in order. All of the other
                                    parameters (inner layers) can be set as well by passing additional names in the
                                    list. The order is by the order of the parameters in the model. If None, the inputs
                                    will be read from the handler's inputs. If it's also None, the default is:
                                    "input_0", "input_1", ...
        :param output_layers_names: List of names to assign to the output nodes of the graph in order. If None, the
                                    outputs will be read from the handler's outputs. If it's also None, the default
                                    is: "output_0" (for multiple outputs, this parameter must be provided).
        :param dynamic_axes:        If part of the input / output shape is dynamic, like (batch_size, 3, 32, 32) you can
                                    specify it by giving a dynamic axis to the input / output layer by its name as
                                    follows: {
                                        "input layer name": {0: "batch_size"},
                                        "output layer name": {0: "batch_size"},
                                    }
                                    If provided, the 'is_batched' flag will be ignored. Default: None.
        :param is_batched:          Whether to include a batch size as the first axis in every input and output layer.
                                    Default: True. Will be ignored if 'dynamic_axes' is provided.
        :param output_path:         In order to save the ONNX model, pass here the output directory. The model file will
                                    be named with the model name given. Default: None (not saving).
        :param log:                 In order to log the ONNX model, pass True. If None, the model will be logged if this
                                    handler has a MLRun context set. Default: None.

        :return: The converted ONNX model (onnx.ModelProto).

        :raise MLRunMissingDependencyError: If some of the ONNX packages are missing.
        """
        # Import onnx related modules:
        try:
            from mlrun.frameworks.onnx import ONNXModelHandler
        except ModuleNotFoundError:
            raise mlrun.errors.MLRunMissingDependencyError(
                "ONNX conversion requires additional packages to be installed. "
                "Please run 'pip install mlrun[pytorch]' to install MLRun's PyTorch package."
            )

        # Set the onnx model name:
        model_name = self._get_default_onnx_model_name(model_name=model_name)

        # Read the input signature in case its None:
        if input_sample is None:
            # Validate there are inputs set:
            if self._inputs is None:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "An input sample must be provided. Either use the method 'set_inputs' or pass an inputs sample "
                    "here."
                )
            # Parse the input features into a sample:
            input_sample = tuple(
                [
                    torch.zeros(
                        size=input_feature.dims,
                        dtype=PyTorchUtils.convert_value_type_to_torch_dtype(
                            value_type=input_feature.value_type
                        ),
                    )
                    for input_feature in self._inputs
                ]
            )
            if len(input_sample) == 1:
                input_sample = input_sample[0]

        # Set the default input layers names if not provided:
        if input_layers_names is None:
            input_layers_names = (
                (
                    [f"input_{i}" for i in range(len(input_sample))]
                    if isinstance(input_sample, tuple)
                    else ["input_0"]
                )
                if self._inputs is None
                else (
                    [
                        f"input_{i}" if layer.name == "" else layer.name
                        for i, layer in enumerate(self._inputs)
                    ]
                )
            )

        # Set the default output layers names if not provided:
        if output_layers_names is None:
            output_layers_names = (
                ["output_0"]
                if self._outputs is None
                else [
                    f"output_{i}" if layer.name == "" else layer.name
                    for i, layer in enumerate(self._outputs)
                ]
            )

        # Setup first axis to be a batch_size if needed:
        if dynamic_axes is None and is_batched:
            dynamic_axes = {
                layer: {0: "batch_size"}
                for layer in input_layers_names + output_layers_names
            }

        # Set the output model file:
        onnx_file = f"{model_name}.onnx"
        if output_path is None:
            output_path = "./"
        else:
            onnx_file = os.path.join(output_path, f"{model_name}.onnx")

        # Set the logging flag:
        log = self._context is not None if log is None else log

        # Convert to ONNX:
        torch.onnx.export(
            self._model,
            input_sample,
            onnx_file,
            input_names=input_layers_names,
            output_names=output_layers_names,
            dynamic_axes=dynamic_axes,
        )

        # Create a handler for the ONNX model:
        onnx_handler = ONNXModelHandler(
            model_name=model_name, model_path=output_path, context=self._context
        )

        # Pass on the inputs and outputs properties:
        if self._inputs is not None:
            onnx_handler.set_inputs(features=self._inputs)
        if self._outputs is not None:
            onnx_handler.set_outputs(features=self._outputs)

        # Load the ONNX model:
        onnx_handler.load()

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

    def _collect_files_from_store_object(self):
        """
        If the model path given is of a store object, collect the needed model files into this handler for later loading
        the model.
        """
        # Read the model's class name:
        self._model_class_name = self._model_artifact.labels[
            self._LabelKeys.MODEL_CLASS_NAME
        ]

        # Continue collecting from abstract class:
        super()._collect_files_from_store_object()

    def _collect_files_from_local_path(self):
        """
        If the model path given is of a local path, search for the needed model files and collect them into this handler
        for later loading the model.

        :raise MLRunInvalidArgumentError: If the provided model class name from the user was None.
        :raise MLRunNotFoundError:        If the weights '.pt' file was not found.
        """
        # Read the model class provided:
        if self._model_class_name is None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "The model class name must be provided when loading the model from local path. Otherwise, the handler "
                "will not be able to load the model."
            )

        # Collect the weights file:
        self._model_file = os.path.join(self._model_path, f"{self._model_name}.pt")
        if not os.path.exists(self._model_file):
            raise mlrun.errors.MLRunNotFoundError(
                f"The model weights file '{self._model_name}.pt' was not found within the given 'model_path': "
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
        elif isinstance(sample, torch.Tensor):
            return Feature(
                value_type=PyTorchUtils.convert_torch_dtype_to_value_type(
                    torch_dtype=sample.dtype
                ),
                dims=list(sample.shape),
            )

        # Unsupported type:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"The sample type given '{type(sample)}' is not supported. The input / output ports are readable from "
            f"samples of the following types: torch.Tensor, np.ndarray."
        )
