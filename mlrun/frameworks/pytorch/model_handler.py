import os
from typing import Dict, List, Type, Union

import numpy as np
import torch
from torch.nn import Module

import mlrun
from mlrun.artifacts import Artifact
from mlrun.data_types import ValueType
from mlrun.features import Feature
from mlrun.frameworks._dl_common import DLModelHandler


class PyTorchModelHandler(DLModelHandler):
    """
    Class for handling a PyTorch model, enabling loading and saving it during runs.
    """

    # Framework name:
    _FRAMEWORK_NAME = "pytorch"

    # Declare a type of an input sample:
    IOSample = Union[torch.Tensor, np.ndarray]

    class _LabelKeys:
        """
        Required labels keys to log with the model.
        """

        MODEL_CLASS_NAME = "model-class-name"

    def __init__(
        self,
        model_name: str,
        model_class: Union[Type[Module], str] = None,
        custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
        custom_objects_directory: str = None,
        model_path: str = None,
        model: Module = None,
        context: mlrun.MLClientCtx = None,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading.
        :param model_name:               The model name for saving and logging the model.
        :param model_class:              The model's class type object. Can be passed as the class's name (string) as
                                         well. The model class must appear in the custom objects map dictionary / json.
                                         If the model path given is of a store object, this model class name will be
                                         read from the logged label of the model.
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
        :param model_path:               Path to the model's directory with the saved '.pt' file. The file must start
                                         with the given model name. The model path can be also passed as a model object
                                         path in the following format:
                                         'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'.
        :param model:                    Model to handle or None in case a loading parameters were supplied.
        :param context:                  MLRun context to work with for logging the model.

        :raise MLRunInvalidArgumentError: If the provided model path is of a local model files but the model class name
                                          was not provided (=None).
        """
        # Store the model's class name:
        if model is not None:
            # Check if no value was provided:
            if model_class is None:
                # Take it from the model provided:
                model_class = type(model).__name__
            # Parse the class name and store it:
            self._model_class_name = (
                model_class if isinstance(model_class, str) else model_class.__name__
            )
        else:
            # Store the given value and edit later in one of the 'collect_files_...' methods:
            self._model_class_name = model_class

        # Setup the base handler class:
        super(PyTorchModelHandler, self).__init__(
            model_name=model_name,
            model_path=model_path,
            model=model,
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
            context=context,
        )

        # Set the required labels:
        self.update_labels()

    def update_labels(
        self,
        to_update: Dict[str, Union[str, int, float]] = None,
        to_remove: List[str] = None,
    ):
        """
        Update the labels dictionary of this model artifact. There are required labels that cannot be edited or removed.

        :param to_update: The labels to update.
        :param to_remove: A list of labels keys to remove.
        """
        # Update the user's labels:
        super(PyTorchModelHandler, self).update_labels(
            to_update=to_update, to_remove=to_remove
        )

        # Set the required labels:
        self._labels[self._LabelKeys.MODEL_CLASS_NAME] = self._model_class_name

    def save(
        self, output_path: str = None, *args, **kwargs
    ) -> Union[Dict[str, Artifact], None]:
        """
        Save the handled model at the given output path.

        :param output_path: The full path to the directory to save the handled model at. If not given, the context
                            stored will be used to save the model in the defaulted location.

        :return The saved model artifacts dictionary if context is available and None otherwise.

        :raise MLRunRuntimeError:         In case there is no model initialized in this handler.
        :raise MLRunInvalidArgumentError: If an output path was not given, yet a context was not provided in
                                          initialization.
        """
        super(PyTorchModelHandler, self).save(output_path=output_path)

        # Set the output path:
        if output_path is None:
            output_path = os.path.join(self._context.artifact_path, self._model_name)

        # Save the model:
        self._model_file = f"{self._model_name}.pt"
        torch.save(self._model.state_dict(), self._model_file)

        return None

    def load(self, checkpoint: str = None, *args, **kwargs):
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
        super(PyTorchModelHandler, self).load()

        # Validate the model's class is in the custom objects map:
        if self._model_class_name not in self._custom_objects:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The model class '{self._model_class_name}' was not found in the given custom objects map. The custom "
                f"objects map must include the model's class name in its values. Usually the model class should appear "
                f"last in the map dictionary as it is imported from the top to the bottom."
            )

        # Initialize the model:
        self._model = self._custom_objects[self._model_class_name](**kwargs)

        # Load the state dictionary into it:
        self._model.load_state_dict(torch.load(self._model_file))

    def to_onnx(
        self,
        model_name: str = None,
        input_sample: Union[torch.Tensor, Dict[str, torch.Tensor]] = None,
        input_layers_names: List[str] = None,
        output_layers_names: List[str] = None,
        optimize: bool = True,
        output_path: str = None,
        log: bool = None,
    ):
        """
        Convert the model in this handler to an ONNX model. The layer names are optional, they do not change the
        semantics of the model, it is only for readability.

        :param model_name:          The name to give to the converted ONNX model. If not given the default name will be
                                    the stored model name with the suffix '_onnx'.
        :param optimize:            Whether to optimize the ONNX model using 'onnxoptimizer' before saving the model.
                                    Defaulted to True.
        :param input_sample:        A torch.Tensor with the shape and data type of the expected input to the model. It
                                    is optional but recommended.
        :param input_layers_names:  List of names to assign to the input nodes of the graph in order. All of the other
                                    parameters (inner layers) can be set as well by passing additional names in the
                                    list. The order is by the order of the parameters in the model.
        :param output_layers_names: List of names to assign to the output nodes of the graph in order.
        :param output_path:         In order to save the ONNX model, pass here the output directory. The model file will
                                    be named with the model name given. Defaulted to None (not saving).
        :param log:                 In order to log the ONNX model, pass True. If None, the model will be logged if this
                                    handler has a MLRun context set. Defaulted to None.

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
            input_sample = torch.stack(
                [
                    torch.zeros(
                        size=input_feature.dims,
                        dtype=self.convert_value_type_to_torch_dtype(
                            value_type=input_feature.value_type
                        ),
                    )
                    for input_feature in self._inputs
                ]
            )

        # Set the output path:
        if output_path is not None:
            output_path = os.path.join(output_path, f"{model_name}.onnx")

        # Set the logging flag:
        log = self._context is not None if log is None else log

        # Convert to ONNX:
        torch.onnx.export(
            self._model,
            input_sample,
            output_path,
            input_names=input_layers_names,
            output_names=output_layers_names,
        )

        # Create a handler for the model:
        onnx_handler = ONNXModelHandler(
            model_name=model_name, model_path=output_path, context=self._context
        )
        onnx_handler.set_inputs(features=self._inputs)
        onnx_handler.set_outputs(features=self._outputs)
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

    @staticmethod
    def convert_value_type_to_torch_dtype(value_type: str) -> torch.dtype:
        """
        Get the 'torch.dtype' equivalent to the given MLRun data type.

        :param value_type: The MLRun value type to convert to torch data type.

        :return: The 'torch.dtype' equivalent to the given MLRun data type.

        :raise MLRunInvalidArgumentError: If torch is not supporting the given value type.
        """
        # Initialize the mlrun to torch data type conversion map:
        conversion_map = {
            ValueType.BOOL: torch.bool,
            ValueType.INT8: torch.int8,
            ValueType.INT16: torch.int16,
            ValueType.INT32: torch.int32,
            ValueType.INT64: torch.int64,
            ValueType.UINT8: torch.uint8,
            ValueType.BFLOAT16: torch.bfloat16,
            ValueType.FLOAT16: torch.float16,
            ValueType.FLOAT: torch.float32,
            ValueType.DOUBLE: torch.float64,
        }

        # Convert and return:
        if value_type in conversion_map:
            return conversion_map[value_type]
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"The ValueType given is not supported in torch: '{value_type}'."
        )

    @staticmethod
    def convert_torch_dtype_to_value_type(torch_dtype: Union[torch.dtype, str]) -> str:
        """
        Convert the given torch data type to MLRun value type. All of the CUDA supported data types are supported. For
        more information regarding torch data types, visit: https://pytorch.org/docs/stable/tensors.html#data-types

        :param torch_dtype: The torch data type to convert to MLRun's value type. Expected to be a 'torch.dtype' or
                            'str'.

        :return: The MLRun value type converted from the given data type.

        :raise MLRunInvalidArgumentError: If the torch data type is not supported by MLRun.
        """
        # Initialize the torch to mlrun data type conversion map:
        conversion_map = {
            str(torch.bool): ValueType.BOOL,
            str(torch.int8): ValueType.INT8,
            str(torch.short): ValueType.INT16,
            str(torch.int16): ValueType.INT16,
            str(torch.int): ValueType.INT32,
            str(torch.int32): ValueType.INT32,
            str(torch.long): ValueType.INT64,
            str(torch.int64): ValueType.INT64,
            str(torch.uint8): ValueType.UINT8,
            str(torch.bfloat16): ValueType.BFLOAT16,
            str(torch.half): ValueType.FLOAT16,
            str(torch.float16): ValueType.FLOAT16,
            str(torch.float): ValueType.FLOAT,
            str(torch.float32): ValueType.FLOAT,
            str(torch.double): ValueType.DOUBLE,
            str(torch.float64): ValueType.DOUBLE,
        }

        # Parse the given torch data type to string:
        if isinstance(torch_dtype, torch.dtype):
            torch_dtype = str(torch_dtype)

        # Convert and return:
        if torch_dtype in conversion_map:
            return conversion_map[torch_dtype]
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"MLRun value type is not supporting the given torch data type: '{torch_dtype}'."
        )

    def _collect_files_from_store_object(self):
        """
        If the model path given is of a store object, collect the needed model files into this handler for later loading
        the model.
        """
        # Get the artifact and model file along with its extra data:
        (
            self._model_file,
            self._model_artifact,
            self._extra_data,
        ) = mlrun.artifacts.get_model(self._model_path)

        # Read the model's class name:
        self._model_class_name = self._model_artifact.labels[
            self._LabelKeys.MODEL_CLASS_NAME
        ]

        # Continue collecting from abstract class:
        super(PyTorchModelHandler, self)._collect_files_from_store_object()

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
        self._model_class_name = (
            self._model_class_name
            if isinstance(self._model_class_name, str)
            else self._model_class_name.__name__
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
            return super(PyTorchModelHandler, self)._read_sample(sample=sample)
        elif isinstance(sample, torch.Tensor):
            return Feature(
                value_type=self.convert_torch_dtype_to_value_type(
                    torch_dtype=sample.dtype
                ),
                dims=list(sample.shape),
            )

        # Unsupported type:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"The sample type given '{type(sample)}' is not supported. The input / output ports are readable from "
            f"samples of the following types: torch.Tensor, np.ndarray."
        )
