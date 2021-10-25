import os
from typing import Any, Dict, List, Type, Union

import torch
from torch.nn import Module

import mlrun
from mlrun.artifacts import Artifact
from mlrun.frameworks._common import ModelHandler


class PyTorchModelHandler(ModelHandler):
    """
    Class for handling a PyTorch model, enabling loading and saving it during runs.
    """

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
        # Will hold the model's weights .pt file:
        self._weights_file = None  # type: str

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

    def set_inputs(
        self,
        from_sample: torch.Tensor = None,
        names: List[str] = None,
        data_types: List[torch.dtype] = None,
        shapes: List[List[int]] = None,
    ):
        """
        Set the inputs property of this model to be logged along with it. The method 'to_onnx' can use this property as
        well for the conversion process.

        :param from_sample: Read the inputs properties from a given input sample to the model.
        :param names:       List of names for each input layer.
        :param data_types:  List of data types for each input layer.
        :param shapes:      List of tensor shapes for each input layer.
        """
        # TODO: Implement for IOLogging
        pass

    def set_outputs(
        self,
        from_sample: torch.Tensor = None,
        names: List[str] = None,
        data_types: List[torch.dtype] = None,
        shapes: List[List[int]] = None,
    ):
        """
        Set the outputs property of this model to be logged along with it. The method 'to_onnx' can use this property as
        well for the conversion process.

        :param from_sample: Read the inputs properties from a given input sample to the model.
        :param names:       List of names for each output layer.
        :param data_types:  List of data types for each output layer.
        :param shapes:      List of tensor shapes for each output layer.
        """
        # TODO: Implement for IOLogging
        pass

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
        weights_file = "{}.pt".format(self._model_name)
        torch.save(self._model.state_dict(), weights_file)
        self._weights_file = weights_file

        # Update the paths and log artifact if context is available:
        artifacts = None
        if self._context is not None:
            artifacts = {
                self._get_weights_file_artifact_name(): self._context.log_artifact(
                    weights_file,
                    local_path=weights_file,
                    artifact_path=output_path,
                    db_key=False,
                )
            }

        return artifacts

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
                "The model class '{}' was not found in the given custom objects map. The custom objects map must "
                "include the model's class name in its values. Usually the model class should appear last in the "
                "map dictionary as it is imported from the top to the bottom.".format(
                    self._model_class_name
                )
            )

        # Initialize the model:
        self._model = self._custom_objects[self._model_class_name](*args, **kwargs)

        # Load the state dictionary into it:
        self._model.load_state_dict(torch.load(self._weights_file))

    def log(
        self,
        labels: Dict[str, Union[str, int, float]] = None,
        parameters: Dict[str, Union[str, int, float]] = None,
        extra_data: Dict[str, Any] = None,
        artifacts: Dict[str, Artifact] = None,
    ):
        """
        Log the model held by this handler into the MLRun context provided.

        :param labels:     Labels to log the model with.
        :param parameters: Parameters to log with the model.
        :param extra_data: Extra data to log with the model.
        :param artifacts:  Artifacts to log the model with. Will be added to the extra data.

        :raise MLRunInvalidArgumentError: In case a context is missing or there is no model in this handler.
        """
        super(PyTorchModelHandler, self).log(
            labels=labels,
            parameters=parameters,
            extra_data=extra_data,
            artifacts=artifacts,
        )

        # Set default values:
        labels = {} if labels is None else labels
        parameters = {} if parameters is None else parameters
        extra_data = {} if extra_data is None else extra_data
        artifacts = {} if artifacts is None else artifacts

        # Save the model:
        model_artifacts = self.save()

        # Log the custom objects:
        custom_objects_artifacts = (
            self._log_custom_objects() if self._custom_objects_map is not None else {}
        )

        # Log the model:
        self._context.log_model(
            self._model_name,
            db_key=self._model_name,
            model_file=self._weights_file,
            framework="pytorch",
            labels={"model-class-name": self._model_class_name, **labels},
            parameters=parameters,
            metrics=self._context.results,
            extra_data={
                **model_artifacts,
                **custom_objects_artifacts,
                **artifacts,
                **extra_data,
            },
        )

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
        :param input_sample:        A torch.Tensor with the shape and data type of the expected input to the model. It
                                    is optional but recommended.
        :param input_layers_names:  List of names to assign to the input nodes of the graph in order. All of the other
                                    parameters (inner layers) can be set as well by passing additional names in the
                                    list. The order is by the order of the parameters in the model.
        :param output_layers_names: List of names to assign to the input nodes of the graph in order.
        :param optimize:            Whether or not to optimize the ONNX model using 'onnxoptimizer' before saving the
                                    model. Defaulted to True.
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

        # Set the input signature:
        # TODO: Read the input signature parsing in case its None (from the PyTorchModelHandler - IOLogging).

        # Set the output path:
        if output_path is not None:
            output_path = os.path.join(output_path, "{}.onnx".format(model_name))

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

    def _collect_files_from_store_object(self):
        """
        If the model path given is of a store object, collect the needed model files into this handler for later loading
        the model.
        """
        # Get the artifact and model file along with its extra data:
        (
            self._weights_file,
            self._model_artifact,
            self._extra_data,
        ) = mlrun.artifacts.get_model(self._model_path)

        # Read the model's class name:
        self._model_class_name = self._model_artifact.labels["model-class-name"]

        # Read the custom objects:
        self._custom_objects_map = self._extra_data[
            self._get_custom_objects_map_artifact_name()
        ].local()
        self._custom_objects_directory = self._extra_data[
            self._get_custom_objects_directory_artifact_name()
        ].local()

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
        self._weights_file = os.path.join(
            self._model_path, "{}.pt".format(self._model_name)
        )
        if not os.path.exists(self._weights_file):
            raise mlrun.errors.MLRunNotFoundError(
                "The model weights file '{}.pt' was not found within the given 'model_path': "
                "'{}'".format(self._model_name, self._model_path)
            )
