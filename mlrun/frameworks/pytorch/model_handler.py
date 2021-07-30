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
        model_class: Union[Type[Module], str],
        custom_objects_map: Union[Dict[Union[str, List[str]], str], str],
        custom_objects_directory: str,
        model_path: str = None,
        model: Module = None,
        context: mlrun.MLClientCtx = None,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading.
        :param model_name:               The model name for saving and logging the model.
        :param model_class:              The model's class type object. Can be passed as the class's name (string) as
                                         well. The model class must appear in the custom objects map dictionary / json.
        :param custom_objects_map:       A dictionary of all the custom objects required for loading the model. The keys
                                         are the classes / functions names and the values are their paths to the python
                                         files for the handler to import them from. If multiple objects needed to be
                                         imported from the same py file a list can be given. Notice, if the model was
                                         saved with the 'save_traces' flag on (True) the custom objects are not needed
                                         for loading the model. The map can be passed as a path to a json file as well.
                                         For example:
                                         {
                                             "optimizer": "/custom_objects_directory/.../custom_optimizer.py",
                                             ["layer1", "layer2"]: "/custom_objects_directory/.../custom_layers.py"
                                         }
        :param custom_objects_directory: Path to the directory with all the python files required for the custom
                                         objects.
        :param model_path:               Path to the model's directory with the saved '.pt' file. The file must start
                                         with the given model name. The model path can be also passed as a model object
                                         path in the following format:
                                         'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'.
        :param model:                    Model to handle or None in case a loading parameters were supplied.
        :param context:                  MLRun context to work with for logging the model.
        """
        # Setup the handler with name, path, model and context:
        super(PyTorchModelHandler, self).__init__(
            model_name=model_name, model_path=model_path, model=model, context=context
        )

        # Setup the initial model properties:
        self._custom_objects_map = custom_objects_map
        self._custom_objects_directory = custom_objects_directory
        self._custom_objects = {}  # type: Dict[str, Type]
        self._weights_file = None  # type: str
        self._class_name = None  # type: str
        self._class_py_file = None  # type: str

        # If the model's class name was given, import it so the class will be ready to use for loading:
        if isinstance(model_class, str):
            raise NotImplementedError
            # TODO: Need to import the custom objects.
            # # Validate input:
            # if model_class not in self._custom_objects_map:
            #     raise KeyError(
            #         "Model class was given by name, yet its py file is not available in the custom objects "
            #         "dictionary. The custom objects must have the model's class name as key with the py "
            #         "file path as his value."
            #     )
            # # Pop the model's import information:
            # self._class_name = model_class
            # self._class_py_file = self._custom_objects_sources.pop(model_class)
            # # Import the custom objects:
            # self._import_custom_objects()
            # # Import the model:
            # self._class = self._import_module(
            #     classes_names=[self._class_name], py_file_path=self._class_py_file
            # )[self._class_name]
        else:
            # Model is imported, store its class:
            self._class = model_class
            self._class_name = model_class.__name__

    # TODO: Save the custom objects dictionary as well.
    def save(
        self, output_path: str = None, *args, **kwargs
    ) -> Union[Dict[str, Artifact], None]:
        """
        Save the handled model at the given output path.

        :param output_path:  The full path to the directory to save the handled model at. If not given, the context
                             stored will be used to save the model in the defaulted location.

        :return The saved model artifacts dictionary if context is available and None otherwise.

        :raise RuntimeError: In case there is no model initialized in this handler.
        :raise ValueError:   If an output path was not given, yet a context was not provided in initialization.
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
                "weights_file": self._context.log_artifact(
                    weights_file,
                    local_path=weights_file,
                    artifact_path=output_path,
                    db_key=False,
                )
            }

        return artifacts

    def load(self, *args, **kwargs):
        """
        Load the specified model in this handler.
        """
        super(PyTorchModelHandler, self).load()

        # Initialize the model:
        self._model = self._class(*args, **kwargs)

        # Load the state dictionary into it:
        self._model.load_state_dict(torch.load(self._weights_file))

    def log(
        self,
        labels: Dict[str, Union[str, int, float]],
        parameters: Dict[str, Union[str, int, float]],
        extra_data: Dict[str, Any],
        artifacts: Dict[str, Artifact],
    ):
        """
        Log the model held by this handler into the MLRun context provided.

        :param labels:     Labels to log the model with.
        :param parameters: Parameters to log with the model.
        :param extra_data: Extra data to log with the model.
        :param artifacts:  Artifacts to log the model with.

        :raise RuntimeError: In case there is no model in this handler.
        :raise ValueError:   In case a context is missing.
        """
        super(PyTorchModelHandler, self).log(
            labels=labels,
            parameters=parameters,
            extra_data=extra_data,
            artifacts=artifacts,
        )

        # Save the model:
        model_artifacts = self.save()

        # Log the model:
        self._context.log_model(
            self._model_name,
            model_file=self._weights_file,
            framework="pytorch",
            labels=labels,
            parameters=parameters,
            metrics=self._context.results,
            extra_data={**model_artifacts, **artifacts, **extra_data},
        )

    def _get_model_from_path(self):
        """
        Use the 'get_model' method to get the logged model file, artifact and extra data if any are available and read
        them into this handler.
        """
        # Path of a store uri:
        if mlrun.datastore.is_store_uri(self._model_path):
            # Get the artifact and model file along with its extra data:
            (
                self._weights_file,
                self._model_artifact,
                self._extra_data,
            ) = mlrun.artifacts.get_model(self._model_path)
            # TODO: Read the custom objects logged as well.

    def _import_custom_objects(self):
        """
        Import the custom objects from the map and directory provided.
        """
        if self._custom_objects_map is None:
            return
        self._custom_objects = {}
        for custom_objects_names, py_file in self._custom_objects_map.items():
            self._custom_objects = {
                **self._custom_objects,
                **self._import_module(
                    classes_names=(
                        custom_objects_names
                        if isinstance(custom_objects_names, list)
                        else [custom_objects_names]
                    ),
                    py_file_path=py_file,
                ),
            }
