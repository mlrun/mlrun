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
        model_class: Union[Type[Module], str],
        custom_objects: Dict[Union[str, List[str]], str],
        model: Module = None,
        model_name: str = None,
        pt_file_path: str = None,
        context: mlrun.MLClientCtx = None,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading.

        :param model_class:    The model's class type object. Can be passed as the class's name (string) as well.
        :param custom_objects: Custom objects the model is using. Expecting a dictionary with the classes names to
                               import as keys (if multiple classes needed to be imported from the same py file a
                               list can be given) and the python file from where to import them as their values. The
                               model class itself must be specified in order to properly save it for later being loaded
                               with a handler. For example:
                               {
                                   "class_name": "/path/to/model.py",
                                   ["layer1", "layer2"]: "/path/to/custom_layers.py"
                               }
        :param model:          Model to handle or None in case a loading parameters were supplied.
        :param model_name:     The model name for saving and logging the model. Defaulted to the given model class.
        :param pt_file_path:   The model's saved '.pt' file with its tensors and attributes to load.
        :param context:        Context to save, load and log the model.
        """
        # Setup the handler with name, context and model:
        if model_name is None:
            model_name = (
                model_class if isinstance(model_class, str) else model_class.__name__
            )
        super(PyTorchModelHandler, self).__init__(
            model=model, model_name=model_name, context=context
        )

        # Setup the initial model properties:
        self._custom_objects_sources = (
            custom_objects if custom_objects is not None else {}
        )
        self._imported_custom_objects = {}  # type: Dict[str, Type]
        self._pt_file_path = pt_file_path
        self._class_name = None  # type: str
        self._class_py_file = None  # type: str

        # If the model's class name was given, import it so the class will be ready to use for loading:
        if isinstance(model_class, str):
            # Validate input:
            if model_class not in self._custom_objects_sources:
                raise KeyError(
                    "Model class was given by name, yet its py file is not available in the custom objects "
                    "dictionary. The custom objects must have the model's class name as key with the py "
                    "file path as his value."
                )
            # Pop the model's import information:
            self._class_name = model_class
            self._class_py_file = self._custom_objects_sources.pop(model_class)
            # Import the custom objects:
            self._import_custom_objects()
            # Import the model:
            self._class = self._import_module(
                classes_names=[self._class_name], py_file_path=self._class_py_file
            )[self._class_name]
        else:
            # Model is imported, store its class:
            self._class = model_class
            self._class_name = model_class.__name__

    def save(
        self, output_path: str = None, update_paths: bool = True, *args, **kwargs
    ) -> Union[Dict[str, Artifact], None]:
        """
        Save the handled model at the given output path.

        :param output_path:  The full path to the directory to save the handled model at. If not given, the context
                             stored will be used to save the model in the defaulted location.
        :param update_paths: Whether or not to update the model and weights paths to the newly saved model. Defaulted to
                             True.

        :return The saved model artifacts dictionary if context is available and None otherwise.

        :raise RuntimeError: In case there is no model initialized in this handler.
        :raise ValueError:   If an output path was not given, yet a context was not provided in initialization.
        """
        super(PyTorchModelHandler, self).save(output_path=output_path)

        # Set the output path:
        if output_path is None:
            output_path = os.path.join(self._context.artifact_path, self._model_name)

        # Save the model:
        pt_file_path = "{}.pt".format(self._model_name)
        torch.save(self._model.state_dict(), pt_file_path)
        if update_paths:
            self._pt_file_path = pt_file_path

        # Update the paths and log artifact if context is available:
        artifacts = None
        if self._context:
            artifacts = {
                "model_file": self._context.log_artifact(
                    pt_file_path,
                    local_path=pt_file_path,
                    artifact_path=output_path,
                    db_key=False,
                )
            }

        return artifacts

    def load(self, uid: str = None, epoch: int = None, *args, **kwargs):
        """
        Load the specified model in this handler. Additional parameters can be passed to the model class constructor via
        the args and kwargs parameters. If a context was provided during initialization, the defaulted version
        of the model in the project will be loaded. To specify the model's version, its uid can be supplied along side
        an epoch for loading a callback of this run.

        :param uid:   To load a specific version of the model by the run uid that generated the model.
        :param epoch: To load a checkpoint of a given training, add the checkpoint's epoch number.

        :raise ValueError: If a context was not provided during the handler initialization yet a uid was provided or if
                           an epoch was provided but a uid was not.
        """
        super(PyTorchModelHandler, self).load(uid=uid, epoch=epoch)

        # Initialize the model:
        self._model = self._class(*args, **kwargs)

        # Load the state dictionary into it:
        self._model.load_state_dict(torch.load(self._pt_file_path))

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
        model_artifacts = self.save(update_paths=True)

        # Log the model:
        self._context.log_model(
            self._model_name,
            model_file=self._pt_file_path,
            framework="pytorch",
            labels=labels,
            parameters=parameters,
            metrics=self._context.results,
            extra_data={**model_artifacts, **artifacts, **extra_data},
        )

    def _import_custom_objects(self):
        """
        Import the custom objects from the 'self._custom_objects_sources' dictionary into the
        'self._imported_custom_objects'.
        """
        if self._custom_objects_sources:
            for classes_names, py_file in self._custom_objects_sources.items():
                self._imported_custom_objects = self._import_module(
                    classes_names=(
                        classes_names
                        if isinstance(classes_names, list)
                        else [classes_names]
                    ),
                    py_file_path=py_file,
                )
