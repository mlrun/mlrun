import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypeVar, Union

import mlrun
from mlrun.artifacts import Artifact

# Define a generic model type for the handler to have:
Model = TypeVar("Model")


class ModelHandler(ABC):
    """
    An abstract interface for handling a model of the supported frameworks.
    """

    def __init__(
        self,
        model: Model = None,
        model_name: str = "model",
        context: mlrun.MLClientCtx = None,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading.

        :param model:      Model to handle or None in case a loading parameters were supplied.
        :param model_name: The model name for saving and logging the model. Defaulted to 'model'.
        :param context:    MLRun context to work with for automatic loading and saving to the project directory.
        """
        self._model = model
        self._model_name = model_name
        self._context = context

    @property
    def model(self) -> Model:
        """
        Get the handled model. Will return None in case the model is not initialized.

        :return: The handled model.
        """
        return self._model

    def set_context(self, context: mlrun.MLClientCtx):
        """
        Set this handler MLRun context.

        :param context: The context to set to.
        """
        self._context = context

    @abstractmethod
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
        if self._model is None:
            raise RuntimeError(
                "Model cannot be save as it was not given in initialization or loaded during this run."
            )
        if output_path is None and self._context is None:
            raise ValueError(
                "An output path was not given and a context was not provided during the initialization of "
                "this model handler. To save the model, one of the two parameters must be supplied."
            )
        return None

    @abstractmethod
    def load(self, uid: str = None, epoch: int = None, *args, **kwargs):
        """
        Load the specified model in this handler. If a context was provided during initialization, the defaulted version
        of the model in the project will be loaded. To specify the model's version, its uid can be supplied along side
        an epoch for loading a callback of this run.

        :param uid:   To load a specific version of the model by the run uid that generated the model.
        :param epoch: To load a checkpoint of a given training (training's uid), add the checkpoint's epoch number.

        :raise ValueError: If a context was not provided during the handler initialization yet a uid was provided or if
                           an epoch was provided but a uid was not.
        """
        # Validate input:
        # # Epoch [V], UID [X]:
        if epoch is not None and uid is None:
            raise ValueError(
                "To load a model from a checkpoint of an epoch, the training run uid must be given."
            )
        # # Epoch [?], UID [V], Context [X]:
        if uid is not None and self._context is None:
            raise ValueError(
                "To load a specific version (by uid) of a model a context must be provided during the "
                "handler initialization."
            )

        # If a model instance is already loaded, delete it from memory:
        if self._model:
            del self._model

    @abstractmethod
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
        :param artifacts:  Artifacts to log the model with. Will be added to the extra data.

        :raise RuntimeError: In case there is no model in this handler.
        :raise ValueError:   In case a context is missing.
        """
        if self._model is None:
            raise RuntimeError(
                "Model cannot be logged as it was not given in initialization or loaded during this run."
            )
        if self._context is None:
            raise ValueError(
                "Cannot log model if a context was not provided during initialization."
            )

    def _get_model_directory(
        self, uid: Union[str, None], epoch: Union[int, None]
    ) -> str:
        """
        Get the model directory from the database specified in the context. By default with None in both 'uid' and
        'epoch', the latest model directory will be returned. If 'uid' is given then the directory that was produced
        with the function related to the uid will be returned. If an epoch number is given in addition to the uid, a
        checkpoint of the run in the given epoch will be returned.

        :param uid:   Function uid to look for.
        :param epoch: An epoch that produced a checkpoint to look for.

        :return: The model's directory path.
        """
        # TODO: Implement using tags in db
        pass

    @staticmethod
    def _get_model_name_from_file(path: str):
        """
        Get the model's name from its file (without the file's type).

        :param path: The path to the model's file.

        :return: The model file's name.
        """
        return os.path.basename(path).split(".")[0]

    @staticmethod
    def _import_module(classes_names: List[str], py_file_path: str) -> Dict[str, Any]:
        """
        Import the given class by its name from the given python file as: from 'py_file_path' import 'class_name'. If
        the class specified is already imported, a reference would simply be returned.

        :param classes_names: The classes names to be imported from the given python file.
        :param py_file_path:  Path to the python file with the classes code.

        :return: The imported classes dictionary where the keys are the classes names and the values are their imported
                 classes.
        """
        # Initialize the imports dictionary:
        classes_imports = {}

        # Go through the given classes:
        for class_name in classes_names:
            if class_name in sys.modules:
                # It is already imported:
                classes_imports[class_name] = sys.modules[class_name]
            else:
                # Import the class:
                spec = importlib.util.spec_from_file_location(
                    name=class_name, location=py_file_path
                )
                module = importlib.util.module_from_spec(spec=spec)
                spec.loader.exec_module(module)
                # Get the imported class and store it:
                classes_imports[class_name] = getattr(module, class_name)

        return classes_imports
