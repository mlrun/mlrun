import importlib.util
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypeVar, Union

import mlrun
from mlrun.artifacts import Artifact, ModelArtifact

# Define a generic model type for the handler to have:
Model = TypeVar("Model")


class ModelHandler(ABC):
    """
    An abstract interface for handling a model of the supported frameworks.
    """

    def __init__(
        self,
        model_name,
        model_path: str = None,
        model: Model = None,
        context: mlrun.MLClientCtx = None,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading. Note you must provide at least
        one of 'model' and 'model_path'.

        :param model_name: The model name for saving and logging the model.
        :param model_path: Path to the directory with the model files. Can be passed as a model object path in the
                           following format: 'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'
        :param model:      Model to handle or None in case a loading parameters were supplied.
        :param context:    MLRun context to work with for logging the model.

        :raise ValueError: In case both model and model path were not given.
        """
        # Validate input:
        if model_path is None and model is None:
            raise ValueError(
                "At least one of 'model' or 'model_path' must be provided to the model handler."
            )

        # Store parameters:
        self._model_name = model_name
        self._model_path = model_path
        self._model = model
        self._context = context

        # Local path to the model file (should be initialized before 'load' is called):
        self._model_file = None  # type: str

        # If the model path is of a model object, this will be the ModelArtifact object. Otherwise it will remain None.
        # (should be initialized before 'load' is called):
        self._model_artifact = None  # type: ModelArtifact

        # If the model path is of a model object, this will be the extra data as DataItems ready to be downloaded.
        # Else, the model path is of a directory and the extra data will be the files in this directory. (should be
        # initialized before 'load' is called):
        self._extra_data = None  # type: Union[Dict[str, Artifact], Dict[str, str]]

    @property
    def model(self) -> Model:
        """
        Get the handled model. Will return None in case the model is not initialized.

        :return: The handled model.
        """
        return self._model

    @property
    def model_name(self) -> str:
        """
        Get the handled model's name.

        :return: The handled model's name.
        """
        return self._model_name

    def set_context(self, context: mlrun.MLClientCtx):
        """
        Set this handler MLRun context.

        :param context: The context to set to.
        """
        self._context = context

    @abstractmethod
    def save(
        self, output_path: str = None, *args, **kwargs
    ) -> Union[Dict[str, Artifact], None]:
        """
        Save the handled model at the given output path.

        :param output_path:  The full path to the directory to save the handled model at. If not given, the context
                             stored will be used to save the model in the defaulted artifacts location.
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
    def load(self, *args, **kwargs):
        """
        Load the specified model in this handler. To access the model, call the 'model' property.
        """
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
