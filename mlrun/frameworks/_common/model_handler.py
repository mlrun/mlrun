import importlib.util
import json
import os
import shutil
import sys
import zipfile
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

    # Constant artifact names:
    _CUSTOM_OBJECTS_MAP_ARTIFACT_NAME = "{}_custom_objects_map.json"
    _CUSTOM_OBJECTS_DIRECTORY_ARTIFACT_NAME = "{}_custom_objects.zip"

    def __init__(
        self,
        model_name,
        model_path: str = None,
        model: Model = None,
        custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
        custom_objects_directory: str = None,
        context: mlrun.MLClientCtx = None,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading. Note you must provide at least
        one of 'model' and 'model_path'.

        :param model_name:               The model name for saving and logging the model.
        :param model_path:               Path to the directory with the model files. Can be passed as a model object
                                         path in the following format:
                                         'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'
        :param model:                    Model to handle or None in case a loading parameters were supplied.
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

        :raise ValueError: In case one of the given parameters are invalid.
        """
        # Validate input:
        self._validate_model_parameters(model=model, model_path=model_path)
        self._validate_custom_objects_parameters(
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
        )

        # Store parameters:
        self._model_name = model_name
        self._model_path = model_path
        self._model = model
        self._custom_objects_map = custom_objects_map
        self._custom_objects_directory = custom_objects_directory
        self._context = context

        # The imported custom objects from the map. None until the '_import_custom_objects' method is called.
        self._custom_objects = None  # type: Dict[str, Any]

        # Local path to the model file:
        self._model_file = None  # type: str

        # If the model path is of a model object, this will be the ModelArtifact object.
        self._model_artifact = None  # type: ModelArtifact

        # If the model path is of a store model object, this will be the extra data as DataItems ready to be downloaded.
        self._extra_data = None  # type: Union[Dict[str, Artifact], Dict[str, str]]

        # Collect the relevant files of the model into the handler's parameters:
        if model_path is not None:
            if mlrun.datastore.is_store_uri(self._model_path):
                self._collect_files_from_store_object()
            else:
                self._collect_files_from_local_path()

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
        Load the specified model in this handler. The custom objects will be imported right before loading the model. To
        access the model, call the 'model' property.
        """
        # If a model instance is already loaded, delete it from memory:
        if self._model:
            del self._model

        # Import the custom objects if needed (will be only imported once):
        if self._custom_objects is None:
            self._import_custom_objects()

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

    @abstractmethod
    def _collect_files_from_store_object(self):
        """
        If the model path given is of a store object, collect the needed model files into this handler for later loading
        the model.
        """
        pass

    @abstractmethod
    def _collect_files_from_local_path(self):
        """
        If the model path given is of a local path, search for the needed model files and collect them into this handler
        for later loading the model.
        """
        pass

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
            with open(self._custom_objects_map, "r") as map_json_file:
                self._custom_objects_map = json.loads(map_json_file.read())

        # Unzip the custom objects files if the directory was given as a zip:
        if not os.path.isdir(self._custom_objects_directory):
            with zipfile.ZipFile(self._custom_objects_directory, "r") as zip_file:
                # Update the root directory of all the custom obejcts py files:
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
                **self._import_module(
                    classes_names=(
                        custom_objects_names
                        if isinstance(custom_objects_names, list)
                        else [custom_objects_names]
                    ),
                    py_file_path=os.path.abspath(
                        os.path.join(self._custom_objects_directory, py_file)
                    ),
                ),
            }

    def _log_custom_objects(self) -> Dict[str, Artifact]:
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
        custom_objects_map_json = self._CUSTOM_OBJECTS_MAP_ARTIFACT_NAME.format(
            self._model_name
        )
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
            db_key=False,
        )

        # Zip the custom objects directory:
        custom_objects_zip = self._CUSTOM_OBJECTS_DIRECTORY_ARTIFACT_NAME.format(
            self._model_name
        )
        if self._custom_objects_directory.endswith(".zip"):
            # The custom objects are still zipped (model was not loaded but given as a live object):
            shutil.copy(self._custom_objects_directory, custom_objects_zip)
        else:
            # Zip the custom objects contents:
            shutil.make_archive(
                base_name=custom_objects_zip.rsplit(".", 1)[0],
                format="zip",
                root_dir=os.path.abspath(self._custom_objects_directory),
                # base_dir=os.path.basename(self._custom_objects_directory),
            )

        # Log the zip file artifact:
        artifacts[custom_objects_zip] = self._context.log_artifact(
            custom_objects_zip,
            local_path=custom_objects_zip,
            artifact_path=self._context.artifact_path,
            db_key=False,
        )

        return artifacts

    @staticmethod
    def _validate_model_parameters(model_path: str, model: Model):
        """
        Validate the given model parameters.

        :param model_path: Path to the directory with the model files. Can be passed as a model object path in the
                           following format: 'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'
        :param model:      Model to handle or None in case a loading parameters were supplied.

        :raise ValueError: If both parameters were None or both parameters were provided.
        """
        if (model_path is None and model is None) or (
            model_path is not None and model is not None
        ):
            raise ValueError(
                "Only one of 'model' or 'model_path' must be provided to the model handler."
            )

    @staticmethod
    def _validate_custom_objects_parameters(
        custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
        custom_objects_directory: str = None,
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

        :raise ValueError: If one of the parameters is not None but the other is or if the paths were of incorrect file
                           formats.
        """
        # Validate that if one is provided (not None), both are provided:
        if (custom_objects_map is not None and custom_objects_directory is None) or (
            custom_objects_map is None and custom_objects_directory is not None
        ):
            raise ValueError(
                "Either 'custom_objects_map' or 'custom_objects_directory' are None. Custom objects must be supplied "
                "with the custom object map dictionary (or json) and the directory with all the python files."
            )

        # Validate that if the map is a path, it is a path to a json file:
        if custom_objects_map is not None:
            if isinstance(custom_objects_map, str):
                if not (
                    custom_objects_map.endswith(".json")
                    and os.path.exists(custom_objects_map)
                ):
                    raise ValueError(
                        "The 'custom_objects_map' is either not found or not a dictionary or a path to a json file. "
                        "received: '{}'".format(custom_objects_map)
                    )

        # Validate that the path is of a directory or a zip file:
        if custom_objects_directory is not None:
            if not (
                os.path.isdir(custom_objects_directory)
                or custom_objects_directory.endswith(".zip")
            ):
                raise ValueError(
                    "The 'custom_objects_directory' is either not found or not a directory / zip file, "
                    "received: '{}'".format(custom_objects_directory)
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
