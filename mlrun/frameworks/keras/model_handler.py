import os
import shutil
import zipfile
from typing import Any, Dict, List, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

import mlrun
from mlrun.artifacts import Artifact
from mlrun.frameworks._common import ModelHandler


class KerasModelHandler(ModelHandler):
    """
    Class for handling a tensorflow.keras model, enabling loading and saving it during runs.
    """

    class ModelFormats:
        """
        Model formats to pass to the 'KerasModelHandler' for loading and saving keras models.
        """

        SAVED_MODEL = "SavedModel"
        H5 = "H5"
        JSON_ARCHITECTURE_H5_WEIGHTS = "json_H5"

    def __init__(
        self,
        model_name: str,
        model_path: str = None,
        model: Model = None,
        context: mlrun.MLClientCtx = None,
        custom_objects_map: Union[Dict[Union[str, List[str]], str], str] = None,
        custom_objects_directory: str = None,
        model_format: str = ModelFormats.H5,
        save_traces: bool = False,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading. Notice that if the model path
        given is of a previously logged model (store model object path), all of the other configurations will be loaded
        automatically as they were logged with the model, hence they are optional.

        :param model_name:               The model name for saving and logging the model.
        :param model_path:               Path to the model's directory to load it from. The model files must start with
                                         the given model name and the directory must contain based on the given model
                                         formats:
                                         * SavedModel - A zip file 'model_name.zip'.
                                         * H5 - A h5 file 'model_name.h5'.
                                         * Architecture and weights - The json file 'model_name.json' and h5 weight file
                                           'model_name.h5'.
                                         The model path can be also passed as a model object path in the following
                                         format: 'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'.
        :param model:                    Model to handle or None in case a loading parameters were supplied.
        :param context:                  MLRun context to work with for logging the model.
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
        :param model_format:             The format to use for saving and loading the model. Should be passed as a
                                         member of the class 'ModelFormats'. Defaulted to
                                         'ModelFormats.JSON_ARCHITECTURE_H5'.
        :param save_traces:              Whether or not to use functions saving (only available for the 'SavedModel'
                                         format) for loading the model later without the custom objects dictionary. Only
                                         from tensorflow version >= 2.4.0. Using this setting will increase the model
                                         saving size.

        :raise ValueError: In case the input was incorrect:
                           * Model format is unrecognized.
                           * There was no model or model directory supplied.
                           * 'save_traces' parameter was miss-used.
        """
        # TODO: Implement local directory loading and from fb.
        if model_path is not None and not mlrun.datastore.is_store_uri(model_path):
            raise NotImplementedError(
                "Initializing a keras handler with a directory path is not yet implemented. "
                "Please use a store object of a loaded model instead."
            )
        # TODO: Implement the custom objects support.
        if custom_objects_map is not None and custom_objects_directory is not None:
            raise NotImplementedError("Custom objects are not yet supported.")

        # Setup the handler with name, path, model and context:
        super(KerasModelHandler, self).__init__(
            model_name=model_name, model_path=model_path, model=model, context=context
        )

        # Validate given format:
        if model_format not in [
            KerasModelHandler.ModelFormats.SAVED_MODEL,
            KerasModelHandler.ModelFormats.H5,
            KerasModelHandler.ModelFormats.JSON_ARCHITECTURE_H5_WEIGHTS,
        ]:
            raise ValueError(
                "Unrecognized model format: '{}'. Please use one of the class members of "
                "'KerasModelHandler.ModelFormats'".format(model_format)
            )

        # Validate custom objects input:
        if (custom_objects_map is not None and custom_objects_directory is None) or (
            custom_objects_map is None and custom_objects_directory is not None
        ):
            raise ValueError(
                "Custom objects must be supplied with the custom object to python file map and the "
                "directory with all the python files."
            )

        # Validate 'save_traces':
        if save_traces:
            if float(tf.__version__.rsplit(".", 1)[0]) < 2.4:
                raise ValueError(
                    "The 'save_traces' parameter can be true only for tensorflow versions >= 2.4. Current "
                    "version is {}".format(tf.__version__)
                )
            if model_format != KerasModelHandler.ModelFormats.SAVED_MODEL:
                raise ValueError(
                    "The 'save_traces' parameter is valid only for the 'SavedModel' format."
                )

        # Store the custom objects:
        self._custom_objects_map = custom_objects_map
        self._custom_objects_directory = custom_objects_directory
        self._custom_objects = None

        # Store the configuration:
        self._model_format = model_format
        self._save_traces = save_traces

        # Get the model files and configurations from the given model path, downloading to local if needed:
        if model_path is not None:
            self._get_model_from_path()

        # Import the custom objects:
        if self._custom_objects_map is not None:
            self._import_custom_objects()

    # TODO: output_path won't work well with logging artifacts. Need to look into changing the logic of 'log_artifact'.
    def save(
        self, output_path: str = None, *args, **kwargs
    ) -> Union[Dict[str, Artifact], None]:
        """
        Save the handled model at the given output path.

        :param output_path:  The full path to the directory to save the handled model at. If not given, the context
                             stored will be used to save the model in the defaulted artifacts location.

        :return The saved model artifacts dictionary if context is available and None otherwise.
        """
        super(KerasModelHandler, self).save(output_path=output_path)

        # Setup the returning model artifacts list:
        artifacts = {}  # type: Dict[str, Artifact]
        model_file = None  # type: str
        weights_file = None  # type: str

        # Set the output path:
        if output_path is None:
            output_path = os.path.join(self._context.artifact_path, self._model_name)

        # ModelFormats.H5 - Save as a h5 file:
        if self._model_format == KerasModelHandler.ModelFormats.H5:
            model_file = "{}.h5".format(self._model_name)
            self._model.save(model_file)

        # ModelFormats.SAVED_MODEL - Save as a SavedModel directory and zip its file:
        elif self._model_format == KerasModelHandler.ModelFormats.SAVED_MODEL:
            # Save it in a SavedModel format directory:
            if self._save_traces is True:
                # Save traces can only be used in versions >= 2.4, so only if its true we use it in the call:
                self._model.save(self._model_name, save_traces=self._save_traces)
            else:
                self._model.save(self._model_name)
            # Zip it:
            model_file = "{}.zip".format(self._model_name)
            shutil.make_archive(model_file, "zip", self._model_name)

        # ModelFormats.JSON_ARCHITECTURE_H5_WEIGHTS - Save as a json architecture and h5 weights files:
        else:
            # Save the model architecture (json):
            model_architecture = self._model.to_json()
            model_file = "{}.json".format(self._model_name)
            with open(model_file, "w") as json_file:
                json_file.write(model_architecture)
            # Save the model weights (h5):
            weights_file = "{}.h5".format(self._model_name)
            self._model.save_weights(weights_file)

        # Save the custom objects:
        # TODO: Save the custom objects dictionary as a json file named "custom_objects_map.json", zip it with the
        #       custom objects directory and log them both as a zip file named "custom_objects.zip".

        # Update the paths and log artifacts if context is available:
        if model_file:
            self._model_file = model_file
            if self._context is not None:
                artifacts[
                    "{}_model_file".format(self._model_name)
                ] = self._context.log_artifact(
                    model_file,
                    local_path=model_file,
                    artifact_path=output_path,
                    db_key=False,
                )
        if weights_file:
            if self._context is not None:
                artifacts[
                    "{}_weights_file".format(self._model_name)
                ] = self._context.log_artifact(
                    weights_file,
                    local_path=weights_file,
                    artifact_path=output_path,
                    db_key=False,
                )

        return artifacts if self._context is not None else None

    def load(self, weights_path: str = None, *args, **kwargs):
        """
        Load the specified model in this handler. If a checkpoint is required to be loaded, it can be given by its tag
        (epoch) it was logged with as an artifact or the local path to the file. Additional parameters for the class
        initializer can be passed via the args list and kwargs dictionary.

        :param weights_path: Tag to look for in the extra data for weights that were logged along side the model, or
                             path to the local file of the weights.
        """
        super(KerasModelHandler, self).load()

        # Get the model file and if available, its artifacts:
        model_file = None  # type: str

        # ModelFormats.H5 - Load from a .h5 file:
        if self._model_format == KerasModelHandler.ModelFormats.H5:
            model_file = self._model_file

        # ModelFormats.SAVED_MODEL - Load from a SavedModel directory:
        elif self._model_format == KerasModelHandler.ModelFormats.SAVED_MODEL:
            # Unzip the SavedModel directory:
            with zipfile.ZipFile(self._model_file, "r") as zip_file:
                zip_file.extractall(os.path.dirname(self._model_file))
            # Load the model from the unzipped directory:
            model_file = self._model_file.split(".")[0]

        # ModelFormats.JSON_ARCHITECTURE_H5_WEIGHTS - Load from a .json architecture file and a .h5 weights file:
        else:
            raise NotImplementedError  # TODO: Implement the weights file lookup in the extra data of the model.
            # # Get the model from the model path:
            # self._get_model(model_file_suffix=".json")
            # # Load the model architecture (json):
            # with open(self._model_file, "r") as json_file:
            #     model_architecture = json_file.read()
            # self._model = keras.models.model_from_json(model_architecture)
            # # Load the model weights (h5):
            # weights_path = None
            # self._model.load_weights(weights_path)

        self._model = keras.models.load_model(
            model_file, custom_objects=self._custom_objects
        )

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
        super(KerasModelHandler, self).log(
            labels=labels,
            parameters=parameters,
            extra_data=extra_data,
            artifacts=artifacts,
        )

        # Save the model:
        model_artifacts = self.save()

        # Log the model:
        self._context.log_model(
            self._model.name,
            model_file=self._model_file,
            framework="tensorflow.keras",
            labels={
                "model-format": self._model_format,
                "save-traces": self._save_traces,
                **labels,
            },
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
                self._model_file,
                self._model_artifact,
                self._extra_data,
            ) = mlrun.artifacts.get_model(self._model_path)
            # Read the settings:
            self._model_format = self._model_artifact.labels["model-format"]
            self._save_traces = self._model_artifact.labels["save-traces"]
            # TODO: Read the custom objects logged as well.

    def _import_custom_objects(self):
        """
        Import the custom objects from the map and directory provided.
        """
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
