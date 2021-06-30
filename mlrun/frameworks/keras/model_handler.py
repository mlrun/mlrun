import os
from typing import Any, Dict, List, Type, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

import mlrun
from mlrun.artifacts import Artifact
from mlrun.frameworks._common import ModelHandler


class KerasModelHandler(ModelHandler):
    """
    Class for handling a tensorflow.keras model, enabling loading and saving it during runs.
    """

    class SaveFormats:
        """
        Save formats to pass to the 'KerasModelHandler'.
        """

        SAVED_MODEL = "SavedModel"
        H5 = "H5"
        JSON_ARCHITECTURE_H5_WEIGHTS = "Json_H5"
        TF_CHECKPOINT = "TFCheckpoint"

        @staticmethod
        def _get_formats() -> List[str]:
            """
            Get a list with all the supported saving formats.

            :return: Saving formats list.
            """
            return [
                value
                for key, value in KerasModelHandler.SaveFormats.__dict__.items()
                if not key.startswith("_") and isinstance(value, str)
            ]

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        model: Model = None,
        model_name: str = None,
        model_path: str = None,
        weights_path: str = None,
        custom_objects: Dict[
            str,
            Union[
                str, Type[Model], Type[Layer], Type[Loss], Type[Optimizer], Type[Metric]
            ],
        ] = None,
        save_format: str = SaveFormats.H5,
        save_traces: bool = False,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading.

        :param context:        MLRun context to work with.
        :param model:          Model to handle or None in case a loading parameters were supplied.
        :param model_name:     The model name for saving and logging the model. Defaulted to the given model class.
        :param model_path:     Path to the model directory (SavedModel format) or the model architecture (Json and H5
                               format).
        :param weights_path:   Path to the weights 'h5' file if the model was saved
        :param custom_objects: A dictionary of all the custom objects required for loading the model. The keys are
                               the class name of the custom object and the value can be the class or a path to a python
                               file for the handler to import the class from. Notice, if the model was saved with the
                               'save_traces' flag on (True) the custom objects are not needed for loading the model, but
                               each of the custom object must implement the methods 'get_config' and 'from_config'.
        :param save_format:    The save format to use. Should be passed as a member of the class 'SaveFormats'.
        :param save_traces:    Whether or not to use functions saving (only available for the save format
                               'SaveFormats.SAVED_MODEL') for loading the model later without the custom objects
                               dictionary. Only from tensorflow version >= 2.4.0.

        :raise ValueError: In case the input was incorrect:
                           * Save format is unrecognized.
                           * There was no model or model files supplied.
                           * 'save_traces' parameter was miss-used.
        """
        # Setup the handler with name, context and model:
        if model_name is None:
            if model is not None:
                model_name = model.name
            else:
                model_name = self._get_model_name_from_file(path=model_path)
        super(KerasModelHandler, self).__init__(
            model=model, model_name=model_name, context=context
        )

        # Validate 'save_format':
        if save_format not in KerasModelHandler.SaveFormats._get_formats():
            raise ValueError("Unrecognized save format: '{}'".format(save_format))

        # Validate model was given in some way:
        if model is None:
            if (
                self._save_format
                == KerasModelHandler.SaveFormats.JSON_ARCHITECTURE_H5_WEIGHTS
            ):
                if model_path is None or weights_path is None:
                    raise ValueError(
                        "For 'SaveFormats.JSON_ARCHITECTURE_H5_WEIGHTS' both model and weights file must "
                        "be given."
                    )
            elif self._save_format == KerasModelHandler.SaveFormats.TF_CHECKPOINT:
                raise NotImplementedError
            else:
                # self._save_format = SaveFormats.SAVED_MODEL
                raise NotImplementedError

        # Validate 'save_traces':
        if save_traces:
            if float(tf.__version__.rsplit(".", 1)[0]) < 2.4:
                raise ValueError(
                    "The 'save_traces' parameter can be true only for tensorflow versions >= 2.4. Current "
                    "version is {}".format(tf.__version__)
                )
            if save_format != KerasModelHandler.SaveFormats.SAVED_MODEL:
                raise ValueError(
                    "The 'save_traces' parameter is valid only for the 'SavedModel' format."
                )

        # Store the configuration:
        self._model_path = model_path
        self._weights_path = weights_path
        self._custom_objects = custom_objects
        self._save_format = save_format
        self._save_traces = save_traces

        # Import the custom objects:
        self._import_custom_objects()

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
        super(KerasModelHandler, self).save(output_path=output_path)

        # Setup the returning model artifacts list:
        artifacts = {}  # type: Dict[str, Artifact]
        model_path = None  # type: str
        weights_path = None  # type: str

        # Set the output path:
        if output_path is None:
            output_path = os.path.join(self._context.artifact_path, self._model_name)

        if (
            self._save_format
            == KerasModelHandler.SaveFormats.JSON_ARCHITECTURE_H5_WEIGHTS
        ):
            # Save the model architecture (json):
            model_architecture = self._model.to_json()
            model_path = "{}.json".format(self._model_name)
            with open(model_path, "w") as json_file:
                json_file.write(model_architecture)
            # Save the model weights (h5):
            weights_path = "{}.h5".format(self._model_name)
            self._model.save_weights(weights_path)
        elif self._save_format == KerasModelHandler.SaveFormats.H5:
            # Save the model as a h5 file:
            model_path = "{}.h5".format(self._model_name)
            self._model.save(model_path)
        elif self._save_format == KerasModelHandler.SaveFormats.TF_CHECKPOINT:
            raise NotImplementedError
        else:
            # self._save_format = SaveFormats.SAVED_MODEL
            raise NotImplementedError

        # Update the paths and log artifacts if context is available:
        if model_path:
            self._model_path = model_path
            if self._context:
                artifacts["model_file"] = self._context.log_artifact(
                    model_path,
                    local_path=model_path,
                    artifact_path=output_path,
                    db_key=False,
                )
        if weights_path:
            self._weights_path = weights_path
            if self._context:
                artifacts["weights_file"] = self._context.log_artifact(
                    weights_path,
                    local_path=weights_path,
                    artifact_path=output_path,
                    db_key=False,
                )

        return artifacts if self._context else None

    def load(self, uid: str = None, epoch: int = None, *args, **kwargs):
        """
        Load the specified model in this handler. Additional parameters for the class initializer can be passed via the
        args list and kwargs dictionary.
        """
        super(KerasModelHandler, self).load(uid=uid, epoch=epoch)

        if (
            self._save_format
            == KerasModelHandler.SaveFormats.JSON_ARCHITECTURE_H5_WEIGHTS
        ):
            # Load the model architecture (json):
            with open(self._model_path, "r") as json_file:
                model_architecture = json_file.read()
            self._model = keras.models.model_from_json(model_architecture)
            # Load the model weights (h5):
            self._model.load_weights(self._weights_path)
        elif self._save_format == KerasModelHandler.SaveFormats.TF_CHECKPOINT:
            raise NotImplementedError
        else:
            # self._save_format = SaveFormats.SAVED_MODEL
            raise NotImplementedError

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
        model_artifacts = self.save(update_paths=True)

        # Log the model:
        self._context.log_model(
            self._model.name,
            model_file=self._model_path,
            framework="tensorflow.keras",
            labels={"save-format": self._save_format, **labels},
            parameters=parameters,
            metrics=self._context.results,
            extra_data={**model_artifacts, **artifacts, **extra_data},
        )

    def _import_custom_objects(self):
        """
        Import the custom objects from the 'self._custom_objects_sources' dictionary into the
        'self._imported_custom_objects'.
        """
        if self._custom_objects:
            for object_name in self._custom_objects:
                if isinstance(self._custom_objects[object_name], str):
                    self._custom_objects[object_name] = self._import_module(
                        classes_names=[object_name],
                        py_file_path=self._custom_objects[object_name],
                    )[object_name]
