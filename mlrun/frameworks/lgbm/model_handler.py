import os
import pickle
from typing import Dict, List, Tuple, Union

import cloudpickle
import lightgbm as lgb
import numpy as np
import pandas as pd

import mlrun

from .._ml_common import MLModelHandler


class LGBMModelHandler(MLModelHandler):
    """
    Class for handling a XGBoost model, enabling loading and saving it during runs.
    """

    # Framework name:
    FRAMEWORK_NAME = "lgbm"

    # Declare a type of an input sample:
    IOSample = Union[pd.DataFrame, np.ndarray, List[Tuple[str, str]]]

    class ModelFormats:
        """
        Model formats to pass to the 'LGBMModelHandler' for loading and saving LightGBM models.
        """

        PKL = "pkl"
        TXT = "txt"

    def __init__(
        self,
        model_name: str = None,
        model_path: str = None,
        model: lgb.LGBMModel = None,
        modules_map: Union[Dict[str, Union[None, str, List[str]]], str] = None,
        custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
        custom_objects_directory: str = None,
        context: mlrun.MLClientCtx = None,
        model_format: str = ModelFormats.PKL,
        **kwargs,
    ):
        """
        Initialize the handler. The model can be set here so it won't require loading. Note you must provide at least
        one of 'model' and 'model_path'. If a model is not given, the files in the model path will be collected
        automatically to be ready for loading.

        :param model:                    Model to handle or None in case a loading parameters were supplied.
        :param model_path:               Path to the directory with the model files. Can be passed as a model object
                                         path in the following format:
                                         'store://models/<PROJECT_NAME>/<MODEL_NAME>:<VERSION>'
        :param model_name:               The model name for saving and logging the model:
                                         * Mandatory for loading the model from a local path.
                                         * If given a logged model (store model path) it will be read from the artifact.
                                         * If given a loaded model object and the model name is None, the name will be
                                           set to the model's object name / class.
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
        :param model_format:             The format to use for saving and loading the model. Should be passed as a
                                         member of the class 'ModelFormats'. Defaulted to 'ModelFormats.PKL'.

        :raise MLRunInvalidArgumentError: In case one of the given parameters are invalid.
        """
        # Validate given format:
        if model_format not in [
            LGBMModelHandler.ModelFormats.PKL,
            LGBMModelHandler.ModelFormats.TXT,
        ]:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Unrecognized model format: '{model_format}'. Please use one of the class members of "
                "'TFKerasModelHandler.ModelFormats'"
            )
        if model_format == LGBMModelHandler.ModelFormats.TXT:
            raise NotImplementedError(
                "TXT model format is not yet implemented for LightGBM."
            )

        # Store the configuration:
        self._model_format = model_format

        super(LGBMModelHandler, self).__init__(
            model=model,
            model_path=model_path,
            model_name=model_name,
            modules_map=modules_map,
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
            context=context,
            **kwargs,
        )

    def _collect_files_from_local_path(self):
        """
        If the model path given is of a local path, search for the needed model files and collect them into this handler
        for later loading the model.

        :raise MLRunNotFoundError: If the model file was not found.
        """
        # ModelFormats.PKL - Get the pickle model file:
        if self._model_format == LGBMModelHandler.ModelFormats.PKL:
            self._model_file = os.path.join(self._model_path, f"{self._model_name}.pkl")
            if not os.path.exists(self._model_file):
                raise mlrun.errors.MLRunNotFoundError(
                    f"The model file '{self._model_name}.pkl' was not found within the given 'model_path': "
                    f"'{self._model_path}'"
                )

    def save(self, output_path: str = None, **kwargs):
        """
        Save the handled model at the given output path. If a MLRun context is available, the saved model files will be
        logged and returned as artifacts.

        :param output_path: The full path to the directory to save the handled model at. If not given, the context
                            stored will be used to save the model in the defaulted artifacts location.

        :return The saved model additional artifacts (if needed) dictionary if context is available and None otherwise.
        """
        super(LGBMModelHandler, self).save(output_path=output_path)

        # ModelFormats.PICKLE - Save from a pkl file:
        if self._model_format == LGBMModelHandler.ModelFormats.PKL:
            self._model_file = f"{self._model_name}.pkl"
            with open(self._model_file, "wb") as pickle_file:
                cloudpickle.dump(self._model, pickle_file)

        return None

    def load(self, **kwargs):
        """
        Load the specified model in this handler. Additional parameters for the class initializer can be passed via the
        kwargs dictionary.
        """
        super(LGBMModelHandler, self).load()

        # ModelFormats.PICKLE - Load from a pkl file:
        if self._model_format == LGBMModelHandler.ModelFormats.PKL:
            with open(self._model_file, "rb") as pickle_file:
                self._model = pickle.load(pickle_file)

    def to_onnx(
        self,
        model_name: str = None,
        optimize: bool = True,
        input_sample: IOSample = None,
        log: bool = None,
    ):
        """
        Convert the model in this handler to an ONNX model. The inputs names are optional, they do not change the
        semantics of the model, it is only for readability.

        :param model_name:          The name to give to the converted ONNX model. If not given the default name will be
                                    the stored model name with the suffix '_onnx'.
        :param optimize:            Whether to optimize the ONNX model using 'onnxoptimizer' before saving the model.
                                    Defaulted to True.
        :param input_sample:        An inputs sample with the names and data types of the inputs of the model.
        :param log:                 In order to log the ONNX model, pass True. If None, the model will be logged if this
                                    handler has a MLRun context set. Defaulted to None.

        :return: The converted ONNX model (onnx.ModelProto).

        :raise MLRunMissingDependencyError: If some of the ONNX packages are missing.
        """
        # Import onnx related modules:
        try:
            pass
            # import onnxmltools

            # from mlrun.frameworks.onnx import ONNXModelHandler
        except ModuleNotFoundError:
            raise mlrun.errors.MLRunMissingDependencyError(
                "ONNX conversion requires additional packages to be installed. "
                "Please run 'pip install mlrun[lgbm]' to install MLRun's XGBoost package."
            )

        raise NotImplementedError  # TODO: Finish ONNX conversion
