import os
import pickle
from typing import List, Tuple, Union

import cloudpickle
import numpy as np
import pandas as pd
import sklearn

import mlrun

from .._ml_common import MLModelHandler


class SKLearnModelHandler(MLModelHandler):
    """
    Class for handling a SciKitLearn model, enabling loading and saving it during runs.
    """

    # Framework name:
    FRAMEWORK_NAME = "sklearn"

    # Declare a type of a SciKitLearn model:
    SKLearnModel = Union[
        sklearn.base.BaseEstimator,
        sklearn.base.BiclusterMixin,
        sklearn.base.ClassifierMixin,
        sklearn.base.ClusterMixin,
        sklearn.base.DensityMixin,
        sklearn.base.RegressorMixin,
        sklearn.base.TransformerMixin,
    ]

    # Declare a type of an input sample:
    IOSample = Union[pd.DataFrame, np.ndarray, List[Tuple[str, str]]]

    def _collect_files_from_local_path(self):
        """
        If the model path given is of a local path, search for the needed model files and collect them into this handler
        for later loading the model.

        :raise MLRunNotFoundError: If the model file was not found.
        """
        # Get the pickle model file:
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
        super(SKLearnModelHandler, self).save(output_path=output_path)

        # Save the model pkl file:
        self._model_file = f"{self._model_name}.pkl"
        with open(self._model_file, "wb") as pickle_file:
            cloudpickle.dump(self._model, pickle_file)

        return None

    def load(self, **kwargs):
        """
        Load the specified model in this handler. Additional parameters for the class initializer can be passed via the
        kwargs dictionary.
        """
        super(SKLearnModelHandler, self).load()

        # Load from a pkl file:
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
            # import skl2onnx

            # from mlrun.frameworks.onnx import ONNXModelHandler
        except ModuleNotFoundError:
            raise mlrun.errors.MLRunMissingDependencyError(
                "ONNX conversion requires additional packages to be installed. "
                "Please run 'pip install mlrun[sklearn]' to install MLRun's XGBoost package."
            )

        raise NotImplementedError  # TODO: Finish ONNX conversion
