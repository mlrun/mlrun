# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx
from typing import Type

import mlrun
from mlrun.artifacts import get_model

from .._common import ModelHandler


class AutoMLRun:
    """
    A library of automatic functions for managing models using MLRun's framework features.
    """

    @staticmethod
    def _framework_to_model_handler(framework: str) -> Type[ModelHandler]:
        """
        Get the ModelHandler class of the given framework.

        :param framework: The framework's name.

        :return: The framework's ModelHandler class.

        :raise MLRunInvalidArgumentError: If the given framework is not supported by AutoMLRun.
        """
        if framework == "tf.keras":
            from mlrun.frameworks.tf_keras import TFKerasModelHandler

            return TFKerasModelHandler
        elif framework == "pytorch":
            from mlrun.frameworks.pytorch import PyTorchModelHandler

            return PyTorchModelHandler
        elif framework == "sklearn":
            from mlrun.frameworks.sklearn import SKLearnModelHandler

            return SKLearnModelHandler
        elif framework == "xgboost":
            from mlrun.frameworks.xgboost import XGBoostModelHandler

            return XGBoostModelHandler
        elif framework == "lightgbm":
            from mlrun.frameworks.lgbm import LGBMModelHandler

            return LGBMModelHandler
        elif framework == "onnx":
            from mlrun.frameworks.onnx import ONNXModelHandler

            return ONNXModelHandler
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The framework {framework} is not supported yet by AutoMLRun. "
                f"Please use the ModelHandler class of the framework from mlrun.frameworks.{framework}"
            )

    @staticmethod
    def load_model(
        model_path: str, context: mlrun.MLClientCtx = None, **kwargs
    ) -> ModelHandler:
        """
        Load a logged model using MLRun's ModelHandler. The loaded model can be accessed from the model handler returned
        via model_handler.model.

        :param model_path: A store object path of a logged model object in MLRun.
        :param context:    An MLRun context.
        :param kwargs:     Additional parameters for the specific framework;s ModelHandler class.

        :return: The model inside a MLRun model handler.
        """
        # Get the model primary file, extra data and artifact:
        model_file, model_artifact, extra_data = get_model(model_path)

        # Get the model's framework:
        framework = model_artifact.labels["framework"]  # type: str

        # Get the ModelHandler according to the framework:
        model_handler_class = AutoMLRun._framework_to_model_handler(framework=framework)

        # Initialize the model handler:
        model_handler = model_handler_class(
            model_path=model_path,
            context=context,
            model_file=model_file,
            model_artifact=model_artifact,
            extra_data=extra_data,
            **kwargs,
        )

        # Load the model and return the handler:
        model_handler.load()
        return model_handler
