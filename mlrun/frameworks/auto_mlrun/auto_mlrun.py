# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx
from typing import Callable, Dict, List, Type, Union

import mlrun
from mlrun.artifacts import get_model

from .._common import ModelHandler, ModelType


def get_framework_by_instance(model: ModelType) -> str:
    """
    Get the framework name of the given model by its instance.

    :param model: The model to get his framework.

    :return: The model's framework.

    :raise MLRunInvalidArgumentError: If the given model type is not supported by AutoMLRun or not recognized.
    """
    # TF.Keras:
    try:
        from tensorflow.keras import Model

        from mlrun.frameworks.tf_keras import TFKerasModelHandler

        if isinstance(model, Model):
            return TFKerasModelHandler.FRAMEWORK_NAME
    except ModuleNotFoundError:
        pass

    # PyTorch:
    try:
        from torch.nn import Module

        from mlrun.frameworks.pytorch import PyTorchModelHandler

        if isinstance(model, Module):
            return PyTorchModelHandler.FRAMEWORK_NAME
    except ModuleNotFoundError:
        pass

    # XGBoost:
    try:
        from xgboost import XGBModel

        from mlrun.frameworks.xgboost import XGBoostModelHandler

        if isinstance(model, XGBModel):
            return XGBoostModelHandler.FRAMEWORK_NAME
    except ModuleNotFoundError:
        pass

    # SciKit-Learn:
    try:
        import inspect

        import sklearn.base as sklearn_base

        from mlrun.frameworks.sklearn import SKLearnModelHandler

        if isinstance(
            model,
            tuple(
                [
                    class_tuple[1]
                    for class_tuple in inspect.getmembers(sklearn_base, inspect.isclass)
                    if "sklearn" in str(class_tuple[1])
                ]
            ),
        ):
            return SKLearnModelHandler.FRAMEWORK_NAME
    except ModuleNotFoundError:
        pass

    # LightGBM:
    try:
        from lightgbm import Booster, LGBMModel

        from mlrun.frameworks.lgbm import LGBMModelHandler

        if isinstance(model, (LGBMModel, Booster)):
            return LGBMModelHandler.FRAMEWORK_NAME
    except ModuleNotFoundError:
        pass

    # ONNX:
    try:
        from onnx import ModelProto

        from mlrun.frameworks.onnx import ONNXModelHandler

        if isinstance(model, ModelProto):
            return ONNXModelHandler.FRAMEWORK_NAME
    except ModuleNotFoundError:
        pass

    # Unrecognized:
    raise mlrun.errors.MLRunInvalidArgumentError(
        f"The type of the model: {type(model)} was not recognized to be of any of the supported frameworks in "
        f"MLRun or perhaps the required package of the framework could not be imported. If the model's framework is "
        f"one of: TF.Keras, PyTorch, XGBoost, LightGBM, SciKit-Learn, please provide the 'framework' parameter to the "
        f"function."
    )


def get_framework_by_class_name(model: ModelType) -> str:
    """
    Get the framework name of the given model by its class name.

    :param model: The model to get its framework.

    :return: The model's framework.

    :raise MLRunInvalidArgumentError: If the given model's class name is not supported by AutoMLRun or not recognized.
    """
    # Read the class name:
    class_name = str(model.__class__)

    # Look for the correct framework:
    if "tensorflow" in class_name:
        from mlrun.frameworks.tf_keras import TFKerasModelHandler

        return TFKerasModelHandler.FRAMEWORK_NAME
    if "torch" in class_name:
        from mlrun.frameworks.pytorch import PyTorchModelHandler

        return PyTorchModelHandler.FRAMEWORK_NAME
    if "sklearn" in class_name:
        from mlrun.frameworks.sklearn import SKLearnModelHandler

        return SKLearnModelHandler.FRAMEWORK_NAME
    if "xgboost" in class_name:
        from mlrun.frameworks.xgboost import XGBoostModelHandler

        return XGBoostModelHandler.FRAMEWORK_NAME
    if "lightgbm" in class_name:
        from mlrun.frameworks.lgbm import LGBMModelHandler

        return LGBMModelHandler.FRAMEWORK_NAME
    if "onnx" in class_name:
        from mlrun.frameworks.onnx import ONNXModelHandler

        return ONNXModelHandler.FRAMEWORK_NAME

    # Framework was not recognized:
    raise mlrun.errors.MLRunInvalidArgumentError(
        f"The model's class name: {class_name} was not recognized to be of any of the supported frameworks in "
        f"MLRun. If the model's framework is one of: TF.Keras, PyTorch, XGBoost, LightGBM, SciKit-Learn, "
        f"please provide the 'framework' parameter to the function."
    )


def framework_to_model_handler(framework: str) -> Type[ModelHandler]:
    """
    Get the ModelHandler class of the given framework's name.

    :param framework: The framework's name.

    :return: The framework's ModelHandler class.

    :raise MLRunInvalidArgumentError: If the given framework is not supported by AutoMLRun.
    """
    # Match the framework:
    if framework == "tf.keras":
        from mlrun.frameworks.tf_keras import TFKerasModelHandler

        return TFKerasModelHandler
    if framework == "pytorch":
        from mlrun.frameworks.pytorch import PyTorchModelHandler

        return PyTorchModelHandler
    if framework == "sklearn":
        from mlrun.frameworks.sklearn import SKLearnModelHandler

        return SKLearnModelHandler
    if framework == "xgboost":
        from mlrun.frameworks.xgboost import XGBoostModelHandler

        return XGBoostModelHandler
    if framework == "lightgbm":
        from mlrun.frameworks.lgbm import LGBMModelHandler

        return LGBMModelHandler
    if framework == "onnx":
        from mlrun.frameworks.onnx import ONNXModelHandler

        return ONNXModelHandler

    # Framework was not recognized:
    raise mlrun.errors.MLRunInvalidArgumentError(
        f"The framework {framework} is not supported yet by AutoMLRun. "
        f"Please use the ModelHandler class of the framework from mlrun.frameworks.{framework}"
    )


def framework_to_apply_mlrun(framework: str) -> Callable[..., ModelHandler]:
    """
    Get the 'apply_mlrun' shortcut function of the given framework's name.

    :param framework: The framework's name.

    :return: The framework's 'apply_mlrun' shortcut function.

    :raise MLRunInvalidArgumentError: If the given framework is not supported by AutoMLRun or if it does not have an
                                      'apply_mlrun' yet.
    """
    # Match the framework:
    if framework == "tf.keras":
        from mlrun.frameworks.tf_keras import apply_mlrun

        return apply_mlrun
    if framework == "pytorch":
        raise mlrun.errors.MLRunInvalidArgumentError(
            "PyTorch has no 'apply_mlrun' shortcut yet. Please use the MLRunPyTorchInterface "
            "from mlrun.frameworks.pytorch instead."
        )
    if framework == "sklearn":
        from mlrun.frameworks.sklearn import apply_mlrun

        return apply_mlrun
    if framework == "xgboost":
        from mlrun.frameworks.xgboost import apply_mlrun

        return apply_mlrun
    if framework == "lightgbm":
        from mlrun.frameworks.lgbm import apply_mlrun

        return apply_mlrun
    if framework == "onnx":
        raise mlrun.errors.MLRunInvalidArgumentError(
            "ONNX has no 'apply_mlrun' shortcut yet. Please use the MLRunONNXInterface from mlrun.frameworks.onnx "
            "instead."
        )

    # Framework was not recognized:
    raise mlrun.errors.MLRunInvalidArgumentError(
        f"The framework {framework} is not supported yet by AutoMLRun. "
        f"Please use the ModelHandler class of the framework from mlrun.frameworks.{framework}"
    )


class AutoMLRun:
    """
    A library of automatic functions for managing models using MLRun's frameworks package.
    """

    @staticmethod
    def load_model(
        model_path: str, context: mlrun.MLClientCtx = None, **kwargs
    ) -> ModelHandler:
        """
        Load a logged model using MLRun's ModelHandler. The loaded model can be accessed from the model handler returned
        via model_handler.model.

        :param model_path: A store object path of a logged model object in MLRun.
        :param context:    A MLRun context.
        :param kwargs:     Additional parameters for the specific framework's ModelHandler class.

        :return: The model inside a MLRun model handler.
        """
        # Get the model primary file, extra data and artifact:
        model_file, model_artifact, extra_data = get_model(model_path)

        # Get the model's framework:
        framework = model_artifact.labels["framework"]  # type: str

        # Get the ModelHandler according to the framework:
        model_handler_class = framework_to_model_handler(framework=framework)

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

    @staticmethod
    def apply_mlrun(
        model: ModelType,
        model_name: str = None,
        modules_map: Union[Dict[str, Union[None, str, List[str]]], str] = None,
        custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
        custom_objects_directory: str = None,
        context: mlrun.MLClientCtx = None,
        framework: str = None,
        **kwargs,
    ) -> ModelHandler:
        """
        Use MLRun's 'apply_mlrun' of the detected given model's framework to wrap the framework relevant methods and
        gain the framework's features in MLRun. A ModelHandler initialized with the model will be returned.

        :param model:                    Model to handle or None in case a loading parameters were supplied.
        :param model_name:               The model name for saving and logging the model.
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
        :param context:                  A MLRun context.
        :param framework:                The model's framework. If None, AutoMLRun will try to figure out the framework.
                                         Defaulted to None.
        :param kwargs:                   Additional parameters for the specific framework's 'apply_mlrun' function.

        :return: The model handler initialized with the given model.
        """
        # Get the model's framework:
        framework = (
            get_framework_by_class_name(model=model) if framework is None else framework
        )

        # Get the framework's 'apply_mlrun':
        apply_mlrun = framework_to_apply_mlrun(framework=framework)

        # Initialize the model handler:
        return apply_mlrun(
            model=model,
            model_name=model_name,
            modules_map=modules_map,
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
            context=context,
            **kwargs,
        )
