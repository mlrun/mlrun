# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx
from typing import Callable, Dict, List, Tuple, Type, Union

import mlrun
from mlrun.artifacts import get_model

from .._common import CommonTypes, ModelHandler


def get_framework_by_instance(model: CommonTypes.ModelType) -> str:
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
        from xgboost import Booster, XGBModel

        from mlrun.frameworks.xgboost import XGBoostModelHandler

        if isinstance(model, (XGBModel, Booster)):
            return XGBoostModelHandler.FRAMEWORK_NAME
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

    # SciKit-Learn: (As SKLearn's API is commonly used in other frameworks, its important to check it last)
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

    # Unrecognized:
    raise mlrun.errors.MLRunInvalidArgumentError(
        f"The type of the model: {type(model)} was not recognized to be of any of the supported frameworks in "
        f"MLRun or perhaps the required package of the framework could not be imported. If the model's framework is "
        f"one of: TF.Keras, PyTorch, XGBoost, LightGBM, SciKit-Learn, please provide the 'framework' parameter to the "
        f"function."
    )


def get_framework_by_class_name(model: CommonTypes.ModelType) -> str:
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
    if "xgboost" in class_name:
        from mlrun.frameworks.xgboost import XGBoostModelHandler

        return XGBoostModelHandler.FRAMEWORK_NAME
    if "lightgbm" in class_name:
        from mlrun.frameworks.lgbm import LGBMModelHandler

        return LGBMModelHandler.FRAMEWORK_NAME
    if "onnx" in class_name:
        from mlrun.frameworks.onnx import ONNXModelHandler

        return ONNXModelHandler.FRAMEWORK_NAME
    # As SKLearn's API is commonly used in other frameworks, its important to check it last:
    if "sklearn" in class_name:
        from mlrun.frameworks.sklearn import SKLearnModelHandler

        return SKLearnModelHandler.FRAMEWORK_NAME

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
    if framework == "tensorflow.keras":
        from mlrun.frameworks.tf_keras import TFKerasModelHandler

        return TFKerasModelHandler
    if framework == "torch":
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
    if framework == "tensorflow.keras":
        from mlrun.frameworks.tf_keras import apply_mlrun

        return apply_mlrun
    if framework == "torch":
        raise mlrun.errors.MLRunInvalidArgumentError(
            "PyTorch has no 'apply_mlrun' shortcut yet. Please use the PyTorchMLRunInterface "
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
            "ONNX has no 'apply_mlrun' shortcut yet. Please use the ONNXMLRunInterface from mlrun.frameworks.onnx "
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
    def _get_framework(
        model: CommonTypes.ModelType = None, model_path: str = None
    ) -> Union[Tuple[str, dict]]:
        """
        Try to get the framework from the model or model path provided. The framework can be read from the model path
        only if the model path is of a logged model artifact (store object uri).

        :param model:      The model instance to get its framework.
        :param model_path: The store object uri of a model artifact.

        :return: A tuple of:
                 [0] = The framework.
                 [1] = Collected data (model_file, model_artifact, extra_data) if read from model path or an empty
                       dictionary if read from model.

        :raise MLRunInvalidArgumentError: If both values were None, the framework is not recognized / supported or the
                                          the given model path is not of a store object.
        """
        # If the model is provided:
        if model is not None:
            try:
                # Try to get the framework by the class name:
                return get_framework_by_class_name(model=model), {}
            except mlrun.errors.MLRunInvalidArgumentError:
                # Try to get the framework by the instance's type:
                return get_framework_by_instance(model=model), {}

        # If the model path is provided:
        if model_path is not None:
            # Get the model primary file, extra data and artifact:
            model_file, model_artifact, extra_data = get_model(model_path)
            # If the model path is not of a store object, `model_artifact` will be None:
            if model_artifact is None:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"The model path provided: '{model_path}' is not of a model artifact (store uri) so the framework "
                    f"attribute must be specified."
                )
            # Return the framework and the collected files and artifacts:
            return (
                model_artifact.labels["framework"],
                {
                    "model_file": model_file,
                    "model_artifact": model_artifact,
                    "extra_data": extra_data,
                },
            )

        # Both values were None:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "At least 'mode' or 'model_path' must be given in order to use AutoMLRun."
        )

    @staticmethod
    def load_model(
        model_path: str,
        model_name: str = None,
        context: mlrun.MLClientCtx = None,
        modules_map: Union[Dict[str, Union[None, str, List[str]]], str] = None,
        custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
        custom_objects_directory: str = None,
        framework: str = None,
        **kwargs,
    ) -> ModelHandler:
        """
        Load a model using MLRun's ModelHandler. The loaded model can be accessed from the model handler returned
        via model_handler.model. If the model is a store object uri (it is logged in MLRun) then the framework will be
        read automatically, otherwise (for local path and urls) it must be given. The other parameters will be
        automatically read in case its a logged model in MLRun.

        :param model_path:               A store object path of a logged model object in MLRun.
        :param model_name:               The model name to use for storing the model artifact. If not given will have a
                                         default name according to the framework.
        :param modules_map:              A dictionary of all the modules required for loading the model. Each key is a
                                         path to a module and its value is the object name to import from it. All the
                                         modules will be imported globally. If multiple objects needed to be imported
                                         from the same module a list can be given. The map can be passed as a path to a
                                         json file as well. For example:

                                         .. code-block:: python

                                             {
                                                 "module1": None,  # import module1
                                                 "module2": ["func1", "func2"],  # from module2 import func1, func2
                                                 "module3.sub_module": "func3",  # from module3.sub_module import func3
                                             }

                                         If the model path given is of a store object, the modules map will be read from
                                         the logged modules map artifact of the model.
        :param custom_objects_map:       A dictionary of all the custom objects required for loading the model. Each key
                                         is a path to a python file and its value is the custom object name to import
                                         from it. If multiple objects needed to be imported from the same py file a list
                                         can be given. The map can be passed as a path to a json file as well. For
                                         example:

                                         .. code-block:: python

                                             {
                                                 "/.../custom_model.py": "MyModel",
                                                 "/.../custom_objects.py": ["object1", "object2"]
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
        :param framework:                The model's framework. It must be provided for local paths or urls. If None,
                                         AutoMLRun will assume the model path is of a store uri model artifact and try
                                         to get the framework from it. Default: None.
        :param kwargs:                   Additional parameters for the specific framework's ModelHandler class.

        :return: The model inside a MLRun model handler.

        :raise MLRunInvalidArgumentError: In case the framework is incorrect or missing.
        """
        # Get the model's framework (if needed):
        collected_model_file_and_artifact = {}
        if framework is None:
            framework, collected_model_file_and_artifact = AutoMLRun._get_framework(
                model_path=model_path
            )

        # Get the ModelHandler according to the framework:
        model_handler_class = framework_to_model_handler(framework=framework)

        # Initialize the model handler:
        model_handler = model_handler_class(
            model_path=model_path,
            model_name=model_name,
            modules_map=modules_map,
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
            context=context,
            **collected_model_file_and_artifact,
            **kwargs,
        )

        # Load the model and return the handler:
        model_handler.load()
        return model_handler

    @staticmethod
    def apply_mlrun(
        model: CommonTypes.ModelType = None,
        model_name: str = None,
        tag: str = "",
        model_path: str = None,
        modules_map: Union[Dict[str, Union[None, str, List[str]]], str] = None,
        custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
        custom_objects_directory: str = None,
        context: mlrun.MLClientCtx = None,
        framework: str = None,
        auto_log: bool = True,
        **kwargs,
    ) -> ModelHandler:
        """
        Use MLRun's 'apply_mlrun' of the detected given model's framework to wrap the framework relevant methods and
        gain the framework's features in MLRun. A ModelHandler initialized with the model will be returned.

        :param model:                    The model to wrap. Can be loaded from the model path given as well.
        :param model_name:               The model name to use for storing the model artifact. If not given will have a
                                         default name according to the framework.
        :param tag:                      The model's tag to log with.
        :param model_path:               The model's store object path. Mandatory for evaluation (to know which model to
                                         update). If model is not provided, it will be loaded from this path.
        :param modules_map:              A dictionary of all the modules required for loading the model. Each key is a
                                         path to a module and its value is the object name to import from it. All the
                                         modules will be imported globally. If multiple objects needed to be imported
                                         from the same module a list can be given. The map can be passed as a path to a
                                         json file as well. For example:

                                         .. code-block:: python

                                             {
                                                 "module1": None,  # import module1
                                                 "module2": ["func1", "func2"],  # from module2 import func1, func2
                                                 "module3.sub_module": "func3",  # from module3.sub_module import func3
                                             }

                                         If the model path given is of a store object, the modules map will be read from
                                         the logged modules map artifact of the model.
        :param custom_objects_map:       A dictionary of all the custom objects required for loading the model. Each key
                                         is a path to a python file and its value is the custom object name to import
                                         from it. If multiple objects needed to be imported from the same py file a list
                                         can be given. The map can be passed as a path to a json file as well. For
                                         example:

                                         .. code-block:: python

                                             {
                                                 "/.../custom_model.py": "MyModel",
                                                 "/.../custom_objects.py": ["object1", "object2"]
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
        :param auto_log:                 Whether to enable auto-logging capabilities of MLRun or not. Auto logging will
                                         add default artifacts and metrics besides the one you can pass here.
        :param framework:                The model's framework. If None, AutoMLRun will try to figure out the framework.
                                         From the provided model or model path. Default: None.
        :param kwargs:                   Additional parameters for the specific framework's 'apply_mlrun' function like
                                         metrics, callbacks and more (read the docs of the required framework to know
                                         more).

        :return: The framework's model handler initialized with the given model.
        """
        # Get the model's framework:
        collected_model_file_and_artifact = {}
        if framework is None:
            framework, collected_model_file_and_artifact = AutoMLRun._get_framework(
                model=model, model_path=model_path
            )

        # Get the framework's 'apply_mlrun':
        apply_mlrun = framework_to_apply_mlrun(framework=framework)

        # Initialize the model handler:
        return apply_mlrun(
            model=model,
            model_name=model_name,
            tag=tag,
            model_path=model_path,
            modules_map=modules_map,
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
            context=context,
            auto_log=auto_log,
            model_handler_kwargs=collected_model_file_and_artifact,
            **kwargs,
        )
