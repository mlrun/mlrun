# Copyright 2023 Iguazio
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

from typing import Any, Optional, Union

from tensorflow import keras

import mlrun
import mlrun.common.constants as mlrun_constants

from .callbacks import MLRunLoggingCallback, TensorboardLoggingCallback
from .mlrun_interface import TFKerasMLRunInterface
from .model_handler import TFKerasModelHandler
from .model_server import TFKerasModelServer
from .utils import TFKerasTypes, TFKerasUtils


def apply_mlrun(
    model: keras.Model = None,
    model_name: Optional[str] = None,
    tag: str = "",
    model_path: Optional[str] = None,
    model_format: str = TFKerasModelHandler.ModelFormats.SAVED_MODEL,
    save_traces: bool = False,
    modules_map: Optional[Union[dict[str, Union[None, str, list[str]]], str]] = None,
    custom_objects_map: Optional[Union[dict[str, Union[str, list[str]]], str]] = None,
    custom_objects_directory: Optional[str] = None,
    context: mlrun.MLClientCtx = None,
    auto_log: bool = True,
    tensorboard_directory: Optional[str] = None,
    mlrun_callback_kwargs: Optional[dict[str, Any]] = None,
    tensorboard_callback_kwargs: Optional[dict[str, Any]] = None,
    use_horovod: Optional[bool] = None,
    **kwargs,
) -> TFKerasModelHandler:
    """
    Wrap the given model with MLRun's interface providing it with mlrun's additional features.

    :param model:                       The model to wrap. Can be loaded from the model path given as well.
    :param model_name:                  The model name to use for storing the model artifact. If not given, the
                                        tf.keras.Model.name will be used.
    :param tag:                         The model's tag to log with.
    :param model_path:                  The model's store object path. Mandatory for evaluation (to know which model to
                                        update). If model is not provided, it will be loaded from this path.
    :param model_format:                The format to use for saving and loading the model. Should be passed as a
                                        member of the class 'ModelFormats'. Default: 'ModelFormats.SAVED_MODEL'.
    :param save_traces:                 Whether or not to use functions saving (only available for the 'SavedModel'
                                        format) for loading the model later without the custom objects dictionary. Only
                                        from tensorflow version >= 2.4.0. Using this setting will increase the model
                                        saving size.
    :param modules_map:                 A dictionary of all the modules required for loading the model. Each key is a
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
    :param custom_objects_map:          A dictionary of all the custom objects required for loading the model. Each key
                                        is a path to a python file and its value is the custom object name to import
                                        from it. If multiple objects needed to be imported from the same py file a list
                                        can be given. The map can be passed as a path to a json file as well. For
                                        example:

                                        .. code-block:: python

                                            {
                                                "/.../custom_optimizer.py": "optimizer",
                                                "/.../custom_layers.py": ["layer1", "layer2"],
                                            }

                                        All the paths will be accessed from the given 'custom_objects_directory',
                                        meaning each py file will be read from 'custom_objects_directory/<MAP VALUE>'.
                                        If the model path given is of a store object, the custom objects map will be
                                        read from the logged custom object map artifact of the model.
                                        Notice: The custom objects will be imported in the order they came in this
                                        dictionary (or json). If a custom object is depended on another, make sure to
                                        put it below the one it relies on.
    :param custom_objects_directory:    Path to the directory with all the python files required for the custom objects.
                                        Can be passed as a zip file as well (will be extracted during the run before
                                        loading the model). If the model path given is of a store object, the custom
                                        objects files will be read from the logged custom object artifact of the model.
    :param context:                     MLRun context to work with. If no context is given it will be retrieved via
                                        'mlrun.get_or_create_ctx(None)'
    :param auto_log:                    Whether or not to apply MLRun's auto logging on the model. Default: True.
    :param tensorboard_directory:       If context is not given, or if wished to set the directory even with context,
                                        this will be the output for the event logs of tensorboard. If not given, the
                                        'tensorboard_dir' parameter will be tried to be taken from the provided context.
                                        If not found in the context, the default tensorboard output directory will be:
                                        /User/.tensorboard/<PROJECT_NAME> or if working on local, the set artifacts
                                        path.
    :param mlrun_callback_kwargs:       Key word arguments for the MLRun callback. For further information see the
                                        documentation of the class 'MLRunLoggingCallback'. Note that both 'context'
                                        and 'auto_log' parameters are already given here.
    :param tensorboard_callback_kwargs: Key word arguments for the tensorboard callback. For further information see
                                        the documentation of the class 'TensorboardLoggingCallback'. Note that both
                                        'context' and 'auto_log' parameters are already given here.
    :param use_horovod:                 Whether or not to use horovod - a distributed training framework. Default:
                                        None, meaning it will be read from context if available and if not - False.

    :return: The model with MLRun's interface.
    """
    # Get parameters defaults:
    # # Context:
    if context is None:
        context = mlrun.get_or_create_ctx(TFKerasMLRunInterface.DEFAULT_CONTEXT_NAME)
    # # Use horovod:
    if use_horovod is None:
        use_horovod = (
            context.labels.get(mlrun_constants.MLRunInternalLabels.kind, "") == "mpijob"
            if context is not None
            else False
        )

    # Create a model handler:
    model_handler_kwargs = (
        kwargs.pop("model_handler_kwargs") if "model_handler_kwargs" in kwargs else {}
    )
    handler = TFKerasModelHandler(
        model_name=model_name,
        model_path=model_path,
        model=model,
        model_format=model_format,
        save_traces=save_traces,
        context=context,
        modules_map=modules_map,
        custom_objects_map=custom_objects_map,
        custom_objects_directory=custom_objects_directory,
        **model_handler_kwargs,
    )

    # Load the model if it was not provided:
    if model is None:
        handler.load()
        model = handler.model

    # Add MLRun's interface to the model:
    TFKerasMLRunInterface.add_interface(obj=model)

    # Initialize horovod if needed:
    if use_horovod is True:
        model.use_horovod()

    # Add auto-logging if needed:
    if auto_log:
        # Set the kwargs dictionaries defaults:
        mlrun_callback_kwargs = (
            {} if mlrun_callback_kwargs is None else mlrun_callback_kwargs
        )
        tensorboard_callback_kwargs = (
            {} if tensorboard_callback_kwargs is None else tensorboard_callback_kwargs
        )
        # Add the logging callbacks with the provided parameters:
        model.add_logging_callback(
            logging_callback=MLRunLoggingCallback(
                context=context,
                model_handler=handler,
                log_model_tag=tag,
                auto_log=auto_log,
                **mlrun_callback_kwargs,
            )
        )
        model.add_logging_callback(
            logging_callback=TensorboardLoggingCallback(
                context=context,
                tensorboard_directory=tensorboard_directory,
                auto_log=auto_log,
                **tensorboard_callback_kwargs,
            )
        )

    return handler
