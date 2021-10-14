# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx
from typing import Any, Dict, List, Union

from tensorflow import keras

import mlrun
import mlrun.frameworks.tf_keras.callbacks
from mlrun.frameworks.tf_keras.mlrun_interface import TFKerasMLRunInterface
from mlrun.frameworks.tf_keras.model_handler import TFKerasModelHandler
from mlrun.frameworks.tf_keras.model_server import TFKerasModelServer


def apply_mlrun(
    model: keras.Model,
    model_name: str = None,
    model_path: str = None,
    custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
    custom_objects_directory: str = None,
    context: mlrun.MLClientCtx = None,
    auto_log: bool = True,
    tensorboard_directory: str = None,
    mlrun_callback_kwargs: Dict[str, Any] = None,
    tensorboard_callback_kwargs: Dict[str, Any] = None,
    use_horovod: bool = None,
) -> keras.Model:
    """
    Wrap the given model with MLRun model, saving the model's attributes and methods while giving it mlrun's additional
    features.

    :param model:                       The model to wrap.
    :param model_name:                  The model name to use for storing the model artifact. If not given, the
                                        tf.keras.Model.name will be used.
    :param model_path:                  The model's store object path. Mandatory for evaluation (to know which model to
                                        update).
    :param custom_objects_map:          A dictionary of all the custom objects required for loading the model. Each key
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
    :param custom_objects_directory:    Path to the directory with all the python files required for the custom objects.
                                        Can be passed as a zip file as well (will be extracted during the run before
                                        loading the model). If the model path given is of a store object, the custom
                                        objects files will be read from the logged custom object artifact of the model.
    :param context:                     MLRun context to work with. If no context is given it will be retrieved via
                                        'mlrun.get_or_create_ctx(None)'
    :param auto_log:                    Whether or not to apply MLRun's auto logging on the model. Defaulted to True.
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
    :param use_horovod:                 Whether or not to use horovod - a distributed training framework. Defaulted to
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
            context.labels.get("kind", "") == "mpijob" if context is not None else False
        )

    # Add MLRun's interface to the model:
    TFKerasMLRunInterface.add_interface(model=model)

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
        # Add the additional parameters to Tensorboard's callback kwargs dictionary:
        tensorboard_callback_kwargs["tensorboard_directory"] = tensorboard_directory
        # Add the additional parameters to MLRun's callback kwargs dictionary:
        mlrun_callback_kwargs["model_name"] = model_name
        mlrun_callback_kwargs["model_path"] = model_path
        mlrun_callback_kwargs["custom_objects_map"] = custom_objects_map
        mlrun_callback_kwargs["custom_objects_directory"] = custom_objects_directory
        # Add the logging callbacks with the provided parameters:
        model.auto_log(
            context=context,
            mlrun_callback_kwargs=mlrun_callback_kwargs,
            tensorboard_callback_kwargs=tensorboard_callback_kwargs,
        )

    return model
