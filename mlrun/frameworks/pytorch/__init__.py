# flake8: noqa  - this is until we take care of the F401 violations with respect to __all__ & sphinx
from typing import Any, Dict, List, Tuple, Union

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import mlrun

from .callbacks import Callback, MetricFunctionType, MetricValueType
from .callbacks_handler import CallbacksHandler
from .mlrun_interface import PyTorchMLRunInterface
from .model_handler import PyTorchModelHandler
from .model_server import PyTorchModelServer


def train(
    model: Module,
    training_set: DataLoader,
    loss_function: Module,
    optimizer: Optimizer,
    validation_set: DataLoader = None,
    metric_functions: List[MetricFunctionType] = None,
    scheduler=None,
    scheduler_step_frequency: Union[int, float, str] = "epoch",
    epochs: int = 1,
    training_iterations: int = None,
    validation_iterations: int = None,
    callbacks_list: List[Callback] = None,
    use_cuda: bool = True,
    use_horovod: bool = False,
    auto_log: bool = True,
    model_name: str = None,
    modules_map: Union[Dict[str, Union[None, str, List[str]]], str] = None,
    custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
    custom_objects_directory: str = None,
    tensorboard_directory: str = None,
    mlrun_callback_kwargs: Dict[str, Any] = None,
    tensorboard_callback_kwargs: Dict[str, Any] = None,
    context: mlrun.MLClientCtx = None,
):
    """
    Use MLRun's PyTorch interface to train the model with the given parameters. For more information and further options
    regarding the auto logging, see 'PyTorchMLRunInterface' documentation. Notice for auto-logging: In order to log the
    model to MLRun, its class (torch.Module) must be in the custom objects map or the modules map.

    :param model:                       The model to train.
    :param training_set:                A data loader for the training process.
    :param loss_function:               The loss function to use during training.
    :param optimizer:                   The optimizer to use during the training.
    :param validation_set:              A data loader for the validation process.
    :param metric_functions:            The metrics to use on training and validation.
    :param scheduler:                   Scheduler to use on the optimizer at the end of each epoch. The scheduler must
                                        have a 'step' method with no input.
    :param scheduler_step_frequency:    The frequency in which to step the given scheduler. Can be equal to one of the
                                        strings 'epoch' (for at the end of every epoch) and 'batch' (for at the end of
                                        every batch), or an integer that specify per how many iterations to step or a
                                        float percentage (0.0 < x < 1.0) for per x / iterations to step. Defaulted to
                                        'epoch'.
    :param epochs:                      Amount of epochs to perform. Defaulted to a single epoch.
    :param training_iterations:         Amount of iterations (batches) to perform on each epoch's training. If 'None'
                                        the entire training set will be used.
    :param validation_iterations:       Amount of iterations (batches) to perform on each epoch's validation. If 'None'
                                        the entire validation set will be used.
    :param callbacks_list:              The callbacks to use on this run.
    :param use_cuda:                    Whether or not to use cuda. Only relevant if cuda is available. Defaulted to
                                        True.
    :param use_horovod:                 Whether or not to use horovod - a distributed training framework. Defaulted to
                                        False.
    :param auto_log:                    Whether or not to apply auto-logging (to both MLRun and Tensorboard). Defaulted
                                        to True. IF True, the custom objects are not optional.
    :param model_name:                  The model name to use for storing the model artifact. If not given, the model's
                                        class name will be used.
    :param modules_map:                 A dictionary of all the modules required for loading the model. Each key is a
                                        path to a module and its value is the object name to import from it. All the
                                        modules will be imported globally. If multiple objects needed to be imported
                                        from the same module a list can be given. The map can be passed as a path to a
                                        json file as well. For example:
                                        {
                                           "module1": None,  # => import module1
                                           "module2": ["func1", "func2"],  # => from module2 import func1, func2
                                           "module3.sub_module": "func3",  # => from module3.sub_module import func3
                                        }
                                        If the model path given is of a store object, the modules map will be read from
                                        the logged modules map artifact of the model.
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
    :param tensorboard_directory:       If context is not given, or if wished to set the directory even with context,
                                        this will be the output for the event logs of tensorboard. If not given, the
                                        'tensorboard_dir' parameter will be tried to be taken from the provided context.
                                        If not found in the context, the default tensorboard output directory will be:
                                        /User/.tensorboard/<PROJECT_NAME> or if working on local, the set artifacts
                                        path.
    :param mlrun_callback_kwargs:       Key word arguments for the MLRun callback. For further information see the
                                        documentation of the class 'MLRunLoggingCallback'. Note that both 'context',
                                        'custom_objects' and 'auto_log' parameters are already given here.
    :param tensorboard_callback_kwargs: Key word arguments for the tensorboard callback. For further information see
                                        the documentation of the class 'TensorboardLoggingCallback'. Note that both
                                        'context' and 'auto_log' parameters are already given here.
    :param context:                     The context to use for the logs.

    :return: The initialized trainer.

    :raise ValueError: If 'auto_log' is set to True and one of the custom objects parameter given is None.
    """
    # Initialize the interface:
    interface = PyTorchMLRunInterface(model=model, context=context)

    # Add auto logging:
    if auto_log:
        # Parse the custom objects and the kwargs:
        mlrun_callback_kwargs, tensorboard_callback_kwargs = _parse_callbacks_kwargs(
            model_name=model_name,
            model_path=None,
            modules_map=modules_map,
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
            tensorboard_directory=tensorboard_directory,
            mlrun_callback_kwargs=mlrun_callback_kwargs,
            tensorboard_callback_kwargs=tensorboard_callback_kwargs,
            is_training=True,
        )
        # Add the logging callbacks with the provided parameters:
        interface.add_auto_logging_callbacks(
            mlrun_callback_kwargs=mlrun_callback_kwargs,
            tensorboard_callback_kwargs=tensorboard_callback_kwargs,
        )

    # Train:
    interface.train(
        training_set=training_set,
        loss_function=loss_function,
        optimizer=optimizer,
        validation_set=validation_set,
        metric_functions=metric_functions,
        scheduler=scheduler,
        scheduler_step_frequency=scheduler_step_frequency,
        epochs=epochs,
        training_iterations=training_iterations,
        validation_iterations=validation_iterations,
        callbacks=callbacks_list,
        use_cuda=use_cuda,
        use_horovod=use_horovod,
    )


def evaluate(
    model: Module,
    model_path: str,
    dataset: DataLoader,
    loss_function: Module = None,
    metric_functions: List[MetricFunctionType] = None,
    iterations: int = None,
    callbacks_list: List[Callback] = None,
    use_cuda: bool = True,
    use_horovod: bool = False,
    auto_log: bool = True,
    model_name: str = None,
    modules_map: Union[Dict[str, Union[None, str, List[str]]], str] = None,
    custom_objects_map: Union[Dict[str, Union[str, List[str]]], str] = None,
    custom_objects_directory: str = None,
    mlrun_callback_kwargs: Dict[str, Any] = None,
    context: mlrun.MLClientCtx = None,
) -> List[MetricValueType]:
    """
    Use MLRun's PyTorch interface to evaluate the model with the given parameters. For more information and further
    options regarding the auto logging, see 'PyTorchMLRunInterface' documentation. Notice for auto-logging: In order to
    log the model to MLRun, its class (torch.Module) must be in the custom objects map or the modules map.

    :param model:                    The model to evaluate.
    :param model_path:               The model's store object path. Mandatory for evaluation (to know which model to
                                     update).
    :param dataset:                  A data loader for the validation process.
    :param loss_function:            The loss function to use during training.
    :param metric_functions:         The metrics to use on training and validation.
    :param iterations:               Amount of iterations (batches) to perform on the dataset. If 'None' the entire
                                     dataset will be used.
    :param callbacks_list:           The callbacks to use on this run.
    :param use_cuda:                 Whether or not to use cuda. Only relevant if cuda is available. Defaulted to True.
    :param use_horovod:              Whether or not to use horovod - a distributed training framework. Defaulted to
                                     False.
    :param auto_log:                 Whether or not to apply auto-logging to MLRun. Defaulted to True.
    :param model_name:               The model name to use for storing the model artifact. If not given, the model's
                                     class name will be used.
    :param modules_map:              A dictionary of all the modules required for loading the model. Each key is a path
                                     to a module and its value is the object name to import from it. All the modules
                                     will be imported globally. If multiple objects needed to be imported from the same
                                     module a list can be given. The map can be passed as a path to a json file as well.
                                     For example:
                                     {
                                         "module1": None,  # => import module1
                                         "module2": ["func1", "func2"],  # => from module2 import func1, func2
                                         "module3.sub_module": "func3",  # => from module3.sub_module import func3
                                     }
                                     If the model path given is of a store object, the modules map will be read from
                                     the logged modules map artifact of the model.
    :param custom_objects_map:       A dictionary of all the custom objects required for loading the model. Each key is
                                     a path to a python file and its value is the custom object name to import from it.
                                     If multiple objects needed to be imported from the same py file a list can be
                                     given. The map can be passed as a path to a json file as well. For example:
                                     {
                                         "/.../custom_optimizer.py": "optimizer",
                                         "/.../custom_layers.py": ["layer1", "layer2"]
                                     }
                                     All the paths will be accessed from the given 'custom_objects_directory', meaning
                                     each py file will be read from 'custom_objects_directory/<MAP VALUE>'. If the model
                                     path given is of a store object, the custom objects map will be read from the
                                     logged custom object map artifact of the model. Notice: The custom objects will be
                                     imported in the order they came in this dictionary (or json). If a custom object is
                                     depended on another, make sure to put it below the one it relies on.
    :param custom_objects_directory: Path to the directory with all the python files required for the custom objects.
                                     Can be passed as a zip file as well (will be extracted during the run before
                                     loading the model). If the model path given is of a store object, the custom
                                     objects files will be read from the logged custom object artifact of the model.
    :param mlrun_callback_kwargs:    Key word arguments for the MLRun callback. For further information see the
                                     documentation of the class 'MLRunLoggingCallback'. Note that both 'context',
                                     'custom_objects' and 'auto_log' parameters are already given here.
    :param context:                  The context to use for the logs.

    :return: The initialized evaluator.
    """
    # Initialize the interface:
    interface = PyTorchMLRunInterface(model=model, context=context)

    # Add auto logging:
    if auto_log:
        # Parse the custom objects and the kwargs:
        mlrun_callback_kwargs, _ = _parse_callbacks_kwargs(
            model_name=model_name,
            model_path=model_path,
            modules_map=modules_map,
            custom_objects_map=custom_objects_map,
            custom_objects_directory=custom_objects_directory,
            tensorboard_directory=None,
            mlrun_callback_kwargs=mlrun_callback_kwargs,
            tensorboard_callback_kwargs=None,
            is_training=False,
        )
        # Add the logging callbacks with the provided parameters:
        interface.add_auto_logging_callbacks(
            mlrun_callback_kwargs=mlrun_callback_kwargs, add_tensorboard_logger=False
        )

    # Evaluate:
    return interface.evaluate(
        dataset=dataset,
        loss_function=loss_function,
        metric_functions=metric_functions,
        iterations=iterations,
        callbacks=callbacks_list,
        use_cuda=use_cuda,
        use_horovod=use_horovod,
    )


def _parse_callbacks_kwargs(
    model_name: Union[str, None],
    model_path: Union[str, None],
    modules_map: Union[Dict[str, Union[None, str, List[str]]], str],
    custom_objects_map: Union[Dict[str, Union[str, List[str]]], str],
    custom_objects_directory: str,
    tensorboard_directory: Union[str, None],
    mlrun_callback_kwargs: Union[Dict[str, Any], None],
    tensorboard_callback_kwargs: Union[Dict[str, Any], None],
    is_training: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Parse the given parameters into the MLRun and Tensorboard callbacks kwargs.

    :param model_name:                  The model name to use for storing the model artifact. If not given, the model's
                                        class name will be used.
    :param model_path:                  The model's store object path. Mandatory for evaluation (to know which model to
                                        update).
    :param modules_map:                 A dictionary of all the modules required for loading the model. Each key is a
                                        path to a module and its value is the object name to import from it. All the
                                        modules will be imported globally. If multiple objects needed to be imported
                                        from the same module a list can be given. The map can be passed as a path to a
                                        json file as well. For example:
                                        {
                                           "module1": None,  # => import module1
                                           "module2": ["func1", "func2"],  # => from module2 import func1, func2
                                           "module3.sub_module": "func3",  # => from module3.sub_module import func3
                                        }
                                        If the model path given is of a store object, the modules map will be read from
                                        the logged modules map artifact of the model.
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
                                        Notice: The custom objects will be imported in the order they came in this
                                        dictionary (or json). If a custom object is depended on another, make sure to
                                        put it below the one it relies on.
    :param custom_objects_directory:    Path to the directory with all the python files required for the custom
                                        objects. Can be passed as a zip file as well (will be extracted during the run
                                        before loading the model).
    :param tensorboard_directory:       If context is not given, or if wished to set the directory even with context,
                                        this will be the output for the event logs of tensorboard. If not given, the
                                        'tensorboard_dir' parameter will be tried to be taken from the provided context.
                                        If not found in the context, the default tensorboard output directory will be:
                                        /User/.tensorboard/<PROJECT_NAME> or if working on local, the artifacts path.
    :param mlrun_callback_kwargs:       Key word arguments for the MLRun callback. For further information see the
                                        documentation of the class 'MLRunLoggingCallback'. Note that the 'context',
                                        'custom_objects' and 'auto_log' parameters are already given here.
    :param tensorboard_callback_kwargs: Key word arguments for the tensorboard callback. For further information see
                                        the documentation of the class 'TensorboardLoggingCallback'. Note that both
                                        'context' and 'auto_log' parameters are already given here.
    :param is_training:                 Whether to validate the arguments as part of a training session or evaluation
                                        session.

    :return: Tuple of the callbacks kwargs:
             [0] = MLRun's kwargs.
             [1] = Tensorboard kwargs.

    :raise MLRunInvalidArgumentError: In case of a training session: if one or more of the custom objects parameters
                                      were not given. In case of an evaluation session, if the model path was not given.
    """
    # Validate the custom objects parameters were provided:
    if is_training:
        if modules_map is None:
            if custom_objects_map is None or custom_objects_directory is None:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "If custom objects are required for loading the model, both custom objects parameters: "
                    "'custom_objects_map' and 'custom_objects_directory' are mandatory."
                )
        else:
            if custom_objects_map is None or custom_objects_directory is None:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "At least 'modules_map' or both custom objects parameters: 'custom_objects_map' and "
                    "'custom_objects_directory' are mandatory for auto logging as the class must be located in a "
                    "custom object python file or an installed module. Without one of them the model will not be able "
                    "to be saved and logged"
                )
    elif model_path is None:
        raise mlrun.errors.MLRunInvalidArgumentError(
            "To evaluate the model and log it, the store path of the model ('model_path' parameter) must be provided."
        )

    # Set the kwargs dictionaries defaults:
    mlrun_callback_kwargs = (
        {} if mlrun_callback_kwargs is None else mlrun_callback_kwargs
    )
    tensorboard_callback_kwargs = (
        {} if tensorboard_callback_kwargs is None else tensorboard_callback_kwargs
    )

    # Add the additional parameters to tensorboard's callback kwargs dictionary:
    tensorboard_callback_kwargs["tensorboard_directory"] = tensorboard_directory

    # Add the additional parameters to MLRun's callback kwargs dictionary:
    mlrun_callback_kwargs["model_name"] = model_name
    mlrun_callback_kwargs["model_path"] = model_path
    mlrun_callback_kwargs["modules_map"] = modules_map
    mlrun_callback_kwargs["custom_objects_map"] = custom_objects_map
    mlrun_callback_kwargs["custom_objects_directory"] = custom_objects_directory

    return mlrun_callback_kwargs, tensorboard_callback_kwargs
