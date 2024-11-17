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

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import mlrun

from .callbacks import Callback
from .callbacks_handler import CallbacksHandler
from .mlrun_interface import PyTorchMLRunInterface
from .model_handler import PyTorchModelHandler
from .model_server import PyTorchModelServer
from .utils import PyTorchTypes, PyTorchUtils


def train(
    model: Module,
    training_set: DataLoader,
    loss_function: Module,
    optimizer: Optimizer,
    validation_set: DataLoader = None,
    metric_functions: Optional[list[PyTorchTypes.MetricFunctionType]] = None,
    scheduler=None,
    scheduler_step_frequency: Union[int, float, str] = "epoch",
    epochs: int = 1,
    training_iterations: Optional[int] = None,
    validation_iterations: Optional[int] = None,
    callbacks_list: Optional[list[Callback]] = None,
    use_cuda: bool = True,
    use_horovod: Optional[bool] = None,
    auto_log: bool = True,
    model_name: Optional[str] = None,
    modules_map: Optional[Union[dict[str, Union[None, str, list[str]]], str]] = None,
    custom_objects_map: Optional[Union[dict[str, Union[str, list[str]]], str]] = None,
    custom_objects_directory: Optional[str] = None,
    tensorboard_directory: Optional[str] = None,
    mlrun_callback_kwargs: Optional[dict[str, Any]] = None,
    tensorboard_callback_kwargs: Optional[dict[str, Any]] = None,
    context: mlrun.MLClientCtx = None,
) -> PyTorchModelHandler:
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
                                        float percentage (0.0 < x < 1.0) for per x / iterations to step. Default:
                                        'epoch'.
    :param epochs:                      Amount of epochs to perform. Default: a single epoch.
    :param training_iterations:         Amount of iterations (batches) to perform on each epoch's training. If 'None'
                                        the entire training set will be used.
    :param validation_iterations:       Amount of iterations (batches) to perform on each epoch's validation. If 'None'
                                        the entire validation set will be used.
    :param callbacks_list:              The callbacks to use on this run.
    :param use_cuda:                    Whether or not to use cuda. Only relevant if cuda is available. Default:
                                        True.
    :param use_horovod:                 Whether or not to use horovod - a distributed training framework. Default:
                                        False.
    :param auto_log:                    Whether or not to apply auto-logging (to both MLRun and Tensorboard). Default:
                                        True. IF True, the custom objects are not optional.
    :param model_name:                  The model name to use for storing the model artifact. If not given, the model's
                                        class name will be used.
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

    :return: A model handler with the provided model and parameters.

    :raise ValueError: If 'auto_log' is set to True and one all of the custom objects or modules parameters given is
                       None.
    """
    # Get the context if not given:
    context = (
        context
        if context is not None
        else mlrun.get_or_create_ctx(PyTorchMLRunInterface.DEFAULT_CONTEXT_NAME)
    )

    # Create the model handler:
    handler = PyTorchModelHandler(
        model_name=model_name,
        model=model,
        modules_map=modules_map,
        custom_objects_map=custom_objects_map,
        custom_objects_directory=custom_objects_directory,
        context=context,
    )

    # Initialize the interface:
    interface = PyTorchMLRunInterface(model=model, context=context)

    # Add auto logging:
    if auto_log:
        # Parse the custom objects and the kwargs:
        mlrun_callback_kwargs, tensorboard_callback_kwargs = _parse_callbacks_kwargs(
            handler=handler,
            tensorboard_directory=tensorboard_directory,
            mlrun_callback_kwargs=mlrun_callback_kwargs,
            tensorboard_callback_kwargs=tensorboard_callback_kwargs,
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

    return handler


def evaluate(
    model_path: str,
    dataset: DataLoader,
    model: Module = None,
    loss_function: Module = None,
    metric_functions: Optional[list[PyTorchTypes.MetricFunctionType]] = None,
    iterations: Optional[int] = None,
    callbacks_list: Optional[list[Callback]] = None,
    use_cuda: bool = True,
    use_horovod: bool = False,
    auto_log: bool = True,
    model_name: Optional[str] = None,
    modules_map: Optional[Union[dict[str, Union[None, str, list[str]]], str]] = None,
    custom_objects_map: Optional[Union[dict[str, Union[str, list[str]]], str]] = None,
    custom_objects_directory: Optional[str] = None,
    mlrun_callback_kwargs: Optional[dict[str, Any]] = None,
    context: mlrun.MLClientCtx = None,
) -> tuple[PyTorchModelHandler, list[PyTorchTypes.MetricValueType]]:
    """
    Use MLRun's PyTorch interface to evaluate the model with the given parameters. For more information and further
    options regarding the auto logging, see 'PyTorchMLRunInterface' documentation. Notice for auto-logging: In order to
    log the model to MLRun, its class (torch.Module) must be in the custom objects map or the modules map.

    :param model_path:               The model's store object path. Mandatory for evaluation (to know which model to
                                     update).
    :param dataset:                  A data loader for the validation process.
    :param model:                    The model to evaluate. IF None, the model will be loaded from the given store model
                                     path.
    :param loss_function:            The loss function to use during training.
    :param metric_functions:         The metrics to use on training and validation.
    :param iterations:               Amount of iterations (batches) to perform on the dataset. If 'None' the entire
                                     dataset will be used.
    :param callbacks_list:           The callbacks to use on this run.
    :param use_cuda:                 Whether or not to use cuda. Only relevant if cuda is available. Default: True.
    :param use_horovod:              Whether or not to use horovod - a distributed training framework. Default:
                                     False.
    :param auto_log:                 Whether or not to apply auto-logging to MLRun. Default: True.
    :param model_name:               The model name to use for storing the model artifact. If not given, the model's
                                     class name will be used.
    :param modules_map:              A dictionary of all the modules required for loading the model. Each key is a path
                                     to a module and its value is the object name to import from it. All the modules
                                     will be imported globally. If multiple objects needed to be imported from the same
                                     module a list can be given. The map can be passed as a path to a json file as well.
                                     For example:

                                     .. code-block:: python

                                         {
                                             "module1": None,  # import module1
                                             "module2": ["func1", "func2"],  # from module2 import func1, func2
                                             "module3.sub_module": "func3",  # from module3.sub_module import func3
                                         }

                                     If the model path given is of a store object, the modules map will be read from
                                     the logged modules map artifact of the model.
    :param custom_objects_map:       A dictionary of all the custom objects required for loading the model. Each key is
                                     a path to a python file and its value is the custom object name to import from it.
                                     If multiple objects needed to be imported from the same py file a list can be
                                     given. The map can be passed as a path to a json file as well. For example:

                                     .. code-block:: python

                                         {
                                             "/.../custom_optimizer.py": "optimizer",
                                             "/.../custom_layers.py": ["layer1", "layer2"],
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

    :return: A tuple of:
             [0] = Initialized model handler with the evaluated model.
             [1] = The evaluation metrics results list.
    """
    # Get the context if not given:
    context = (
        context
        if context is not None
        else mlrun.get_or_create_ctx(PyTorchMLRunInterface.DEFAULT_CONTEXT_NAME)
    )

    # Create the model handler:
    handler = PyTorchModelHandler(
        model_path=model_path,
        model_name=model_name,
        model=model,
        modules_map=modules_map,
        custom_objects_map=custom_objects_map,
        custom_objects_directory=custom_objects_directory,
        context=context,
    )

    # Check if the model is needed to be loaded:
    if model is None:
        handler.load()

    # Initialize the interface:
    interface = PyTorchMLRunInterface(model=handler.model, context=context)

    # Add auto logging:
    if auto_log:
        # Parse the custom objects and the kwargs:
        mlrun_callback_kwargs, _ = _parse_callbacks_kwargs(
            handler=handler,
            tensorboard_directory=None,
            mlrun_callback_kwargs=mlrun_callback_kwargs,
            tensorboard_callback_kwargs=None,
        )
        # Add the logging callbacks with the provided parameters:
        interface.add_auto_logging_callbacks(
            mlrun_callback_kwargs=mlrun_callback_kwargs, add_tensorboard_logger=False
        )

    # Evaluate:
    return (
        handler,
        interface.evaluate(
            dataset=dataset,
            loss_function=loss_function,
            metric_functions=metric_functions,
            iterations=iterations,
            callbacks=callbacks_list,
            use_cuda=use_cuda,
            use_horovod=use_horovod,
        ),
    )


def _parse_callbacks_kwargs(
    handler: PyTorchModelHandler,
    tensorboard_directory: Union[str, None],
    mlrun_callback_kwargs: Union[dict[str, Any], None],
    tensorboard_callback_kwargs: Union[dict[str, Any], None],
) -> tuple[dict, dict]:
    """
    Parse the given parameters into the MLRun and Tensorboard callbacks kwargs.

    :param handler:                     An initialized model handler to insert to the mlrun callback kwargs.
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

    :return: Tuple of the callbacks kwargs:
             [0] = MLRun's kwargs.
             [1] = Tensorboard kwargs.

    :raise MLRunInvalidArgumentError: In case of a training session: if one or more of the custom objects parameters
                                      were not given. In case of an evaluation session, if the model path was not given.
    """
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
    mlrun_callback_kwargs["model_handler"] = handler

    return mlrun_callback_kwargs, tensorboard_callback_kwargs
