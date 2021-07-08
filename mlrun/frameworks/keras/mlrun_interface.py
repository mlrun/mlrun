import importlib
import os
from abc import ABC
from typing import Any, Dict, List, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    BaseLogger,
    Callback,
    CSVLogger,
    ModelCheckpoint,
    ProgbarLogger,
    TensorBoard,
)
from tensorflow.keras.optimizers import Optimizer

import mlrun
from mlrun.frameworks._common import MLRunInterface
from mlrun.frameworks.keras.callbacks import (
    MLRunLoggingCallback,
    TensorboardLoggingCallback,
)


class KerasMLRunInterface(MLRunInterface, keras.Model, ABC):
    """
    MLRun model is for enabling additional features supported by MLRun in keras. With MLRun model one can apply horovod
    and use auto logging with ease.
    """

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-keras"

    # Properties attributes to be inserted so the keras mlrun interface will be fully enabled:
    _PROPERTIES = {
        # Auto enabled callbacks list:
        "_callbacks": [],
        # Variable to hold the horovod module:
        "_hvd": None,
        # List of all the callbacks that should only be applied on rank 0 when using horovod:
        "_RANK_0_ONLY_CALLBACKS": [
            MLRunLoggingCallback.__name__,
            TensorboardLoggingCallback.__name__,
            ModelCheckpoint.__name__,
            TensorBoard.__name__,
            ProgbarLogger.__name__,
            CSVLogger.__name__,
            BaseLogger.__name__,
        ],
    }

    # Methods attributes to be inserted so the keras mlrun interface will be fully enabled:
    _METHODS = [
        "auto_log",
        "use_horovod",
        "note_rank_0_callback",
        "_pre_compile",
        "_pre_fit",
    ]  # type: List[str]

    @classmethod
    def add_interface(cls, model: keras.Model, *args, **kwargs):
        """
        Wrap the given model with MLRun model features, providing it with MLRun model attributes including its
        parameters and methods.

        :param model: The model to wrap.

        :return: The wrapped model.
        """
        super(KerasMLRunInterface, cls).add_interface(model=model)

        # Wrap the compile method:
        def compile_wrapper(compile_method):
            def wrapper(*args, **kwargs):
                # Call the pre compile method:
                (optimizer, experimental_run_tf_function) = model._pre_compile(
                    optimizer=kwargs["optimizer"]
                )
                # Assign parameters:
                kwargs["optimizer"] = optimizer
                if experimental_run_tf_function is not None:
                    kwargs[
                        "experimental_run_tf_function"
                    ] = experimental_run_tf_function
                # Call the original compile method:
                compile_method(*args, **kwargs)

            return wrapper

        setattr(model, "compile", compile_wrapper(model.compile))

        # Wrap the fit method:
        def fit_wrapper(fit_method):
            def wrapper(*args, **kwargs):
                # Setup the callbacks list:
                if "callbacks" not in kwargs or kwargs["callbacks"] is None:
                    kwargs["callbacks"] = []
                # Add auto log callbacks if they were added:
                kwargs["callbacks"] = kwargs["callbacks"] + model._callbacks
                # Setup default values if needed:
                if "verbose" not in kwargs:
                    kwargs["verbose"] = 1
                if "steps_per_epoch" not in kwargs:
                    kwargs["steps_per_epoch"] = None
                if "validation_steps" not in kwargs:
                    kwargs["validation_steps"] = None
                # Call the pre fit method:
                (
                    callbacks,
                    verbose,
                    steps_per_epoch,
                    validation_steps,
                ) = model._pre_fit(
                    callbacks=kwargs["callbacks"],
                    verbose=kwargs["verbose"],
                    steps_per_epoch=kwargs["steps_per_epoch"],
                    validation_steps=kwargs["validation_steps"],
                )
                # Assign parameters:
                kwargs["callbacks"] = callbacks
                kwargs["verbose"] = verbose
                kwargs["steps_per_epoch"] = steps_per_epoch
                kwargs["validation_steps"] = validation_steps
                # Call the original fit method:
                fit_method(*args, **kwargs)

            return wrapper

        setattr(model, "fit", fit_wrapper(model.fit))

    def auto_log(
        self,
        context: mlrun.MLClientCtx = None,
        add_mlrun_logger: bool = True,
        mlrun_callback__kwargs: Dict[str, Any] = None,
        add_tensorboard_logger: bool = True,
        tensorboard_callback_kwargs: Dict[str, Any] = None,
    ):
        """
        Initialize the defaulted logging callbacks by MLRun. Given the context, the method will setup a list of
        callbacks with the most common settings for logging a training session in tensorflow.keras. For further
        information regarding the logging callbacks, see 'mlrun.frameworks.keras.callbacks.MLRunLoggingCallback' and
        'mlrun.frameworks.keras.callbacks.TensorboardLoggingCallback'.

        :param context:                     The MLRun context to log with.
        :param add_mlrun_logger:            Whether or not to add the 'MLRunLoggingCallback'. Defaulted to True.
        :param mlrun_callback__kwargs:      Key word arguments for the MLRun callback. For further information see the
                                            documentation of the class 'MLRunLoggingCallback'. Note that both 'context'
                                            and 'auto_log' parameters are already given here.
        :param add_tensorboard_logger:      Whether or not to add the 'TensorboardLoggingCallback'. Defaulted to True.
        :param tensorboard_callback_kwargs: Key word arguments for the tensorboard callback. For further information see
                                            the documentation of the class 'TensorboardLoggingCallback'. Note that both
                                            'context' and 'auto_log' parameters are already given here.
        """
        # If horovod is being used, there is no need to add the logging callbacks to ranks other than 0:
        if self._hvd is not None and self._hvd.rank() != 0:
            return

        # Get default context in case it was not given:
        if context is None:
            context = mlrun.get_or_create_ctx(KerasMLRunInterface.DEFAULT_CONTEXT_NAME)

        # Set the dictionaries defaults:
        mlrun_callback__kwargs = (
            {} if mlrun_callback__kwargs is None else mlrun_callback__kwargs
        )
        tensorboard_callback_kwargs = (
            {} if tensorboard_callback_kwargs is None else tensorboard_callback_kwargs
        )

        # Add the loggers:
        if add_mlrun_logger:
            # Add the MLRun logging callback:
            self._callbacks.append(
                MLRunLoggingCallback(
                    context=context, auto_log=True, **mlrun_callback__kwargs
                )
            )
        if add_tensorboard_logger:
            # Add the Tensorboard logging callback:
            self._callbacks.append(
                TensorboardLoggingCallback(
                    context=context, auto_log=True, **tensorboard_callback_kwargs
                )
            )

    def use_horovod(self):
        """
        Setup the model or wrapped model to run with horovod.
        """
        # Import horovod:
        self._hvd = importlib.import_module("horovod.tensorflow.keras")

        # Initialize horovod:
        self._hvd.init()

    def note_rank_0_callback(self, callback_name: str):
        """
        Note an additional custom callback to be applied only on rank 0 when using horovod.

        :param callback_name: The name of the callback.
        """
        self._RANK_0_ONLY_CALLBACKS.append(callback_name)

    def _pre_compile(self, optimizer: Optimizer) -> Tuple[Optimizer, Union[bool, None]]:
        """
        Method to call before calling 'compile' to setup the run and inputs for using horovod.

        :param optimizer: The optimzier to compile. It will be wrapped in horovod's distributed optimizer:
                          'hvd.DistributedOptimizer'.

        :return: The updated parameters:
                 [0] = Wrapped optimizer.
                 [1] = The 'experimental_run_tf_function' parameter for 'compile' kwargs or 'None' if horovod should not
                       be used.

        :raise ValueError: In case the optimizer was passed as a string.
        """
        # Check if needed to run with horovod:
        if self._hvd is None:
            return optimizer, None

        # Validate the optimizer input:
        if isinstance(optimizer, str):
            raise ValueError(
                "When using horovod, the compile mehotd is expecting an initialized optimizer "
                "instance and not a string."
            )

        # Setup the device to run on GPU if available:
        if tf.config.experimental.list_physical_devices("GPU"):
            # Pin each GPU to a single process:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                tf.config.experimental.set_visible_devices(
                    gpus[self._hvd.local_rank()], "GPU"
                )
        else:
            # No GPUs were found, or 'use_cuda' was false:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # Adjust learning rate based on the number of GPUs:
        optimizer.lr = optimizer.lr * self._hvd.size()

        # Wrap the optimizer in horovod's distributed optimizer: 'hvd.DistributedOptimizer'.
        optimizer = self._hvd.DistributedOptimizer(optimizer)

        # Compile the model with `experimental_run_tf_function=False` to ensure Tensorflow uses the distributed
        # optimizer to compute gradients:
        experimental_run_tf_function = False

        return optimizer, experimental_run_tf_function

    def _pre_fit(
        self,
        callbacks: List[Callback],
        verbose: int,
        steps_per_epoch: Union[int, None],
        validation_steps: Union[int, None],
    ) -> Tuple[List[Callback], int, Union[int, None], Union[int, None]]:
        """
        Method to call before calling 'fit' to setup the run and inputs for using horovod.

        :param callbacks:        Callbacks to use in the run. The callbacks will be split among the ranks so only
                                 certain callbacks (mainly logging and checkpoints) will be in rank 0.
        :param verbose:          Whether or not to print the progress of training. If '1' or '2' only rank 0 will be
                                 applied with the verbose.
        :param steps_per_epoch:  Amount of training steps to run in each epoch. The steps will be divided by the size of
                                 ranks (horovod workers).
        :param validation_steps: Amount of validation steps to run in each epoch. The steps will be divided by the size
                                 of ranks (horovod workers).

        :return: The updated parameters according to the used rank:
                 [0] = Callbacks list.
                 [1] = Verbose
                 [2] = Steps per epoch or None if not given.
                 [3] = Validation steps or None if not given.
        """
        # Check if needed to run with horovod:
        if self._hvd is None:
            return callbacks, verbose, steps_per_epoch, validation_steps

        # Setup the callbacks:
        metric_average_callback = self._hvd.callbacks.MetricAverageCallback()
        metric_average_callback._supports_tf_logs = True
        horovod_callbacks = [
            self._hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            metric_average_callback,
            self._hvd.callbacks.LearningRateWarmupCallback(
                initial_lr=float(self.optimizer.lr)
            ),
        ]
        if self._hvd.rank() != 0:
            callbacks = [
                callback
                for callback in callbacks
                if type(callback).__name__ not in self._RANK_0_ONLY_CALLBACKS
            ]
        callbacks = horovod_callbacks + callbacks

        # Pick the verbose:
        if self._hvd.rank() != 0:
            verbose = 0

        # Adjust the number of steps per epoch based on the number of GPUs (if given):
        if steps_per_epoch is not None:
            steps_per_epoch = steps_per_epoch // self._hvd.size()
        if validation_steps is not None:
            validation_steps = validation_steps // self._hvd.size()

        return callbacks, verbose, steps_per_epoch, validation_steps
