import importlib
import os
from abc import ABC
from typing import Any, Dict, Generator, Iterator, List, Sequence, Tuple, Union

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

from .._common import MLRunInterface
from .callbacks import MLRunLoggingCallback, TensorboardLoggingCallback


class TFKerasMLRunInterface(MLRunInterface, ABC):
    """
    MLRun model is for enabling additional features supported by MLRun in keras. With MLRun model one can apply horovod
    and use auto logging with ease.
    """

    # Typing hints for 'x' and 'y' parameters of 'fit' and 'evaluate':
    Dataset = Union[
        tf.data.Dataset, tf.keras.utils.Sequence, Generator, Iterator, Sequence
    ]
    GroundTruths = Union[Generator, Iterator, Sequence, None]

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-tf-keras"

    # Properties attributes to be inserted so the keras mlrun interface will be fully enabled:
    _PROPERTIES = {
        # Auto enabled callbacks list:
        "_auto_log_callbacks": [],
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
        "_pre_evaluate",
    ]  # type: List[str]

    @classmethod
    def add_interface(cls, model: keras.Model):
        """
        Wrap the given model with MLRun model features, providing it with MLRun model attributes including its
        parameters and methods.

        :param model: The model to wrap.
        """
        super(TFKerasMLRunInterface, cls).add_interface(model=model)

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
                return compile_method(*args, **kwargs)

            return wrapper

        setattr(model, "compile", compile_wrapper(model.compile))

        # Wrap the fit method:
        def fit_wrapper(fit_method, evaluate_method):
            def wrapper(*args, **kwargs):
                # Unwrap the evaluation method as fit will use it:
                setattr(model, "evaluate", evaluate_method)
                # Setup the callbacks list:
                if "callbacks" not in kwargs or kwargs["callbacks"] is None:
                    kwargs["callbacks"] = []
                # Add auto log callbacks if they were added:
                kwargs["callbacks"] = kwargs["callbacks"] + model._auto_log_callbacks
                # Setup default values if needed:
                kwargs["verbose"] = kwargs.get("verbose", 1)
                kwargs["steps_per_epoch"] = kwargs.get("steps_per_epoch", None)
                kwargs["validation_steps"] = kwargs.get("validation_steps", None)
                kwargs["validation_data"] = kwargs.get("validation_data", None)
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
                result = fit_method(*args, **kwargs)
                # Wrap the evaluation method again:
                setattr(model, "evaluate", evaluate_wrapper(evaluate_method))
                return result

            return wrapper

        setattr(model, "fit", fit_wrapper(model.fit, model.evaluate))

        # Wrap the evaluate method:
        def evaluate_wrapper(evaluate_method):
            def wrapper(*args, **kwargs):
                # Setup the callbacks list:
                if "callbacks" not in kwargs or kwargs["callbacks"] is None:
                    kwargs["callbacks"] = []
                # Add auto log callbacks if they were added:
                kwargs["callbacks"] = kwargs["callbacks"] + model._auto_log_callbacks
                # Setup default values if needed:
                kwargs["steps"] = kwargs.get("steps", None)
                # Call the pre evaluate method:
                (callbacks, steps) = model._pre_evaluate(
                    callbacks=kwargs["callbacks"], steps=kwargs["steps"],
                )
                # Assign parameters:
                kwargs["callbacks"] = callbacks
                kwargs["steps"] = steps
                # Call the original fit method:
                return evaluate_method(*args, **kwargs)

            return wrapper

        setattr(model, "evaluate", evaluate_wrapper(model.evaluate))

    def auto_log(
        self,
        context: mlrun.MLClientCtx = None,
        add_mlrun_logger: bool = True,
        mlrun_callback_kwargs: Dict[str, Any] = None,
        add_tensorboard_logger: bool = True,
        tensorboard_callback_kwargs: Dict[str, Any] = None,
    ):
        """
        Initialize the defaulted logging callbacks by MLRun. Given the context, the method will setup a list of
        callbacks with the most common settings for logging a training session in tensorflow.keras. For further
        information regarding the logging callbacks, see 'mlrun.frameworks.tf_keras.callbacks.MLRunLoggingCallback' and
        'mlrun.frameworks.tf_keras.callbacks.TensorboardLoggingCallback'.

        :param context:                     The MLRun context to log with.
        :param add_mlrun_logger:            Whether or not to add the 'MLRunLoggingCallback'. Defaulted to True.
        :param mlrun_callback_kwargs:       Key word arguments for the MLRun callback. For further information see the
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
            context = mlrun.get_or_create_ctx(
                TFKerasMLRunInterface.DEFAULT_CONTEXT_NAME
            )

        # Set the dictionaries defaults:
        mlrun_callback_kwargs = (
            {} if mlrun_callback_kwargs is None else mlrun_callback_kwargs
        )
        tensorboard_callback_kwargs = (
            {} if tensorboard_callback_kwargs is None else tensorboard_callback_kwargs
        )

        # Add the loggers:
        if add_mlrun_logger:
            # Add the MLRun logging callback:
            self._auto_log_callbacks.append(
                MLRunLoggingCallback(
                    context=context, auto_log=True, **mlrun_callback_kwargs
                )
            )
        if add_tensorboard_logger:
            # Add the Tensorboard logging callback:
            self._auto_log_callbacks.append(
                TensorboardLoggingCallback(
                    context=context, auto_log=True, **tensorboard_callback_kwargs
                )
            )

    # TODO: Add horovod callbacks configurations. If not set (None), use the defaults.
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

        :raise MLRunInvalidArgumentError: In case the optimizer was passed as a string.
        """
        # Check if needed to run with horovod:
        if self._hvd is None:
            return optimizer, None

        # Validate the optimizer input:
        if isinstance(optimizer, str):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "When using horovod, the compile method is expecting an initialized optimizer instance and not a "
                "string."
            )

        # Setup the device to run on GPU if available:
        if (
            tf.config.experimental.list_physical_devices("GPU")
            and os.environ.get("CUDA_VISIBLE_DEVICES", "None") != "-1"
        ):
            # Pin each GPU to a single process:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                tf.config.experimental.set_visible_devices(
                    gpus[self._hvd.local_rank()], "GPU"
                )
                print(
                    f"Horovod worker #{self._hvd.rank()} is using GPU #{self._hvd.local_rank()}"
                )
        else:
            # No GPUs were found, or 'use_cuda' was false:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            print(f"Horovod worker #{self._hvd.rank()} is using CPU")

        # Adjust learning rate based on the number of GPUs:
        optimizer.lr = optimizer.lr * self._hvd.size()

        # Wrap the optimizer in horovod's distributed optimizer: 'hvd.DistributedOptimizer'.
        optimizer = self._hvd.DistributedOptimizer(optimizer)

        # Compile the model with `experimental_run_tf_function=False` to ensure Tensorflow uses the distributed
        # optimizer to compute the gradients:
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

        :raise MLRunInvalidArgumentError: If horovod is being used but the 'steps_per_epoch' parameter were not given.
        """
        # Check if needed to run with horovod:
        if self._hvd is None:
            return callbacks, verbose, steps_per_epoch, validation_steps

        # Validate steps provided for horovod:
        if steps_per_epoch is None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "When using Horovod, the parameter 'steps_per_epoch' must be provided to the 'fit' method in order to "
                "split the steps between the workers."
            )

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

        # Adjust the number of steps per epoch based on the number of workers:
        steps_per_epoch = steps_per_epoch // self._hvd.size()
        if validation_steps is not None:
            validation_steps = validation_steps // self._hvd.size()

        return callbacks, verbose, steps_per_epoch, validation_steps

    def _pre_evaluate(
        self, callbacks: List[Callback], steps: Union[int, None],
    ) -> Tuple[List[Callback], Union[int, None]]:
        """
        Method to call before calling 'evaluate' to setup the run and inputs for using horovod.

        :param callbacks: Callbacks to use in the run. The callbacks will be split among the ranks so only certain
                          callbacks (mainly logging and checkpoints) will be in rank 0.
        :param steps:     Amount of evaluation steps to run in each epoch. The steps will be divided by the size of
                          ranks (horovod workers).

        :return: The updated parameters according to the used rank:
                 [0] = Callbacks list.
                 [1] = Steps.

        :raise MLRunInvalidArgumentError: If horovod is being used but the 'steps' parameter were not given.
        """
        # Remove the 'auto_log' callback 'TensorboardLoggingCallback' (only relevant for training):
        callbacks = [
            callback
            for callback in callbacks
            if not isinstance(callback, TensorboardLoggingCallback)
        ]

        # Check if needed to run with horovod:
        if self._hvd is None:
            return callbacks, steps

        # Validate steps provided for horovod:
        if steps is None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "When using Horovod, the parameter 'steps' must be provided to the 'evaluate' method in order to "
                "split the steps between the workers."
            )

        # Setup the callbacks:
        metric_average_callback = self._hvd.callbacks.MetricAverageCallback()
        metric_average_callback._supports_tf_logs = True
        horovod_callbacks = [
            self._hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            metric_average_callback,
        ]
        if self._hvd.rank() != 0:
            callbacks = [
                callback
                for callback in callbacks
                if type(callback).__name__ not in self._RANK_0_ONLY_CALLBACKS
            ]
        callbacks = horovod_callbacks + callbacks

        # Adjust the number of steps per epoch based on the number of workers:
        steps = steps // self._hvd.size()

        return callbacks, steps
