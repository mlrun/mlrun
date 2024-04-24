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
#
import importlib
import os
from abc import ABC
from typing import Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.keras.callbacks import (
    BaseLogger,
    Callback,
    CSVLogger,
    ModelCheckpoint,
    ProgbarLogger,
    TensorBoard,
)

import mlrun

from .._common import MLRunInterface
from .callbacks import LoggingCallback
from .utils import TFKerasTypes


class TFKerasMLRunInterface(MLRunInterface, ABC):
    """
    Interface for adding MLRun features for tensorflow keras API.
    """

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-tf-keras"

    # Attributes to be inserted so the MLRun interface will be fully enabled.
    _PROPERTIES = {
        # Logging callbacks list:
        "_logging_callbacks": set(),  # type: Set[Callback]
        # Variable to hold the horovod module:
        "_hvd": None,  # type: ModuleType
        # List of all the callbacks that should only be applied on rank 0 when using horovod:
        "_RANK_0_ONLY_CALLBACKS": {  # type: Set[str]
            "LoggingCallback",
            "MLRunLoggingCallback",
            "TensorboardLoggingCallback",
            ModelCheckpoint.__name__,
            TensorBoard.__name__,
            ProgbarLogger.__name__,
            CSVLogger.__name__,
            BaseLogger.__name__,
        },
    }
    _METHODS = [
        "add_logging_callback",
        "use_horovod",
        "note_rank_0_callback",
        "_pre_compile",
        "_pre_fit",
        "_pre_evaluate",
    ]

    # Attributes to replace so the MLRun interface will be fully enabled.
    _REPLACED_METHODS = ["compile", "fit", "evaluate"]

    @classmethod
    def add_interface(
        cls,
        obj: keras.Model,
        restoration: TFKerasTypes.MLRunInterfaceRestorationType = None,
    ):
        """
        Enrich the object with this interface properties, methods and functions, so it will have this TensorFlow.Keras
        MLRun's features.

        :param obj:                     The object to enrich his interface.
        :param restoration: Restoration information tuple as returned from 'remove_interface' in order to
                                        add the interface in a certain state.
        """
        super().add_interface(obj=obj, restoration=restoration)

    def mlrun_compile(self, *args, **kwargs):
        """
        MLRun's tf.keras.Model.compile wrapper. It will setup the optimizer when using horovod. The optimizer must be
        passed in a keyword argument and when using horovod, it must be passed as an Optimizer instance, not a string.

        :raise MLRunInvalidArgumentError: In case the optimizer provided did not follow the instructions above.
        """
        # Validate the optimizer is passed via keyword:
        if "optimizer" not in kwargs:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "The optimizer must be passed as a keyword argument:\n"
                "model.compile(\n"
                "    optimizer=...\n"
                ")"
            )

        # Call the pre compile method:
        (optimizer, experimental_run_tf_function) = self._pre_compile(
            optimizer=kwargs["optimizer"]
        )

        # Assign parameters:
        kwargs["optimizer"] = optimizer
        if experimental_run_tf_function is not None:
            kwargs["experimental_run_tf_function"] = experimental_run_tf_function

        # Call the original compile method:
        return self.original_compile(*args, **kwargs)

    @classmethod
    def mlrun_fit(cls):
        """
        MLRun's tf.keras.Model.fit wrapper. It will setup the optimizer when using horovod. The optimizer must be
        passed in a keyword argument and when using horovod, it must be passed as an Optimizer instance, not a string.

        :raise MLRunInvalidArgumentError: In case the optimizer provided did not follow the instructions above.
        """

        def wrapper(self: keras.Model, *args, **kwargs):
            # Restore the evaluation method as fit will use it:
            cls._restore_attribute(obj=self, attribute_name="evaluate")

            # Setup the callbacks list:
            if "callbacks" not in kwargs or kwargs["callbacks"] is None:
                kwargs["callbacks"] = []

            # Add auto logging callbacks if they were added:
            kwargs["callbacks"] = kwargs["callbacks"] + list(self._logging_callbacks)

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
            ) = self._pre_fit(
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
            result = self.original_fit(*args, **kwargs)

            # Replace the evaluation method again:
            cls._replace_function(obj=self, function_name="evaluate")

            return result

        return wrapper

    def mlrun_evaluate(self, *args, **kwargs):
        """
        MLRun tf.keras.Model.evaluate wrapper. Will enable automatic logging if set.
        """
        # Setup the callbacks list:
        if "callbacks" not in kwargs or kwargs["callbacks"] is None:
            kwargs["callbacks"] = []

        # Add auto log callbacks if they were added:
        kwargs["callbacks"] = kwargs["callbacks"] + list(self._logging_callbacks)

        # Setup default values if needed:
        kwargs["steps"] = kwargs.get("steps", None)

        # Call the pre evaluate method:
        (callbacks, steps) = self._pre_evaluate(
            callbacks=kwargs["callbacks"],
            steps=kwargs["steps"],
        )

        # Assign parameters:
        kwargs["callbacks"] = callbacks
        kwargs["steps"] = steps

        # Call the original evaluation method:
        return self.original_evaluate(*args, **kwargs)

    def add_logging_callback(self, logging_callback: LoggingCallback):
        """
        Add the given logging callback to model's logging callbacks list. For further information regarding the logging
        callbacks, see 'mlrun.frameworks.tf_keras.callbacks.MLRunLoggingCallback' and
        'mlrun.frameworks.tf_keras.callbacks.TensorboardLoggingCallback'.

        :param logging_callback: The logging callback to add.
        """
        # If horovod is being used, there is no need to add the logging callbacks to ranks other than 0:
        if self._hvd is not None and self._hvd.rank() != 0:
            return

        # Add the logging callback:
        self._logging_callbacks.add(logging_callback)

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
        self._RANK_0_ONLY_CALLBACKS.add(callback_name)

    def _pre_compile(self, optimizer: Optimizer) -> tuple[Optimizer, Union[bool, None]]:
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
        callbacks: list[Callback],
        verbose: int,
        steps_per_epoch: Union[int, None],
        validation_steps: Union[int, None],
    ) -> tuple[list[Callback], int, Union[int, None], Union[int, None]]:
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
                initial_lr=float(
                    self.optimizer.lr
                    if hasattr(self.optimizer, "lr")
                    else self.optimizer.learning_rate
                )
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
        self,
        callbacks: list[Callback],
        steps: Union[int, None],
    ) -> tuple[list[Callback], Union[int, None]]:
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
            if type(callback).__name__ != "TensorboardLoggingCallback"
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
