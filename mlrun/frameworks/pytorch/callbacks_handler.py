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
from typing import Dict, List, Tuple, Union

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .callbacks import Callback
from .utils import PyTorchTypes


class _CallbackInterface:
    """
    An interface of all the methods every callback must implement for dynamically calling the methods from the callbacks
    handler.
    """

    ON_SETUP = "on_setup"
    ON_RUN_BEGIN = "on_run_begin"
    ON_RUN_END = "on_run_end"
    ON_EPOCH_BEGIN = "on_epoch_begin"
    ON_EPOCH_END = "on_epoch_end"
    ON_TRAIN_BEGIN = "on_train_begin"
    ON_TRAIN_END = "on_train_end"
    ON_VALIDATION_BEGIN = "on_validation_begin"
    ON_VALIDATION_END = "on_validation_end"
    ON_TRAIN_BATCH_BEGIN = "on_train_batch_begin"
    ON_TRAIN_BATCH_END = "on_train_batch_end"
    ON_VALIDATION_BATCH_BEGIN = "on_validation_batch_begin"
    ON_VALIDATION_BATCH_END = "on_validation_batch_end"
    ON_INFERENCE_BEGIN = "on_inference_begin"
    ON_INFERENCE_END = "on_inference_end"
    ON_TRAIN_LOSS_BEGIN = "on_train_loss_begin"
    ON_TRAIN_LOSS_END = "on_train_loss_end"
    ON_VALIDATION_LOSS_BEGIN = "on_validation_loss_begin"
    ON_VALIDATION_LOSS_END = "on_validation_loss_end"
    ON_TRAIN_METRICS_BEGIN = "on_train_metrics_begin"
    ON_TRAIN_METRICS_END = "on_train_metrics_end"
    ON_VALIDATION_METRICS_BEGIN = "on_validation_metrics_begin"
    ON_VALIDATION_METRICS_END = "on_validation_metrics_end"
    ON_BACKWARD_BEGIN = "on_backward_begin"
    ON_BACKWARD_END = "on_backward_end"
    ON_OPTIMIZER_STEP_BEGIN = "on_optimizer_step_begin"
    ON_OPTIMIZER_STEP_END = "on_optimizer_step_end"
    ON_SCHEDULER_STEP_BEGIN = "on_scheduler_step_begin"
    ON_SCHEDULER_STEP_END = "on_scheduler_step_end"
    ON_CALL_CHECK = "on_call_check"


class CallbacksHandler:
    """
    A class for handling multiple callbacks during a run.
    """

    def __init__(self, callbacks: List[Union[Callback, Tuple[str, Callback]]]):
        """
        Initialize the callbacks handler with the given callbacks he will handle. The callbacks can be passed as their
        initialized instances or as a tuple where [0] is a name that will be attached to him and [1] will be the
        initialized callback instance. Notice that if a callback was given without a name, his name to refer will be
        his class name.

        :param callbacks: The callbacks to handle.

        :raise KeyError: In case of duplicating callbacks names.
        """
        # Initialize the callbacks dictionary:
        self._callbacks = {}  # type: Dict[str, Callback]

        # Add the given callback to the dictionary:
        for callback in callbacks:
            if isinstance(callback, tuple):
                # Add by given name:
                if callback[0] in self._callbacks:
                    raise KeyError(
                        f"A callback with the name '{callback[0]}' is already exist in the callbacks handler."
                    )
                self._callbacks[callback[0]] = callback[1]
            else:
                # Add by class name:
                if callback.__class__.__name__ in self._callbacks:
                    raise KeyError(
                        f"A callback with the name '{callback.__class__.__name__}' is already exist in the callbacks "
                        f"handler."
                    )
                self._callbacks[callback.__class__.__name__] = callback

    @property
    def callbacks(self) -> Dict[str, Callback]:
        """
        Get the callbacks dictionary handled by this handler.

        :return: The callbacks dictionary.
        """
        return self._callbacks

    def on_setup(
        self,
        model: Module,
        training_set: DataLoader,
        validation_set: DataLoader,
        loss_function: Module,
        optimizer: Optimizer,
        metric_functions: List[PyTorchTypes.MetricFunctionType],
        scheduler,
        callbacks: List[str] = None,
    ) -> bool:
        """
        Call the 'on_setup' method of every callback in the callbacks list. If the list is 'None' (not given), all
        callbacks will be called.

        :param model:            The model to pass to the method.
        :param training_set:     The training set to pass to the method.
        :param validation_set:   The validation set to pass to the method.
        :param loss_function:    The loss function to pass to the method.
        :param optimizer:        The optimizer to pass to the method.
        :param metric_functions: The metric functions to pass to the method.
        :param scheduler:        The scheduler to pass to the method.
        :param callbacks:        The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_SETUP,
            callbacks=self._parse_names(names=callbacks),
            model=model,
            training_set=training_set,
            validation_set=validation_set,
            loss_function=loss_function,
            optimizer=optimizer,
            metric_functions=metric_functions,
            scheduler=scheduler,
        )

    def on_run_begin(self, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_run_begin' method of every callback in the callbacks list. If the list is 'None' (not given), all
        callbacks will be called.

        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_RUN_BEGIN,
            callbacks=self._parse_names(names=callbacks),
        )

    def on_run_end(self, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_run_end' method of every callback in the callbacks list. If the list is 'None' (not given), all
        callbacks will be called.

        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_RUN_END,
            callbacks=self._parse_names(names=callbacks),
        )

    def on_epoch_begin(self, epoch: int, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_epoch_begin' method of every callback in the callbacks list. If the list is 'None' (not given), all
        callbacks will be called.

        :param epoch:     The current epoch iteration of when this method is called.
        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_EPOCH_BEGIN,
            callbacks=self._parse_names(names=callbacks),
            epoch=epoch,
        )

    def on_epoch_end(self, epoch: int, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_epoch_end' method of every callback in the callbacks list. If the list is 'None' (not given), all
        callbacks will be called.

        :param epoch:     The current epoch iteration of when this method is called.
        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_EPOCH_END,
            callbacks=self._parse_names(names=callbacks),
            epoch=epoch,
        )

    def on_train_begin(self, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_train_begin' method of every callback in the callbacks list. If the list is 'None' (not given), all
        callbacks will be called.

        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_TRAIN_BEGIN,
            callbacks=self._parse_names(names=callbacks),
        )

    def on_train_end(self, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_train_end' method of every callback in the callbacks list. If the list is 'None' (not given), all
        callbacks will be called.

        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_TRAIN_END,
            callbacks=self._parse_names(names=callbacks),
        )

    def on_validation_begin(self, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_validation_begin' method of every callback in the callbacks list. If the list is 'None'
        (not given), all callbacks will be called.

        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_VALIDATION_BEGIN,
            callbacks=self._parse_names(names=callbacks),
        )

    def on_validation_end(
        self,
        loss_value: PyTorchTypes.MetricValueType,
        metric_values: List[float],
        callbacks: List[str] = None,
    ) -> bool:
        """
        Call the 'on_validation_end' method of every callback in the callbacks list. If the list is 'None' (not given),
        all callbacks will be called.

        :param loss_value:    The loss summary of this validation.
        :param metric_values: The metrics summaries of this validation.
        :param callbacks:     The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_VALIDATION_END,
            callbacks=self._parse_names(names=callbacks),
            loss_value=loss_value,
            metric_values=metric_values,
        )

    def on_train_batch_begin(
        self, batch: int, x, y_true: Tensor, callbacks: List[str] = None
    ) -> bool:
        """
        Call the 'on_train_batch_begin' method of every callback in the callbacks list. If the list is 'None'
        (not given), all callbacks will be called.

        :param batch:     The current batch iteration of when this method is called.
        :param x:         The input part of the current batch.
        :param y_true:    The true value part of the current batch.
        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_TRAIN_BATCH_BEGIN,
            callbacks=self._parse_names(names=callbacks),
            batch=batch,
            x=x,
            y_true=y_true,
        )

    def on_train_batch_end(
        self,
        batch: int,
        x,
        y_pred: Tensor,
        y_true: Tensor,
        callbacks: List[str] = None,
    ) -> bool:
        """
        Call the 'on_train_batch_end' method of every callback in the callbacks list. If the list is 'None' (not given),
        all callbacks will be called.

        :param batch:     The current batch iteration of when this method is called.
        :param x:         The input part of the current batch.
        :param y_pred:    The prediction (output) of the model for this batch's input ('x').
        :param y_true:    The true value part of the current batch.
        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_TRAIN_BATCH_END,
            callbacks=self._parse_names(names=callbacks),
            batch=batch,
            x=x,
            y_pred=y_pred,
            y_true=y_true,
        )

    def on_validation_batch_begin(
        self, batch: int, x, y_true: Tensor, callbacks: List[str] = None
    ) -> bool:
        """
        Call the 'on_validation_batch_begin' method of every callback in the callbacks list. If the list is 'None'
        (not given), all callbacks will be called.

        :param batch:     The current batch iteration of when this method is called.
        :param x:         The input part of the current batch.
        :param y_true:    The true value part of the current batch.
        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_VALIDATION_BATCH_BEGIN,
            callbacks=self._parse_names(names=callbacks),
            batch=batch,
            x=x,
            y_true=y_true,
        )

    def on_validation_batch_end(
        self,
        batch: int,
        x,
        y_pred: Tensor,
        y_true: Tensor,
        callbacks: List[str] = None,
    ) -> bool:
        """
        Call the 'on_validation_batch_end' method of every callback in the callbacks list. If the list is 'None'
        (not given), all callbacks will be called.

        :param batch:     The current batch iteration of when this method is called.
        :param x:         The input part of the current batch.
        :param y_pred:    The prediction (output) of the model for this batch's input ('x').
        :param y_true:    The true value part of the current batch.
        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_VALIDATION_BATCH_END,
            callbacks=self._parse_names(names=callbacks),
            batch=batch,
            x=x,
            y_pred=y_pred,
            y_true=y_true,
        )

    def on_inference_begin(
        self,
        x,
        callbacks: List[str] = None,
    ) -> bool:
        """
        Call the 'on_inference_begin' method of every callback in the callbacks list. If the list is 'None' (not given),
        all callbacks will be called.

        :param x:         The input part of the current batch.
        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_INFERENCE_BEGIN,
            callbacks=self._parse_names(names=callbacks),
            x=x,
        )

    def on_inference_end(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        callbacks: List[str] = None,
    ) -> bool:
        """
        Call the 'on_inference_end' method of every callback in the callbacks list. If the list is 'None' (not given),
        all callbacks will be called.

        :param y_pred:    The prediction (output) of the model for this batch's input ('x').
        :param y_true:    The true value part of the current batch.
        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_INFERENCE_END,
            callbacks=self._parse_names(names=callbacks),
            y_pred=y_pred,
            y_true=y_true,
        )

    def on_train_loss_begin(self, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_train_loss_begin' method of every callback in the callbacks list. If the list is 'None'
        (not given), all callbacks will be called.

        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_TRAIN_LOSS_BEGIN,
            callbacks=self._parse_names(names=callbacks),
        )

    def on_train_loss_end(
        self, loss_value: PyTorchTypes.MetricValueType, callbacks: List[str] = None
    ) -> bool:
        """
        Call the 'on_train_loss_end' method of every callback in the callbacks list. If the list is 'None' (not given),
        all callbacks will be called.

        :param loss_value: The recent loss value calculated during training.
        :param callbacks:  The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_TRAIN_LOSS_END,
            callbacks=self._parse_names(names=callbacks),
            loss_value=loss_value,
        )

    def on_validation_loss_begin(self, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_validation_loss_begin' method of every callback in the callbacks list. If the list is 'None'
        (not given), all callbacks will be called.

        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_VALIDATION_LOSS_BEGIN,
            callbacks=self._parse_names(names=callbacks),
        )

    def on_validation_loss_end(
        self, loss_value: PyTorchTypes.MetricValueType, callbacks: List[str] = None
    ) -> bool:
        """
        Call the 'on_validation_loss_end' method of every callback in the callbacks list. If the list is 'None'
        (not given), all callbacks will be called.

        :param loss_value: The recent loss value calculated during validation.
        :param callbacks:  The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_VALIDATION_LOSS_END,
            callbacks=self._parse_names(names=callbacks),
            loss_value=loss_value,
        )

    def on_train_metrics_begin(self, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_train_metrics_begin' method of every callback in the callbacks list. If the list is 'None'
        (not given), all callbacks will be called.

        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_TRAIN_METRICS_BEGIN,
            callbacks=self._parse_names(names=callbacks),
        )

    def on_train_metrics_end(
        self,
        metric_values: List[PyTorchTypes.MetricValueType],
        callbacks: List[str] = None,
    ) -> bool:
        """
        Call the 'on_train_metrics_end' method of every callback in the callbacks list. If the list is 'None'
        (not given), all callbacks will be called.

        :param metric_values: The recent metric values calculated during training.
        :param callbacks:     The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_TRAIN_METRICS_END,
            callbacks=self._parse_names(names=callbacks),
            metric_values=metric_values,
        )

    def on_validation_metrics_begin(self, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_validation_metrics_begin' method of every callback in the callbacks list. If the list is 'None'
        (not given), all callbacks will be called.

        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_VALIDATION_METRICS_BEGIN,
            callbacks=self._parse_names(names=callbacks),
        )

    def on_validation_metrics_end(
        self,
        metric_values: List[PyTorchTypes.MetricValueType],
        callbacks: List[str] = None,
    ) -> bool:
        """
        Call the 'on_validation_metrics_end' method of every callback in the callbacks list. If the list is 'None'
        (not given), all callbacks will be called.

        :param metric_values: The recent metric values calculated during validation.
        :param callbacks:     The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_VALIDATION_METRICS_END,
            callbacks=self._parse_names(names=callbacks),
            metric_values=metric_values,
        )

    def on_backward_begin(self, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_backward_begin' method of every callback in the callbacks list. If the list is 'None' (not given),
        all callbacks will be called.

        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_BACKWARD_BEGIN,
            callbacks=self._parse_names(names=callbacks),
        )

    def on_backward_end(self, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_backward_end' method of every callback in the callbacks list. If the list is 'None' (not given),
        all callbacks will be called.

        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_BACKWARD_END,
            callbacks=self._parse_names(names=callbacks),
        )

    def on_optimizer_step_begin(self, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_optimizer_step_begin' method of every callback in the callbacks list. If the list is 'None'
        (not given), all callbacks will be called.

        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_OPTIMIZER_STEP_BEGIN,
            callbacks=self._parse_names(names=callbacks),
        )

    def on_optimizer_step_end(self, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_optimizer_step_end' method of every callback in the callbacks list. If the list is 'None'
        (not given), all callbacks will be called.

        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_OPTIMIZER_STEP_END,
            callbacks=self._parse_names(names=callbacks),
        )

    def on_scheduler_step_begin(self, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_scheduler_step_begin' method of every callback in the callbacks list. If the list is 'None'
        (not given), all callbacks will be called.

        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_SCHEDULER_STEP_BEGIN,
            callbacks=self._parse_names(names=callbacks),
        )

    def on_scheduler_step_end(self, callbacks: List[str] = None) -> bool:
        """
        Call the 'on_scheduler_step_end' method of every callback in the callbacks list. If the list is 'None'
        (not given), all callbacks will be called.

        :param callbacks: The callbacks names to use. If 'None', all of the callbacks will be used.

        :return: True if all of the callbacks called returned True and False if not.
        """
        return self._run_callbacks(
            method_name=_CallbackInterface.ON_SCHEDULER_STEP_END,
            callbacks=self._parse_names(names=callbacks),
        )

    def _parse_names(self, names: Union[List[str], None]) -> List[str]:
        """
        Parse the given callbacks names. If they are not 'None' then the names will be returned as they are, otherwise
        all of the callbacks handled by this handler will be returned (the default behavior of when there were no names
        given to one of the handler's methods).

        :param names: A list of names to parse, can be 'None'.

        :return: The given names if they were not 'None', otherwise the names of all the callbacks in this handler.
        """
        if names:
            return names
        return list(self._callbacks.keys())

    def _run_callbacks(
        self, method_name: str, callbacks: List[str], *args, **kwargs
    ) -> bool:
        """
        Run the given method from the 'CallbackInterface' on all the specified callbacks with the given arguments.

        :param method_name: The name of the method to run. Should be given from the 'CallbackInterface'.
        :param callbacks:   List of all the callbacks names to run the method.

        :return: True if all the callbacks called returned True and False if not.
        """
        all_result = True
        for callback in callbacks:
            if self._callbacks[callback].on_call_check():
                method = getattr(self._callbacks[callback], method_name)
                result = method(*args, **kwargs)
                if result:
                    all_result &= result
        return all_result
