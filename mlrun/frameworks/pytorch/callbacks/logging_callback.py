from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from torch import Tensor
from torch.nn import Module, Parameter

import mlrun
from mlrun.frameworks._common.loggers import Logger, TrackableType
from mlrun.frameworks.pytorch.callbacks.callback import (
    Callback,
    MetricFunctionType,
    MetricValueType,
)


class HyperparametersKeys:
    """
    For easy noting on which object to search for the hyperparameter to track with the logging callback in its
    initialization method.
    """

    MODEL = "model"
    TRAINING_SET = "training_set"
    VALIDATION_SET = "validation_set"
    LOSS_FUNCTION = "loss_function"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    CUSTOM = "custom"


class LoggingCallback(Callback):
    """
    Callback for collecting data during training / evaluation. All the collected data will be available in this callback
    post the training / evaluation process and can be accessed via the 'training_results', 'validation_results',
    'static_hyperparameters', 'dynamic_hyperparameters' and 'summaries' properties.
    """

    class _MetricType:
        """
        Metric can be of two types, a loss metric or accuracy metric.
        """

        LOSS = "Loss"
        ACCURACY = "Accuracy"

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        dynamic_hyperparameters: Dict[
            str, Tuple[str, Union[List[Union[str, int]], Callable[[], TrackableType]]]
        ] = None,
        static_hyperparameters: Dict[
            str, Union[TrackableType, Tuple[str, List[Union[str, int]]]]
        ] = None,
        auto_log: bool = False,
    ):
        """
        Initialize a logging callback with the given hyperparameters and logging configurations.

        :param context:                 MLRun context to automatically log its parameters if 'auto_log' is True.
        :param dynamic_hyperparameters: If needed to track a hyperparameter dynamically (sample it each epoch) it should
                                        be passed here. The parameter expects a dictionary where the keys are the
                                        hyperparameter chosen names and the values are tuples of object key and a list
                                        with the key chain. A key chain is a list of keys and indices to know how to
                                        access the needed hyperparameter. If the hyperparameter is not of accessible
                                        from any of the HyperparametersKeys, a custom callable method can be passed in
                                        the tuple instead of the key chain when providing the word
                                        HyperparametersKeys.CUSTOM. For example, to track the 'lr' attribute of
                                        an optimizer and a custom parameter, one should pass:
                                        {
                                            "learning rate": (HyperparametersKeys.OPTIMIZER, ["param_groups", 0, "lr"]),
                                            "custom parameter": (HyperparametersKeys.CUSTOM, get_custom_parameter)
                                        }
        :param static_hyperparameters:  If needed to track a hyperparameter one time per run it should be passed here.
                                        The parameter expects a dictionary where the keys are the
                                        hyperparameter chosen names and the values are the hyperparameter static value
                                        or a tuple of object key and a list with the key chain just like the dynamic
                                        hyperparameter. For example, to track the 'epochs' of an experiment run, one
                                        should pass:
                                        {
                                            "epochs": 7
                                        }
        :param auto_log:                Whether or not to enable auto logging, trying to track common static and dynamic
                                        hyperparameters.
        """
        super(LoggingCallback, self).__init__()

        # Store the configurations:
        self._dynamic_hyperparameters_keys = (
            dynamic_hyperparameters if dynamic_hyperparameters is not None else {}
        )
        self._static_hyperparameters_keys = (
            static_hyperparameters if static_hyperparameters is not None else {}
        )

        # Initialize the logger:
        self._logger = Logger(context=context)

        # Setup the logger flags:
        self._is_training = None  # type: bool
        self._auto_log = auto_log

    def get_training_results(self) -> Dict[str, List[List[float]]]:
        """
        Get the training results logged. The results will be stored in a dictionary where each key is the metric name
        and the value is a list of lists of values. The first list is by epoch and the second list is by iteration
        (batch).

        :return: The training results.
        """
        return self._logger.training_results

    def get_validation_results(self) -> Dict[str, List[List[float]]]:
        """
        Get the validation results logged. The results will be stored in a dictionary where each key is the metric name
        and the value is a list of lists of values. The first list is by epoch and the second list is by iteration
        (batch).

        :return: The validation results.
        """
        return self._logger.validation_results

    def get_static_hyperparameters(self) -> Dict[str, TrackableType]:
        """
        Get the static hyperparameters logged. The hyperparameters will be stored in a dictionary where each key is the
        hyperparameter name and the value is his logged value.

        :return: The static hyperparameters.
        """
        return self._logger.static_hyperparameters

    def get_dynamic_hyperparameters(self) -> Dict[str, List[TrackableType]]:
        """
        Get the dynamic hyperparameters logged. The hyperparameters will be stored in a dictionary where each key is the
        hyperparameter name and the value is a list of his logged values per epoch.

        :return: The dynamic hyperparameters.
        """
        return self._logger.dynamic_hyperparameters

    def get_summaries(self) -> Dict[str, List[float]]:
        """
        Get the validation summaries of the metrics results. The summaries will be stored in a dictionary where each key
        is the metric names and the value is a list of all the summary values per epoch.

        :return: The validation summaries.
        """
        return self._logger.validation_summaries

    def get_epochs(self) -> int:
        """
        Get the overall epochs this callback participated in.

        :return: The overall epochs this callback participated in.
        """
        return self._logger.epochs

    def get_train_iterations(self) -> int:
        """
        Get the overall train iterations this callback participated in.

        :return: The overall train iterations this callback participated in.
        """
        return self._logger.training_iterations

    def get_validation_iterations(self) -> int:
        """
        Get the overall validation iterations this callback participated in.

        :return: The overall validation iterations this callback participated in.
        """
        return self._logger.validation_iterations

    def on_horovod_check(self, rank: int) -> bool:
        """
        Check whether this callback is fitting to run by the given horovod rank (worker).

        :param rank: The horovod rank (worker) id.

        :return: True if the callback is ok to run on this rank and false if not.
        """
        return rank == 0

    def on_run_begin(self):
        """
        After the run begins, this method will be called to setup the results and hyperparameters dictionaries for
        logging, noting the metrics names and logging the initial hyperparameters values (epoch 0).
        """
        # Setup the results and summaries dictionaries:
        # # Loss:
        self._logger.log_metric(
            metric_name=self._get_metric_name(
                metric_function=self._objects[self._ObjectKeys.LOSS_FUNCTION],
            )
        )
        # # Metrics:
        for metric_function in self._objects[self._ObjectKeys.METRIC_FUNCTIONS]:
            metric_name = self._get_metric_name(metric_function=metric_function)
            self._logger.log_metric(metric_name=metric_name)

        # Setup the hyperparameters dictionaries:
        if self._auto_log:
            self._add_auto_hyperparameters()
        # # Static hyperparameters:
        for name, value in self._static_hyperparameters_keys.items():
            if isinstance(value, Tuple):
                # Its a parameter that needed to be extracted via key chain.
                self._logger.log_static_hyperparameter(
                    parameter_name=name,
                    value=self._get_hyperparameter(source=value[0], key_chain=value[1]),
                )
            else:
                # Its a custom hyperparameter.
                self._logger.log_static_hyperparameter(parameter_name=name, value=value)
        # # Dynamic hyperparameters:
        for name, (source, key_chain) in self._dynamic_hyperparameters_keys.items():
            self._logger.log_dynamic_hyperparameter(
                parameter_name=name,
                value=self._get_hyperparameter(source=source, key_chain=key_chain),
            )

    def on_epoch_begin(self, epoch: int):
        """
        After the given epoch begins, this method will be called to append a new list to each of the metrics results for
        the new epoch.

        :param epoch: The epoch that is about to begin.
        """
        self._logger.log_epoch()

    def on_epoch_end(self, epoch: int):
        """
        Before the given epoch ends, this method will be called to log the dynamic hyperparameters as needed.

        :param epoch: The epoch that has just ended.
        """
        # Update the dynamic hyperparameters dictionary:
        if self._dynamic_hyperparameters_keys:
            for (
                parameter_name,
                (source, key_chain,),
            ) in self._dynamic_hyperparameters_keys.items():
                self._logger.log_dynamic_hyperparameter(
                    parameter_name=parameter_name,
                    value=self._get_hyperparameter(source=source, key_chain=key_chain),
                )

    def on_train_begin(self):
        """
        After the training of the current epoch begins, this method will be called to set the mode for training.
        """
        self._is_training = True

    def on_train_end(self):
        """
        Before the training of the current epoch ends, this method will be called to lof the training summaries.
        """
        # Store the last training loss result of this epoch:
        loss_name = self._get_metric_name(
            metric_function=self._objects[self._ObjectKeys.LOSS_FUNCTION],
        )
        self._logger.log_training_summary(
            metric_name=loss_name,
            result=float(self._logger.training_results[loss_name][-1][-1]),
        )

        # Store the last training metrics results of this epoch:
        for metric_function in self._objects[self._ObjectKeys.METRIC_FUNCTIONS]:
            metric_name = self._get_metric_name(metric_function=metric_function,)
            self._logger.log_training_summary(
                metric_name=metric_name,
                result=float(self._logger.training_results[metric_name][-1][-1]),
            )

    def on_validation_begin(self):
        """
        After the validation (in a training case it will be per epoch) begins, this method will be called to set the
        mode for evaluation if the mode was not set for training already.
        """
        if self._is_training is None:
            self._is_training = False

    def on_validation_end(
        self, loss_value: MetricValueType, metric_values: List[float]
    ):
        """
        Before the validation (in a training case it will be per epoch) ends, this method will be called to log the
        validation results summaries.

        :param loss_value:    The loss summary of this validation.
        :param metric_values: The metrics summaries of this validation.
        """
        # Store the validation loss average of this epoch:
        self._logger.log_validation_summary(
            metric_name=self._get_metric_name(
                metric_function=self._objects[self._ObjectKeys.LOSS_FUNCTION],
            ),
            result=float(loss_value),
        )

        # Store the validation metrics averages of this epoch:
        for metric_function, metric_value in zip(
            self._objects[self._ObjectKeys.METRIC_FUNCTIONS], metric_values
        ):
            self._logger.log_validation_summary(
                metric_name=self._get_metric_name(metric_function=metric_function,),
                result=float(metric_value),
            )

    def on_train_batch_begin(self, batch: int, x: Tensor, y_true: Tensor):
        """
        After the training of the given batch begins, this method will be called to check whether this iteration needs
        to be logged.

        :param batch:  The current batch iteration of when this method is called.
        :param x:      The input part of the current batch.
        :param y_true: The true value part of the current batch.
        """
        self._logger.log_training_iteration()

    def on_validation_batch_begin(self, batch: int, x: Tensor, y_true: Tensor):
        """
        After the validation of the given batch begins, this method will be called to check whether this iteration needs
        to be logged.

        :param batch:  The current batch iteration of when this method is called.
        :param x:      The input part of the current batch.
        :param y_true: The true value part of the current batch.
        """
        self._logger.log_validation_iteration()

    def on_train_loss_end(self, loss_value: MetricValueType):
        """
        After the training calculation of the loss, this method will be called to log the loss value.

        :param loss_value: The recent loss value calculated during training.
        """
        # Store the loss value at the current epoch:
        self._logger.log_training_result(
            metric_name=self._get_metric_name(
                metric_function=self._objects[self._ObjectKeys.LOSS_FUNCTION],
            ),
            result=float(loss_value),
        )

    def on_validation_loss_end(self, loss_value: MetricValueType):
        """
        After the validating calculation of the loss, this method will be called to log the loss value.

        :param loss_value: The recent loss value calculated during validation.
        """
        # Store the loss value at the current epoch:
        self._logger.log_validation_result(
            metric_name=self._get_metric_name(
                metric_function=self._objects[self._ObjectKeys.LOSS_FUNCTION],
            ),
            result=float(loss_value),
        )

    def on_train_metrics_end(self, metric_values: List[MetricValueType]):
        """
        After the training calculation of the metrics, this method will be called to log the metrics values.

        :param metric_values: The recent metric values calculated during training.
        """
        for metric_function, metric_value in zip(
            self._objects[self._ObjectKeys.METRIC_FUNCTIONS], metric_values
        ):
            self._logger.log_training_result(
                metric_name=self._get_metric_name(metric_function=metric_function,),
                result=float(metric_value),
            )

    def on_validation_metrics_end(self, metric_values: List[MetricValueType]):
        """
        After the validating calculation of the metrics, this method will be called to log the metrics values.

        :param metric_values: The recent metric values calculated during validation.
        """
        for metric_function, metric_value in zip(
            self._objects[self._ObjectKeys.METRIC_FUNCTIONS], metric_values
        ):
            self._logger.log_validation_result(
                metric_name=self._get_metric_name(metric_function=metric_function,),
                result=float(metric_value),
            )

    def _add_auto_hyperparameters(self):
        """
        Add auto log's hyperparameters if they are accessible. The automatic hyperparameters being added are:
        batch size, learning rate. In addition to that, the context parameters (if available) will be logged as static
        hyperparameters as well.
        """
        # Log the context parameters:
        if self._logger.context is not None:
            self._logger.log_context_parameters()

        # Add batch size:
        bath_size_key = "batch_size"
        if bath_size_key not in self._static_hyperparameters_keys:
            if self._objects[self._ObjectKeys.TRAINING_SET] is not None and hasattr(
                self._objects[self._ObjectKeys.TRAINING_SET], "batch_size"
            ):
                self._static_hyperparameters_keys[bath_size_key] = getattr(
                    self._objects[self._ObjectKeys.TRAINING_SET], "batch_size"
                )
            elif self._objects[self._ObjectKeys.VALIDATION_SET] is not None and hasattr(
                self._objects[self._ObjectKeys.VALIDATION_SET], "batch_size"
            ):
                self._static_hyperparameters_keys[bath_size_key] = getattr(
                    self._objects[self._ObjectKeys.VALIDATION_SET], "batch_size"
                )

        # Add learning rate:
        learning_rate_key = "lr"
        learning_rate_key_chain = (
            HyperparametersKeys.OPTIMIZER,
            ["param_groups", 0, "lr"],
        )
        if learning_rate_key not in self._dynamic_hyperparameters_keys:
            if self._objects[self._ObjectKeys.OPTIMIZER] is not None:
                try:
                    self._get_hyperparameter(
                        source=learning_rate_key_chain[0],
                        key_chain=learning_rate_key_chain[1],
                    )
                    self._dynamic_hyperparameters_keys[
                        learning_rate_key
                    ] = learning_rate_key_chain
                except (TypeError, KeyError, IndexError, ValueError):
                    pass

    def _get_hyperparameter(
        self,
        source: str,
        key_chain: Union[List[Union[str, int]], Callable[[], TrackableType]],
    ) -> TrackableType:
        """
        Access the hyperparameter from the source using the given key chain.

        :param source:    The object string (out of 'HyperparametersKeys') to get the hyperparamter value from.
        :param key_chain: The keys and indices to get to the hyperparameter from the given source object or if the
                          source is equal to 'HyperparametersKeys.CUSTOM', the callable custom method to get the value
                          to track.

        :return: The hyperparameter value.

        :raise TypeError:  In case the source is 'HyperparametersKeys.CUSTOM' but the given value is not callable.
        :raise KeyError:   In case the one of the keys in the key chain is incorrect.
        :raise IndexError: In case the one of the keys in the key chain is incorrect.
        :raise ValueError: In case the value is not trackable.
        """
        # Get the value using the provided key chain:
        if source == HyperparametersKeys.CUSTOM:
            # It is a custom callable method:
            try:
                value = key_chain()
            except TypeError:
                raise TypeError(
                    "The given value of the source '{}' "
                    "is of type '{}' and is not callable."
                    "".format(source, type(key_chain))
                )
        else:
            # Needed to be extracted via key chain:
            value = self._objects[source]
            for key in key_chain:
                try:
                    if (
                        isinstance(value, dict)
                        or isinstance(value, list)
                        or isinstance(value, tuple)
                    ):
                        value = value[key]
                    else:
                        value = getattr(value, key)
                except (KeyError, IndexError, AttributeError) as KeyChainError:
                    raise KeyChainError(
                        "Error during getting a hyperparameter value from the {} object. "
                        "The {} in it does not have the following key/index from the key provided: {}"
                        "".format(source.__class__, value.__class__, key)
                    )

        # Parse the value:
        if isinstance(value, Tensor) or isinstance(value, Parameter):
            if value.numel() == 1:
                value = float(value)
            else:
                raise ValueError(
                    "The parameter with the following key chain: {} is a pytorch.Tensor with {} elements."
                    "PyTorch tensors are trackable only if they have 1 element."
                    "".format(key_chain, value.numel())
                )
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                value = float(value)
            else:
                raise ValueError(
                    "The parameter with the following key chain: {} is a numpy.ndarray with {} elements."
                    "numpy arrays are trackable only if they have 1 element."
                    "".format(key_chain, value.size)
                )
        elif not (
            isinstance(value, float)
            or isinstance(value, int)
            or isinstance(value, str)
            or isinstance(value, bool)
        ):
            raise ValueError(
                "The parameter with the following key chain: {} is of type '{}'."
                "The only trackable types are: float, int, str and bool."
                "".format(key_chain, type(value))
            )
        return value

    @staticmethod
    def _get_metric_name(metric_function: MetricFunctionType):
        """
        Get the given metric name.

        :param metric_function: The metric function to get its name.

        :return: The metric name.
        """
        if isinstance(metric_function, Module):
            function_name = metric_function.__class__.__name__
        else:
            function_name = metric_function.__name__
        return function_name
