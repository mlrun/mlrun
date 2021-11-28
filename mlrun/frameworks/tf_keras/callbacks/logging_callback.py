from typing import Callable, Dict, List, Union

import numpy as np
import tensorflow as tf
from tensorflow import Tensor, Variable
from tensorflow.keras.callbacks import Callback

import mlrun

from ..._dl_common.loggers import Logger, LoggerMode, TrackableType


class LoggingCallback(Callback):
    """
    Callback for collecting data during training / evaluation. All the collected data will be available in this callback
    post the training / evaluation process and can be accessed via the 'training_results', 'validation_results',
    'static_hyperparameters', 'dynamic_hyperparameters' and 'summaries' properties.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        dynamic_hyperparameters: Dict[
            str, Union[List[Union[str, int]], Callable[[], TrackableType]]
        ] = None,
        static_hyperparameters: Dict[
            str, Union[TrackableType, List[Union[str, int]]]
        ] = None,
        auto_log: bool = False,
    ):
        """
        Initialize a logging callback with the given hyperparameters and logging configurations.

        :param context:                 MLRun context to automatically log its parameters if 'auto_log' is True.
        :param dynamic_hyperparameters: If needed to track a hyperparameter dynamically (sample it each epoch) it should
                                        be passed here. The parameter expects a dictionary where the keys are the
                                        hyperparameter chosen names and the values are a key chain from the model. A key
                                        chain is a list of keys and indices to know how to access the needed
                                        hyperparameter from the model. If the hyperparameter is not of accessible from
                                        the model, a custom callable method can be passed. For example, to track the
                                        'lr' attribute of an optimizer and a custom parameter, one should pass:
                                        {
                                            "learning rate": ["optimizer", "lr"],
                                            "custom_parameter": get_custom_parameter
                                        }
        :param static_hyperparameters:  If needed to track a hyperparameter one time per run it should be passed here.
                                        The parameter expects a dictionary where the keys are the
                                        hyperparameter chosen names and the values are the hyperparameter static value
                                        or a key chain just like the dynamic hyperparameter. For example, to track the
                                        'epochs' of an experiment run, one should pass:
                                        {
                                            "epochs": 7
                                        }
        :param auto_log:                Whether or not to enable auto logging, trying to track common static and dynamic
                                        hyperparameters.
        """
        super(LoggingCallback, self).__init__()
        self._supports_tf_logs = True

        # Store the configurations:
        self._dynamic_hyperparameters_keys = (
            dynamic_hyperparameters if dynamic_hyperparameters is not None else {}
        )
        self._static_hyperparameters_keys = (
            static_hyperparameters if static_hyperparameters is not None else {}
        )

        # Initialize the logger:
        self._logger = Logger(context=context)

        # For calculating the batch's values we need to collect the epochs sums:
        # [Metric: str] -> [Sum: float]
        self._training_epoch_sums = {}  # type: Dict[str, float]
        self._validation_epoch_sums = {}  # type: Dict[str, float]

        # Setup the flags:
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

    def get_training_summaries(self) -> Dict[str, List[float]]:
        """
        Get the training summaries of the metrics results. The summaries will be stored in a dictionary where each key
        is the metric names and the value is a list of all the summary values per epoch.

        :return: The training summaries.
        """
        return self._logger.training_summaries

    def get_validation_summaries(self) -> Dict[str, List[float]]:
        """
        Get the validation summaries of the metrics results. The summaries will be stored in a dictionary where each key
        is the metric names and the value is a list of all the summary values per epoch.

        :return: The validation summaries.
        """
        return self._logger.validation_summaries

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

    def get_epochs(self) -> int:
        """
        Get the overall epochs this callback participated in.

        :return: The overall epochs this callback participated in.
        """
        return self._logger.epochs

    def get_training_iterations(self) -> int:
        """
        Get the overall training iterations this callback participated in.

        :return: The overall training iterations this callback participated in.
        """
        return self._logger.training_iterations

    def get_validation_iterations(self) -> int:
        """
        Get the overall validation iterations this callback participated in.

        :return: The overall validation iterations this callback participated in.
        """
        return self._logger.validation_iterations

    def on_train_begin(self, logs: dict = None):
        """
        Called once at the beginning of training process (one time call).

        :param logs: Currently no data is passed to this argument for this method but that may change in the
                     future.
        """
        self._is_training = True
        self._setup_run()

    def on_test_begin(self, logs: dict = None):
        """
        Called at the beginning of evaluation or validation. Will be called on each epoch according to the validation
        per epoch configuration.

        :param logs: Currently no data is passed to this argument for this method but that may change in the
                     future.
        """
        # Check if needed to mark this run as evaluation:
        if self._is_training is None:
            self._is_training = False
            self._logger.set_mode(mode=LoggerMode.EVALUATION)

        # If this callback is part of evaluation and not training, need to check if the run was setup:
        if not self._is_training:
            self._setup_run()

    def on_test_end(self, logs: dict = None):
        """
        Called at the end of evaluation or validation. Will be called on each epoch according to the validation
        per epoch configuration. The recent evaluation / validation results will be summarized and logged.

        :param logs: Currently no data is passed to this argument for this method but that may change in the
                     future.
        """
        # Store the metrics average of this epoch:
        for metric_name, epoch_values in self._logger.validation_results.items():
            # Check if needed to initialize:
            if metric_name not in self._logger.validation_summaries:
                self._logger.validation_summaries[metric_name] = []
            self._logger.log_validation_summary(
                metric_name=metric_name,
                result=float(sum(epoch_values[-1]) / len(epoch_values[-1])),
            )

    def on_epoch_begin(self, epoch: int, logs: dict = None):
        """
        Called at the start of an epoch, logging it and appending a new epoch to the logger's dictionaries.

        :param epoch: Integer, index of epoch.
        :param logs:  Currently no data is passed to this argument for this method but that may change in the
                      future.
        """
        # Log a new epoch:
        self._logger.log_epoch()

        # Reset the metrics sum:
        for sum_dictionary in [self._training_epoch_sums, self._validation_epoch_sums]:
            for metric in sum_dictionary:
                sum_dictionary[metric] = 0

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """
        Called at the end of an epoch, logging the training summaries and the current dynamic hyperparameters values.

        :param epoch: Integer, index of epoch.
        :param logs:  Metric results for this training epoch, and for the validation epoch if validation is
                      performed. Validation result keys are prefixed with `val_`. For training epoch, the values of the
                      `Model`'s metrics are returned. Example : `{'loss': 0.2, 'acc': 0.7}`.
        """
        # Store the last training result of this epoch:
        for metric_name, epoch_values in self._logger.training_results.items():
            # Check if needed to initialize:
            if metric_name not in self._logger.training_summaries:
                self._logger.training_summaries[metric_name] = []
            self._logger.log_training_summary(
                metric_name=metric_name, result=float(epoch_values[-1][-1])
            )

        # Update the dynamic hyperparameters dictionary:
        if self._dynamic_hyperparameters_keys:
            for name, key_chain in self._dynamic_hyperparameters_keys.items():
                self._logger.log_dynamic_hyperparameter(
                    parameter_name=name,
                    value=self._get_hyperparameter(key_chain=key_chain),
                )

    def on_train_batch_begin(self, batch: int, logs: dict = None):
        """
        Called at the beginning of a training batch in `fit` methods. The logger will check if this batch is needed to
        be logged according to the configuration. Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N` batches.

        :param batch: Integer, index of batch within the current epoch.
        :param logs:  Contains the return value of `model.train_step`. Typically, the values of the `Model`'s
                      metrics are returned. Example: `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        self._logger.log_training_iteration()

    def on_train_batch_end(self, batch: int, logs: dict = None):
        """
        Called at the end of a training batch in `fit` methods. The batch metrics results will be logged. Note that if
        the `steps_per_execution` argument to `compile` in `tf.keras.Model` is set to `N`, this method will only be
        called every `N` batches.

        :param batch: Integer, index of batch within the current epoch.
        :param logs:  Aggregated metric results up until this batch.
        """
        self._on_batch_end(
            results_dictionary=self._logger.training_results,
            sum_dictionary=self._training_epoch_sums,
            logs=logs,
        )

    def on_test_batch_begin(self, batch: int, logs: dict = None):
        """
        Called at the beginning of a batch in `evaluate` methods. Also called at the beginning of a validation batch in
        the `fit` methods, if validation data is provided. The logger will check if this batch is needed to be logged
        according to the configuration. Note that if the `steps_per_execution` argument to `compile` in `tf.keras.Model`
        is set to `N`, this method will only be called every `N` batches.

        :param batch: Integer, index of batch within the current epoch.
        :param logs:  Contains the return value of `model.test_step`. Typically, the values of the `Model`'s
                      metrics are returned.  Example: `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        self._logger.log_validation_iteration()

    def on_test_batch_end(self, batch: int, logs: dict = None):
        """
        Called at the end of a batch in `evaluate` methods. Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided. The batch metrics results will be logged. Note that if the
        `steps_per_execution` argument to `compile` in `tf.keras.Model` is set to `N`, this method will only be called
        every `N` batches.

        :param batch: Integer, index of batch within the current epoch.
        :param logs:  Aggregated metric results up until this batch.
        """
        self._on_batch_end(
            results_dictionary=self._logger.validation_results,
            sum_dictionary=self._validation_epoch_sums,
            logs=logs,
        )

    def _setup_run(self):
        """
        After the trainer / evaluator run begins, this method will be called to setup the logger, logging all the
        hyperparameters pre run (epoch 0).
        """
        # Setup the hyperparameters dictionaries:
        if self._auto_log:
            self._add_auto_hyperparameters()

        # Static hyperparameters:
        for name, value in self._static_hyperparameters_keys.items():
            if isinstance(value, List):
                # Its a parameter that needed to be extracted via key chain.
                self._logger.log_static_hyperparameter(
                    parameter_name=name,
                    value=self._get_hyperparameter(key_chain=value),
                )
            else:
                # Its a custom hyperparameter.
                self._logger.log_static_hyperparameter(parameter_name=name, value=value)

        # Dynamic hyperparameters:
        for name, key_chain in self._dynamic_hyperparameters_keys.items():
            self._logger.log_dynamic_hyperparameter(
                parameter_name=name,
                value=self._get_hyperparameter(key_chain=key_chain),
            )

    def _on_batch_end(self, results_dictionary: dict, sum_dictionary: dict, logs: dict):
        """
        Log the given metrics values to the given results dictionary.

        :param results_dictionary: One of 'self._logger.training_results' or 'self._logger.validation_results'.
        :param sum_dictionary:     One of 'self._training_epoch_sums' or 'self._validation_epoch_sums'.
        :param logs:               The loss and metrics results of the recent batch.
        """
        # Parse the metrics names in the logs:
        logs = self._parse_metrics(logs=logs)

        # Log the given metrics as needed:
        for metric_name, aggregated_value in logs.items():
            # Check if needed to initialize:
            if metric_name not in results_dictionary:
                results_dictionary[metric_name] = [[]]
                sum_dictionary[metric_name] = 0
            # Calculate the last value:
            elements_number = len(results_dictionary[metric_name][-1]) + 1
            elements_sum = sum_dictionary[metric_name]
            last_metric_value = elements_number * aggregated_value - elements_sum
            # Store the metric value at the current epoch:
            sum_dictionary[metric_name] += last_metric_value
            results_dictionary[metric_name][-1].append(float(last_metric_value))

    def _add_auto_hyperparameters(self):
        """
        Add auto log's hyperparameters if they are accessible. The automatic hyperparameters being added are:
        learning rate.  In addition to that, the context parameters (if available) will be logged as static
        hyperparameters as well.
        """
        # Log the context parameters:
        if self._logger.context is not None:
            self._logger.log_context_parameters()

        # Add learning rate:
        learning_rate_key = "lr"
        learning_rate_key_chain = ["optimizer", "lr"]
        if learning_rate_key not in self._dynamic_hyperparameters_keys and hasattr(
            self.model, "optimizer"
        ):
            try:
                self._get_hyperparameter(key_chain=learning_rate_key_chain)
                self._dynamic_hyperparameters_keys[
                    learning_rate_key
                ] = learning_rate_key_chain
            except (KeyError, IndexError, ValueError):
                pass

    def _get_hyperparameter(
        self, key_chain: Union[Callable[[], TrackableType], List[Union[str, int]]]
    ) -> TrackableType:
        """
        Access the hyperparameter from the model stored in this callback using the given key chain.

        :param key_chain: The keys and indices to get to the hyperparameter from the model or a callable method.

        :return: The hyperparameter value.

        :raise KeyError:   In case the one of the keys in the key chain is incorrect.
        :raise IndexError: In case the one of the keys in the key chain is incorrect.
        :raise MLRunInvalidArgumentError: In case the value is not trackable.
        """
        if isinstance(key_chain, Callable):
            # It is a custom callable method:
            value = key_chain()
        else:
            # Get the value using the provided key chain:
            value = self.model
            for key in key_chain:
                try:
                    if isinstance(key, int):
                        value = value[key]
                    else:
                        value = getattr(value, key)
                except KeyError or IndexError as KeyChainError:
                    raise KeyChainError(
                        f"Error during getting a hyperparameter value with the key chain {key_chain}. "
                        f"The {value.__class__} in it does not have the following key/index from the key provided: "
                        f"{key}"
                    )

        # Parse the value:
        if isinstance(value, Tensor) or isinstance(value, Variable):
            if int(tf.size(value)) == 1:
                value = float(value)
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"The parameter with the following key chain: {key_chain} is a tensorflow.Tensor with "
                    f"{value.numel()} elements. Tensorflow tensors are trackable only if they have 1 element."
                )
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                value = float(value)
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"The parameter with the following key chain: {key_chain} is a numpy.ndarray with {value.size} "
                    f"elements. numpy arrays are trackable only if they have 1 element."
                )
        elif not (
            isinstance(value, float)
            or isinstance(value, int)
            or isinstance(value, str)
            or isinstance(value, bool)
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The parameter with the following key chain: {key_chain} is of type '{type(value)}'. The only "
                f"trackable types are: float, int, str and bool."
            )
        return value

    @staticmethod
    def _parse_metrics(logs: dict) -> dict:
        """
        Parse the default logs dictionary metrics names to be clean (without the 'val_' prefix).

        :param logs: The logs given from a callback method.

        :return: The parsed logs.
        """
        parsed_logs = {}
        for metric, value in logs.items():
            if metric.startswith("val_"):
                metric = metric[4:]
            parsed_logs[metric] = value

        return parsed_logs
