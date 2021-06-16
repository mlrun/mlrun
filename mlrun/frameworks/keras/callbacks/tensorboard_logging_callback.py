from datetime import datetime
from typing import Callable, Dict, List, Union

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp_api
from tensorboard.plugins.hparams import api_pb2 as hp_api_pb2
from tensorboard.plugins.hparams import summary as hp_summary
from tensorflow import Tensor, Variable
from tensorflow.keras import Model
from tensorflow.python.ops import summary_ops_v2

import mlrun
from mlrun.frameworks._common.loggers import TensorboardLogger, TrackableType
from mlrun.frameworks.keras.callbacks.logging_callback import LoggingCallback


class _KerasTensorboardLogger(TensorboardLogger):
    """
    The keras framework implementation of the 'TensorboardLogger'.
    """

    def __init__(
        self,
        statistics_functions: List[Callable[[Union[Variable]], Union[float, Variable]]],
        context: mlrun.MLClientCtx = None,
        tensorboard_directory: str = None,
        run_name: str = None,
    ):
        """
        Initialize a tensorboard logger callback with the given configuration. At least one of 'context' and
        'tensorboard_directory' must be given.

        :param statistics_functions:  A list of statistics functions to calculate at the end of each epoch on the
                                      tracked weights. Only relevant if weights are being tracked. The functions in
                                      the list must accept one Weight and return a float (or float convertible) value.
        :param context:               A mlrun context to use for logging into the user's tensorboard directory.
        :param tensorboard_directory: If context is not given, or if wished to set the directory even with context,
                                      this will be the output for the event logs of tensorboard.
        :param run_name:              This experiment run name. Each run name will be indexed at the end of the name so
                                      each experiment will be numbered automatically. If a context was given, the
                                      context's uid will be added instead of an index. If a run name was not given the
                                      current time in the following format: 'YYYY-mm-dd_HH:MM:SS'.
        """
        super(_KerasTensorboardLogger, self).__init__(
            statistics_functions=statistics_functions,
            context=context,
            tensorboard_directory=tensorboard_directory,
            run_name=run_name,
        )

        # Setup the tensorboard writer property:
        self._file_writer = None

    def open(self):
        """
        Create the output path and initialize the tensorboard file writer.
        """
        # Create the output path:
        self._create_output_path()

        # Use the output path to initialize the tensorboard file writer:
        self._file_writer = tf.summary.create_file_writer(self._output_path)
        self._file_writer.set_as_default()

    def log_run_start_text_to_tensorboard(self):
        """
        Log the initial information summary of this training / validation run to tensorboard.
        """
        with self._file_writer.as_default():
            tf.summary.text(
                name="MLRun", data=self._generate_run_start_text(), step=0,
            )

    def log_epoch_text_to_tensorboard(self):
        """
        Log the last epoch summary of this training run to tensorboard.
        """
        with self._file_writer.as_default():
            tf.summary.text(
                name="MLRun",
                data=self._generate_epoch_text(),
                step=self._training_iterations,
            )

    def log_run_end_text_to_tensorboard(self):
        """
        Log the final information summary of this training / validation run to tensorboard.
        """
        with self._file_writer.as_default():
            tf.summary.text(
                name="MLRun",
                data=self._generate_run_end_text(),
                step=(
                    self._validation_iterations
                    if self._training_iterations == 0
                    else self._training_iterations
                ),
            )

    def log_parameters_table_to_tensorboard(self):
        """
        Log the validation summaries, static and dynamic hyperparameters to the 'HParams' table in tensorboard.
        """
        # Check if needed to track hyperparameters:
        if (
            len(self._static_hyperparameters) == 0
            and len(self._dynamic_hyperparameters) == 0
        ):
            return

        # Prepare the static hyperparameters values:
        non_graph_parameters = {"Date": str(datetime.now()).split(".")[0]}
        hp_param_list = [hp_api.HParam("Date")]
        for parameter, value in self._static_hyperparameters.items():
            non_graph_parameters[parameter] = value
            hp_param_list.append(hp_api.HParam(parameter))

        # Prepare the summaries values and the dynamic hyperparameters values (both registered as metrics):
        graph_parameters = {}
        hp_metric_list = []
        for metric in self._training_results:
            for prefix in ["training", "validation"]:
                metric_name = "{}/{}_{}".format(self._Sections.SUMMARY, prefix, metric)
                graph_parameters[metric_name] = 0.0
                hp_metric_list.append(hp_api.Metric(metric_name))
        for parameter, epochs in self._dynamic_hyperparameters.items():
            parameter_name = "{}/{}".format(self._Sections.HYPERPARAMETERS, parameter)
            graph_parameters[parameter_name] = epochs[-1]
            hp_metric_list.append(hp_api.Metric(parameter_name))

        # Write the hyperparameters and summaries to the table:
        with self._file_writer.as_default():
            hp_api.hparams_config(hparams=hp_param_list, metrics=hp_metric_list)
            hp_api.hparams(non_graph_parameters, trial_id=self._run_name)

    def log_training_results_to_tensorboard(self):
        """
        Log the recent training iteration metrics results to tensorboard.
        """
        with self._file_writer.as_default():
            for parameter, epochs in self._training_results.items():
                tf.summary.scalar(
                    name="{}/{}".format(self._Sections.TRAINING, parameter),
                    data=epochs[-1][-1],
                    step=self._training_iterations,
                )

    def log_validation_results_to_tensorboard(self):
        """
        Log the recent validation iteration metrics results to tensorboard.
        """
        with self._file_writer.as_default():
            for parameter, epochs in self._validation_results.items():
                tf.summary.scalar(
                    name="{}/{}".format(self._Sections.VALIDATION, parameter),
                    data=epochs[-1][-1],
                    step=self._validation_iterations,
                )

    def log_dynamic_hyperparameters_to_tensorboard(self):
        """
        Log the recent epoch dynamic hyperparameters values to tensorboard.
        """
        with self._file_writer.as_default():
            for parameter, epochs in self._dynamic_hyperparameters.items():
                tf.summary.scalar(
                    name="{}/{}".format(self._Sections.HYPERPARAMETERS, parameter),
                    data=epochs[-1],
                    step=self._epochs,
                )

    def log_summaries_to_tensorboard(self):
        """
        Log the recent epoch summaries results to tensorboard.
        """
        with self._file_writer.as_default():
            for prefix, summaries in zip(
                ["training", "validation"],
                [self._training_summaries, self._validation_summaries],
            ):
                for metric, epochs in summaries.items():
                    tf.summary.scalar(
                        name="{}/{}_{}".format(self._Sections.SUMMARY, prefix, metric),
                        data=epochs[-1],
                        step=self._epochs,
                    )

    def log_weights_histograms_to_tensorboard(self):
        """
        Log the current state of the weights as histograms to tensorboard.
        """
        with self._file_writer.as_default():
            for weight_name, weight_variable in self._weights.items():
                tf.summary.histogram(
                    name="{}/{}".format(self._Sections.WEIGHTS, weight_name),
                    data=weight_variable,
                    step=self._epochs,
                )

    def log_weights_images_to_tensorboard(self):
        """
        Log the current state of the weights as images to tensorboard.
        """
        pass

    def log_statistics_to_tensorboard(self):
        """
        Log the last stored statistics values this logger collected to tensorboard.
        """
        for statistic, weights in self._weights_statistics.items():
            for weight_name, epoch_values in weights.items():
                tf.summary.scalar(
                    name="{}/{}:{}".format(
                        self._Sections.WEIGHTS, weight_name, statistic
                    ),
                    data=epoch_values[-1],
                    step=self._epochs,
                )

    def log_model_to_tensorboard(self, model: Model):
        """
        Log the given model as a graph in tensorboard.
        """
        with self._file_writer.as_default():
            if tf.__version__ == "2.4.1":
                with summary_ops_v2.always_record_summaries():
                    summary_ops_v2.keras_model(name=model.name, data=model, step=0)
            elif tf.__version__ == "2.5.0":
                from tensorflow.python.keras.callbacks import keras_model_summary

                with summary_ops_v2.record_if(True):
                    keras_model_summary("keras", model, step=0)

    def flush(self):
        """
        Make sure all values were written to the directory logs so it will be available live.
        """
        self._file_writer.flush()

    def close(self):
        """
        Close the file writer object, wrapping up the hyperparameters table.
        """
        # Close the hyperparameters writing:
        if not (
            len(self._static_hyperparameters) == 0
            and len(self._dynamic_hyperparameters) == 0
        ):
            with self._file_writer.as_default():
                pb = hp_summary.session_end_pb(hp_api_pb2.STATUS_SUCCESS)
                raw_pb = pb.SerializeToString()
                tf.compat.v2.summary.experimental.write_raw_pb(raw_pb, step=0)

        # Flush and close the writer:
        self.flush()
        self._file_writer.close()


class TensorboardLoggingCallback(LoggingCallback):
    """
    Callback for logging data during training / evaluation to tensorboard. the available data in tensorboard will be:

    * Summary text of the run with a hyperlink to the MLRun log if it was done.
    * Hyperparameters tuning table: static hyperparameters, dynamic hyperparameters and epoch validation summaries.
    * Plots:

      * Per iteration (batch) plot for the training and validation metrics.
      * Per epoch plot for the dynamic hyperparameters and validation summaries results.
      * Per epoch weights statistics for each weight and statistic.

    * Histograms per epoch for each of the logged weights.
    * Distributions per epoch for each of the logged weights.
    * Images per epoch for each of the logged weights.
    * Model architecture graph.

    All the collected data will be available in this callback post the training / validation process and can be accessed
    via the 'training_results', 'validation_results', 'static_hyperparameters', 'dynamic_hyperparameters', 'summaries',
    'weights', 'weights_mean' and 'weights_std' properties.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        tensorboard_directory: str = None,
        run_name: str = None,
        weights: Union[bool, List[str]] = False,
        statistics_functions: List[
            Callable[[Union[Variable, Tensor]], Union[float, Tensor]]
        ] = None,
        dynamic_hyperparameters: Dict[
            str, Union[List[Union[str, int]], Callable[[], TrackableType]]
        ] = None,
        static_hyperparameters: Dict[
            str, Union[TrackableType, List[Union[str, int]]]
        ] = None,
        per_iteration_logging: int = 1,
        auto_log: bool = False,
    ):
        """
        Initialize a tensorboard logging callback with the given weights, hyperparameters and logging configurations.
        Note that at least one of 'context' and 'tensorboard_directory' must be given.

        :param context:                 A mlrun context to use for logging into the user's tensorboard directory.
        :param tensorboard_directory:   If context is not given, or if wished to set the directory even with context,
                                        this will be the output for the event logs of tensorboard.
        :param run_name:                This experiment run name. Each run name will be indexed at the end of the name
                                        so each experiment will be numbered automatically. If a context was given, the
                                        context's uid will be added instead of an index. If a run name was not given the
                                        current time in the following format: 'YYYY-mm-dd_HH:MM:SS'.
        :param weights:                 If wished to track weights to draw their histograms and calculate statistics per
                                        epoch, the weights names should be passed here. Note that each name given will
                                        be searched as 'if <NAME> in <WEIGHT_NAME>' so a simple module name will be
                                        enough to catch his weights. A boolean value can be passed to track all weights.
                                        Defaulted to False.
        :param statistics_functions:    A list of statistics functions to calculate at the end of each epoch on the
                                        tracked weights. Only relevant if weights are being tracked. The functions in
                                        the list must accept one Parameter (or Tensor) and return a float (or float
                                        convertible) value. The default statistics are 'mean' and 'std'. To get the
                                        default functions list for appending additional functions you can access it via
                                        'TensorboardLoggingCallback.get_default_weight_statistics_list()'. To not track
                                        statistics at all simply pass an empty list '[]'.
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
        :param per_iteration_logging:   Per how many iterations (batches) the callback should log the tracked values.
                                        Defaulted to 1 (meaning every iteration will be logged).
        :param auto_log:                Whether or not to enable auto logging, trying to track common static and dynamic
                                        hyperparameters.

        :raise ValueError: In case both 'context' and 'tensorboard_directory' parameters were not given.
        """
        super(TensorboardLoggingCallback, self).__init__(
            dynamic_hyperparameters=dynamic_hyperparameters,
            static_hyperparameters=static_hyperparameters,
            per_iteration_logging=per_iteration_logging,
            auto_log=auto_log,
        )

        # Validate input:
        if context is None and tensorboard_directory is None:
            raise ValueError(
                "Expecting to receive a mlrun.MLClientCtx context or a path to a directory to output"
                "the logging file but None were given."
            )

        # Replace the logger with a TensorboardLogger:
        del self._logger
        self._logger = _KerasTensorboardLogger(
            statistics_functions=(
                statistics_functions
                if statistics_functions is not None
                else self.get_default_weight_statistics_list()
            ),
            context=context,
            tensorboard_directory=tensorboard_directory,
            run_name=run_name,
        )

        # Save the configurations:
        self._tracked_weights = weights

        # Initialize flags:
        self._is_training = False
        self._logged_model = False
        self._logged_hyperparameters = False

    def get_weights(self) -> Dict[str, Variable]:
        """
        Get the weights tensors tracked. The weights will be stored in a dictionary where each key is the weight's name
        and the value is the weight's parameter (tensor).

        :return: The weights.
        """
        return self._logger.weights

    def get_weights_statistics(self) -> Dict[str, List[float]]:
        """
        Get the weights mean results logged. The results will be stored in a dictionary where each key is the weight's
        name and the value is a list of mean values per epoch.

        :return: The weights mean results.
        """
        return self._logger.weight_statistics

    @staticmethod
    def get_default_weight_statistics_list() -> List[
        Callable[[Union[Variable, Tensor]], Union[float, Tensor]]
    ]:
        """
        Get the default list of statistics functions being applied on the tracked weights each epoch.

        :return: The default statistics functions list.
        """
        return [tf.math.reduce_mean, tf.math.reduce_std]

    def on_train_begin(self, logs: dict = None):
        """
        Called once at the beginning of training process (one time call). Will log the pre-training (epoch 0)
        hyperparameters and weights.

        :param logs: Dict. Currently no data is passed to this argument for this method but that may change in the
                     future.
        """
        # The callback is on a 'fit' method - training:
        self._is_training = True

        # Start the tensorboard logger:
        self._logger.open()

        # Setup the run, logging relevant information and tracking weights:
        self._setup_run()

        # Log the model:
        self._logger.log_model_to_tensorboard(model=self.model)

        # Log the initial summary of the run:
        self._logger.log_run_start_text_to_tensorboard()

        # Log the initial weights (epoch 0):
        self._logger.log_weights_histograms_to_tensorboard()
        self._logger.log_weights_images_to_tensorboard()
        self._logger.log_weights_statistics()
        self._logger.log_statistics_to_tensorboard()

        # Make sure all values were written to the directory logs:
        self._logger.flush()

    def on_train_end(self, logs: dict = None):
        """
        Called at the end of training, wrapping up the tensorboard logging session.

        :param logs: Currently the output of the last call to `on_epoch_end()` is passed to this argument for this
                     method but that may change in the future.
        """
        super(TensorboardLoggingCallback, self).on_train_end()

        # Log the final summary of the run:
        self._logger.log_run_end_text_to_tensorboard()

        # Close the logger:
        self._logger.close()

    def on_test_begin(self, logs: dict = None):
        """
        Called at the beginning of evaluation or validation. Will be called on each epoch according to the validation
        per epoch configuration. In case it is an evaluation, the epoch 0 will be logged.

        :param logs: Currently no data is passed to this argument for this method but that may change in the
                     future.
        """
        # If this callback is part of evaluation and not training, need to check if the run was setup:
        if self._call_setup_run:
            # Start the tensorboard logger:
            self._logger.open()
            # Setup the run, logging relevant information and tracking weights:
            self._setup_run()
            # Log the initial weights (epoch 0):
            self._logger.log_weights_histograms_to_tensorboard()
            self._logger.log_weights_images_to_tensorboard()
            self._logger.log_weights_statistics()
            self._logger.log_statistics_to_tensorboard()
            # Make sure all values were written to the directory logs:
            self._logger.flush()

    def on_test_end(self, logs: dict = None):
        """
        Called at the end of evaluation or validation. Will be called on each epoch according to the validation
        per epoch configuration. The recent evaluation / validation results will be summarized and logged.

        :param logs: Currently no data is passed to this argument for this method but that may change in the
                     future.
        """
        super(TensorboardLoggingCallback, self).on_test_end(logs=logs)

        # Check if needed to end the run (in case of evaluation and not training):
        if not self._is_training:
            # Log the run final summary text:
            self._logger.log_run_end_text_to_tensorboard()
            # Close the logger:
            self._logger.close()

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """
        Called at the end of an epoch, logging the current dynamic hyperparameters values, summaries and weights to
        tensorboard.

        :param epoch: Integer, index of epoch.
        :param logs:  Metric results for this training epoch, and for the validation epoch if validation is
                      performed. Validation result keys are prefixed with `val_`. For training epoch, the values of the
                      `Model`'s metrics are returned. Example : `{'loss': 0.2, 'acc': 0.7}`.
        """
        # Update the dynamic hyperparameters
        super(TensorboardLoggingCallback, self).on_epoch_end(epoch=epoch)

        # Add this epoch text summary:
        self._logger.log_epoch_text_to_tensorboard()

        # Add this epoch loss and metrics summaries to their graphs:
        self._logger.log_summaries_to_tensorboard()

        # Add this epoch dynamic hyperparameters values to their graphs:
        self._logger.log_dynamic_hyperparameters_to_tensorboard()

        # Add weight histograms, images and statistics for all the tracked weights:
        self._logger.log_weights_histograms_to_tensorboard()
        self._logger.log_weights_images_to_tensorboard()
        self._logger.log_weights_statistics()
        self._logger.log_statistics_to_tensorboard()

        # Make sure all values were written to the directory logs:
        self._logger.flush()

    def on_train_batch_begin(self, batch: int, logs: dict = None):
        """
        Called at the beginning of a training batch in `fit` methods. The logger will check if this batch is needed to
        be logged according to the configuration. Note that if the `steps_per_execution` argument to `compile` in
        `tf.keras.Model` is set to `N`, this method will only be called every `N` batches.

        :param batch: Integer, index of batch within the current epoch.
        :param logs:  Contains the return value of `model.train_step`. Typically, the values of the `Model`'s
                      metrics are returned. Example: `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        super(TensorboardLoggingCallback, self).on_train_batch_begin(
            batch=batch, logs=logs
        )

    def on_train_batch_end(self, batch: int, logs: dict = None):
        """
        Called at the end of a training batch in `fit` methods. The batch metrics results will be logged. If it is the
        first batch to end, the model architecture and hyperparameters will be logged as well. Note that if the
        `steps_per_execution` argument to `compile` in `tf.keras.Model` is set to `N`, this method will only be called
        every `N` batches.

        :param batch: Integer, index of batch within the current epoch.
        :param logs:  Aggregated metric results up until this batch.
        """
        # Log the batch's results:
        super(TensorboardLoggingCallback, self).on_train_batch_end(
            batch=batch, logs=logs
        )

        # Add this batch loss and metrics results to their graphs:
        self._logger.log_training_results_to_tensorboard()

        # Check if needed to log hyperparameters:
        if not self._logged_hyperparameters:
            self._logger.log_parameters_table_to_tensorboard()
            self._logged_hyperparameters = True
            self._logger.log_dynamic_hyperparameters_to_tensorboard()

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
        super(TensorboardLoggingCallback, self).on_test_batch_begin(
            batch=batch, logs=logs
        )

    def on_test_batch_end(self, batch: int, logs: dict = None):
        """
        Called at the end of a batch in `evaluate` methods. Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided. The batch metrics results will be logged. In case it is an evaluation
        run, if this was the first batch the model architecture and hyperparameters will be logged as well. Note that if
        the `steps_per_execution` argument to `compile` in `tf.keras.Model` is set to `N`, this method will only be
        called every `N` batches.

        :param batch: Integer, index of batch within the current epoch.
        :param logs:  Aggregated metric results up until this batch.
        """
        # Log the batch's results:
        super(TensorboardLoggingCallback, self).on_test_batch_end(
            batch=batch, logs=logs
        )

        # Add this batch loss and metrics results to their graphs:
        self._logger.log_validation_results_to_tensorboard()

        # Check if needed to log hyperparameters:
        if not self._logged_hyperparameters:
            self._logger.log_parameters_table_to_tensorboard()
            self._logged_hyperparameters = True
            self._logger.log_dynamic_hyperparameters_to_tensorboard()

    def _setup_run(self):
        """
        After the trainer / evaluator run begins, this method will be called to setup the results, hyperparameters
        and weights dictionaries for logging.
        """
        super(TensorboardLoggingCallback, self)._setup_run()

        # Collect the weights for drawing histograms according to the stored configuration:
        if self._tracked_weights is False:
            return

        for layer in self.model.layers:
            collect = False
            if self._tracked_weights is True:  # Collect all weights
                collect = True
            else:
                for tag in self._tracked_weights:  # Collect by given name
                    if tag in layer.name:
                        collect = True
                        break
            if collect:
                for weight_variable in layer.weights:
                    self._logger.log_weight(
                        weight_name=weight_variable.name, weight_holder=weight_variable
                    )
