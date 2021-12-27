from datetime import datetime
from typing import Callable, Dict, List, Union

import tensorflow as tf
from packaging import version
from tensorboard.plugins.hparams import api as hp_api
from tensorboard.plugins.hparams import api_pb2 as hp_api_pb2
from tensorboard.plugins.hparams import summary as hp_summary
from tensorflow import Tensor, Variable
from tensorflow.keras import Model
from tensorflow.python.ops import summary_ops_v2

import mlrun

from ..._dl_common.loggers import TensorboardLogger, TrackableType
from .logging_callback import LoggingCallback


class _TFKerasTensorboardLogger(TensorboardLogger):
    """
    The keras framework implementation of the 'TensorboardLogger'.
    """

    def __init__(
        self,
        statistics_functions: List[Callable[[Union[Variable]], Union[float, Variable]]],
        context: mlrun.MLClientCtx = None,
        tensorboard_directory: str = None,
        run_name: str = None,
        update_frequency: Union[int, str] = "epoch",
    ):
        """
        Initialize a tensorboard logger callback with the given configuration. At least one of 'context' and
        'tensorboard_directory' must be given.

        :param statistics_functions:  A list of statistics functions to calculate at the end of each epoch on the
                                      tracked weights. Only relevant if weights are being tracked. The functions in
                                      the list must accept one Weight and return a float (or float convertible) value.
        :param context:               A MLRun context to use for logging into the user's tensorboard directory. The
                                      context parameters can be logged as static hyperparameters as well.
        :param tensorboard_directory: If context is not given, or if wished to set the directory even with context,
                                      this will be the output for the event logs of tensorboard. If not given, context
                                      must be provided as the default tensorboard output directory will be:
                                      /User/.tensorboard/<PROJECT_NAME> or if working on local, the set artifacts path.
        :param run_name:              This experiment run name. Each run name will be indexed at the end of the name so
                                      each experiment will be numbered automatically. If a context was given, the
                                      context's uid will be added instead of an index. If a run name was not given the
                                      current time stamp will be used.
        :param update_frequency:      Per how many iterations (batches) the callback should write the tracked values to
                                      tensorboard. Can be passed as a string equal to 'epoch' for per epoch and 'batch'
                                      for per single batch, or as an integer specifying per how many iterations to
                                      update. Notice that writing to tensorboard too frequently may cause the training
                                      to be slower. Defaulted to 'epoch'.
        """
        super(_TFKerasTensorboardLogger, self).__init__(
            statistics_functions=statistics_functions,
            context=context,
            tensorboard_directory=tensorboard_directory,
            run_name=run_name,
            update_frequency=update_frequency,
        )

        # Setup the tensorboard writer property:
        self._file_writer = None

    def write_model_to_tensorboard(self, model: Model):
        """
        Write the given model as a graph in tensorboard.

        :param model: The model to write to tensorboard.
        """
        with self._file_writer.as_default():
            # Log the model's graph according to tensorflow's version:
            if version.parse(tf.__version__) < version.parse("2.5.0"):
                with summary_ops_v2.always_record_summaries():
                    summary_ops_v2.keras_model(name=model.name, data=model, step=0)
            else:
                from tensorflow.python.keras.callbacks import keras_model_summary

                with summary_ops_v2.record_if(True):
                    keras_model_summary("keras", model, step=0)

    def write_parameters_table_to_tensorboard(self):
        """
        Write the summaries, static and dynamic hyperparameters to the table in tensorboard's hparams section. This
        method is called once for creating the hparams table.
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
                metric_name = f"{self._Sections.SUMMARY}/{prefix}_{metric}"
                graph_parameters[metric_name] = 0.0
                hp_metric_list.append(hp_api.Metric(metric_name))
        for parameter, epochs in self._dynamic_hyperparameters.items():
            parameter_name = f"{self._Sections.HYPERPARAMETERS}/{parameter}"
            graph_parameters[parameter_name] = epochs[-1]
            hp_metric_list.append(hp_api.Metric(parameter_name))

        # Write the hyperparameters and summaries to the table:
        with self._file_writer.as_default():
            hp_api.hparams_config(hparams=hp_param_list, metrics=hp_metric_list)
            hp_api.hparams(non_graph_parameters, trial_id=self._run_name)

    def open(self):
        """
        Create the output path and initialize the tensorboard file writer.
        """
        # Create the output path:
        self._create_output_path()

        # Use the output path to initialize the tensorboard file writer:
        self._file_writer = tf.summary.create_file_writer(self._output_path)
        self._file_writer.set_as_default()

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

    def _write_text_to_tensorboard(self, tag: str, text: str, step: int):
        """
        Write text to tensorboard's text section. Summary information of this training / validation run will be logged
        to tensorboard using this method.

        :param tag:  The tag of the text (box it will be appearing under).
        :param text: The text to write.
        :param step: The iteration / epoch the text belongs to.
        """
        with self._file_writer.as_default():
            tf.summary.text(
                name=tag, data=text, step=step,
            )

    def _write_scalar_to_tensorboard(self, name: str, value: float, step: int):
        """
        Write the scalar's value into its plot.

        :param name:  The plot's name.
        :param value: The value to add to the plot.
        :param step:  The iteration / epoch the value belongs to.
        """
        with self._file_writer.as_default():
            tf.summary.scalar(
                name=name, data=value, step=step,
            )

    def _write_weight_histogram_to_tensorboard(
        self, name: str, weight: Variable, step: int
    ):
        """
        Write the current state of the weights as histograms to tensorboard.

        :param name:   The weight's name.
        :param weight: The weight to write its histogram.
        :param step:   The iteration / epoch the weight's histogram state belongs to.
        """
        with self._file_writer.as_default():
            tf.summary.histogram(
                name=name, data=weight, step=step,
            )

    def _write_weight_image_to_tensorboard(
        self, name: str, weight: Variable, step: int
    ):
        """
        Log the current state of the weights as images to tensorboard.

        :param name:   The weight's name.
        :param weight: The weight to write its image.
        :param step:   The iteration / epoch the weight's image state belongs to.
        """
        raise NotImplementedError


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
        update_frequency: Union[int, str] = "epoch",
        auto_log: bool = False,
    ):
        """
        Initialize a tensorboard logging callback with the given weights, hyperparameters and logging configurations.
        Note that at least one of 'context' and 'tensorboard_directory' must be given.

        :param context:                 A mlrun context to use for logging into the user's tensorboard directory.
        :param tensorboard_directory:   If context is not given, or if wished to set the directory even with context,
                                        this will be the output for the event logs of tensorboard. If not given, the
                                        'tensorboard_dir' parameter will be tried to be taken from the provided context.
                                        If not found in the context, the default tensorboard output directory will be:
                                        /User/.tensorboard/<PROJECT_NAME> or if working on local, the set artifacts
                                        path.
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
        :param update_frequency:        Per how many iterations (batches) the callback should write the tracked values
                                        to tensorboard. Can be passed as a string equal to 'epoch' for per epoch and
                                        'batch' for per single batch, or as an integer specifying per how many
                                        iterations to update. Notice that writing to tensorboard too frequently may
                                        cause the training to be slower. Defaulted to 'epoch'.
        :param auto_log:                Whether or not to enable auto logging for logging the context parameters and
                                        trying to track common static and dynamic hyperparameters such as learning rate.

        :raise MLRunInvalidArgumentError: In case both 'context' and 'tensorboard_directory' parameters were not given
                                          or the 'update_frequency' was incorrect.
        """
        super(TensorboardLoggingCallback, self).__init__(
            dynamic_hyperparameters=dynamic_hyperparameters,
            static_hyperparameters=static_hyperparameters,
            auto_log=auto_log,
        )

        # Replace the logger with a TensorboardLogger:
        del self._logger
        self._logger = _TFKerasTensorboardLogger(
            statistics_functions=(
                statistics_functions
                if statistics_functions is not None
                else self.get_default_weight_statistics_list()
            ),
            context=context,
            tensorboard_directory=tensorboard_directory,
            run_name=run_name,
            update_frequency=update_frequency,
        )

        # Save the configurations:
        self._tracked_weights = weights

        # Initialize flags:
        self._logged_model = False
        self._logged_hyperparameters = False

    def get_weights(self) -> Dict[str, Variable]:
        """
        Get the weights tensors tracked. The weights will be stored in a dictionary where each key is the weight's name
        and the value is the weight's parameter (tensor).

        :return: The weights.
        """
        return self._logger.weights

    def get_weights_statistics(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Get the weights mean results logged. The results will be stored in a dictionary where each key is the weight's
        name and the value is a list of mean values per epoch.

        :return: The weights mean results.
        """
        return self._logger.weight_statistics

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

        # Write the initial summary of the run:
        self._logger.write_initial_summary_text()

        # Write the model's graph:
        self._logger.write_model_to_tensorboard(model=self.model)

        # Write the initial weights (epoch 0):
        self._logger.write_weights_statistics()
        self._logger.write_weights_histograms()
        self._logger.write_weights_images()

        # Make sure all values were written to the directory logs:
        self._logger.flush()

    def on_train_end(self, logs: dict = None):
        """
        Called at the end of training, wrapping up the tensorboard logging session.

        :param logs: Currently the output of the last call to `on_epoch_end()` is passed to this argument for this
                     method but that may change in the future.
        """
        super(TensorboardLoggingCallback, self).on_train_end()

        # Write the final run summary:
        self._logger.write_final_summary_text()

        # Close the logger:
        self._logger.close()

    def on_test_begin(self, logs: dict = None):
        """
        Called at the beginning of evaluation or validation. Will be called on each epoch according to the validation
        per epoch configuration. In case it is an evaluation, the epoch 0 will be logged.

        :param logs: Currently no data is passed to this argument for this method but that may change in the
                     future.
        """
        # Check if needed to mark this run as evaluation:
        if self._is_training is None:
            self._is_training = False

        # If this callback is part of evaluation and not training, need to check if the run was setup:
        if not self._is_training:
            # Start the tensorboard logger:
            self._logger.open()
            # Setup the run, logging relevant information and tracking weights:
            self._setup_run()
            # Write the initial summary of the run:
            self._logger.write_initial_summary_text()
            # Write the model's graph:
            self._logger.write_model_to_tensorboard(model=self.model)
            # Write the initial data (epoch 0):
            self._logger.write_weights_statistics()
            self._logger.write_weights_histograms()
            self._logger.write_weights_images()
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
            # Write the remaining epoch iterations results:
            self._logger.write_validation_results(ignore_update_frequency=True)
            # Write the epoch loss and metrics summaries to their graphs:
            self._logger.write_validation_summaries()
            # Write the final run summary:
            self._logger.write_final_summary_text()
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

        # Log the weights statistics:
        self._logger.log_weights_statistics()

        # Write the remaining epoch iterations results:
        self._logger.write_training_results(ignore_update_frequency=True)
        self._logger.write_validation_results(ignore_update_frequency=True)

        # Write the epoch text summary:
        self._logger.write_epoch_summary_text()

        # Write the epoch loss and metrics summaries to their graphs:
        self._logger.write_training_summaries()
        self._logger.write_validation_summaries()

        # Write the epoch dynamic hyperparameters values to their graphs:
        self._logger.write_dynamic_hyperparameters()

        # Write the weight histograms, images and statistics for all the tracked weights:
        self._logger.write_weights_statistics()
        self._logger.write_weights_histograms()
        self._logger.write_weights_images()

        # Make sure all values were written to the directory logs:
        self._logger.flush()

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

        # Write the batch loss and metrics results to their graphs:
        self._logger.write_training_results()

        # Check if needed to write the hyperparameters:
        if not self._logged_hyperparameters:
            self._logger.write_parameters_table_to_tensorboard()
            self._logged_hyperparameters = True
            self._logger.write_dynamic_hyperparameters()

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

        # Write the batch loss and metrics results to their graphs:
        self._logger.write_validation_results()

        # Check if needed to write the hyperparameters:
        if not self._logged_hyperparameters:
            self._logger.write_parameters_table_to_tensorboard()
            self._logged_hyperparameters = True
            self._logger.write_dynamic_hyperparameters()

    @staticmethod
    def get_default_weight_statistics_list() -> List[
        Callable[[Union[Variable, Tensor]], Union[float, Tensor]]
    ]:
        """
        Get the default list of statistics functions being applied on the tracked weights each epoch.

        :return: The default statistics functions list.
        """
        return [tf.math.reduce_mean, tf.math.reduce_std]

    def _setup_run(self):
        """
        After the trainer / evaluator run begins, this method will be called to setup the results, hyperparameters
        and weights dictionaries for logging.
        """
        super(TensorboardLoggingCallback, self)._setup_run()

        # Check if needed to track weights:
        if self._tracked_weights is False:
            return

        # Collect the weights for drawing histograms according to the stored configuration:
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

        # Log the initial (epoch 0) weights statistics:
        self._logger.log_weights_statistics()
