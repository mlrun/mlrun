from typing import List, Tuple, Dict, Union, Callable
from datetime import datetime

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

import mlrun
from mlrun.frameworks._common.loggers import TensorboardLogger, TrackableType
from mlrun.frameworks.pytorch.callbacks.logging_callback import (
    LoggingCallback,
    MetricFunctionType,
)


class _MLRunSummaryWriter(SummaryWriter):
    """
    A slightly edited torch's SummaryWriter class to overcome the hyperparameter logging problem (creating a new event
    per call).
    """

    def add_hparams(
        self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None
    ):
        """
        Log the given hyperparameters to the same event file that is currently open.

        :param hparam_dict:            The static hyperparameters to simply log to the 'hparams' table.
        :param metric_dict:            The metrics and dynamic hyper parameters to link with the plots.
        :param hparam_domain_discrete: Not used in this SummaryWriter.
        :param run_name:               Not used in this SummaryWriter.
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)
        self._get_file_writer().add_summary(exp)
        self._get_file_writer().add_summary(ssi)
        self._get_file_writer().add_summary(sei)


class _PyTorchTensorboardLogger(TensorboardLogger):
    """
    The PyTorch framework implementation of the 'TensorboardLogger'.
    """

    def __init__(
        self,
        statistics_functions: List[
            Callable[[Union[Parameter]], Union[float, Parameter]]
        ],
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
        super(_PyTorchTensorboardLogger, self).__init__(
            statistics_functions=statistics_functions,
            context=context,
            tensorboard_directory=tensorboard_directory,
            run_name=run_name,
        )

        # Setup the tensorboard writer property:
        self._summary_writer = None

    def open(self):
        """
        Create the output path and initialize the tensorboard file writer.
        """
        # Create the output path:
        self._create_output_path()

        # Use the output path to initialize the tensorboard writer:
        self._summary_writer = _MLRunSummaryWriter(log_dir=self._output_path)

    def log_run_start_text_to_tensorboard(self):
        """
        Log the initial information summary of this training / validation run to tensorboard.
        """
        self._summary_writer.add_text(
            tag="MLRun", text_string=self._generate_run_start_text(), global_step=0,
        )

    def log_epoch_text_to_tensorboard(self):
        """
        Log the last epoch summary of this training run to tensorboard.
        """
        self._summary_writer.add_text(
            tag="MLRun",
            text_string=self._generate_epoch_text(),
            global_step=self._training_iterations,
        )

    def log_run_end_text_to_tensorboard(self):
        """
        Log the final information summary of this training / validation run to tensorboard.
        """
        self._summary_writer.add_text(
            tag="MLRun",
            text_string=self._generate_run_end_text(),
            global_step=(
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

        # Prepare the hyperparameters values:
        non_graph_parameters = {"Date": str(datetime.now()).split(".")[0]}
        for parameter, value in self._static_hyperparameters.items():
            non_graph_parameters[parameter] = value

        # Prepare the summaries values and the dynamic hyperparameters values:
        graph_parameters = {}
        for metric in self._training_summaries:
            graph_parameters[
                "{}/training_{}".format(self._Sections.SUMMARY, metric)
            ] = 0.0
        for metric in self._validation_summaries:
            graph_parameters[
                "{}/validation_{}".format(self._Sections.SUMMARY, metric)
            ] = 0.0
        for parameter, epochs in self._dynamic_hyperparameters.items():
            graph_parameters[
                "{}/{}".format(self._Sections.HYPERPARAMETERS, parameter)
            ] = epochs[-1]

        # Write the hyperparameters and summaries table:
        self._summary_writer.add_hparams(non_graph_parameters, graph_parameters)

    def log_training_results_to_tensorboard(self):
        """
        Log the recent training iteration metrics results to tensorboard.
        """
        for metric, epochs in self._training_results.items():
            self._summary_writer.add_scalar(
                tag="{}/{}".format(self._Sections.TRAINING, metric),
                scalar_value=epochs[-1][-1],
                global_step=self._training_iterations,
            )

    def log_validation_results_to_tensorboard(self):
        """
        Log the recent validation iteration metrics results to tensorboard.
        """
        for metric, epochs in self._validation_results.items():
            self._summary_writer.add_scalar(
                tag="{}/{}".format(self._Sections.VALIDATION, metric),
                scalar_value=epochs[-1][-1],
                global_step=self._validation_iterations,
            )

    def log_dynamic_hyperparameters_to_tensorboard(self):
        """
        Log the recent epoch dynamic hyperparameters values to tensorboard.
        """
        for parameter, epochs in self._dynamic_hyperparameters.items():
            self._summary_writer.add_scalar(
                tag="{}/{}".format(self._Sections.HYPERPARAMETERS, parameter),
                scalar_value=epochs[-1],
                global_step=self._epochs,
            )

    def log_summaries_to_tensorboard(self):
        """
        Log the recent epoch summaries results to tensorboard.
        """
        for prefix, summaries in zip(
            ["training", "validation"],
            [self._training_summaries, self._validation_summaries],
        ):
            for metric, epochs in summaries.items():
                self._summary_writer.add_scalar(
                    tag="{}/{}_{}".format(self._Sections.SUMMARY, prefix, metric),
                    scalar_value=epochs[-1],
                    global_step=self._epochs,
                )

    def log_weights_histograms_to_tensorboard(self):
        """
        Log the current state of the weights as histograms to tensorboard.
        """
        for weight_name, weight_parameter in self._weights.items():
            self._summary_writer.add_histogram(
                tag="{}/{}".format(self._Sections.WEIGHTS, weight_name),
                values=weight_parameter,
                global_step=self._epochs,
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
                self._summary_writer.add_scalar(
                    tag="{}/{}:{}".format(
                        self._Sections.WEIGHTS, weight_name, statistic
                    ),
                    scalar_value=epoch_values[-1],
                    global_step=self._epochs,
                )

    def log_model_to_tensorboard(self, model: Module, input_sample: Tensor):
        """
        Log the given model as a graph in tensorboard.
        """
        self._summary_writer.add_graph(
            model=model, input_to_model=input_sample,
        )

    def flush(self):
        """
        Make sure all values were written to the directory logs so it will be available live.
        """
        self._summary_writer.flush()


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
            Callable[[Union[Parameter, Tensor]], Union[float, Tensor]]
        ] = None,
        dynamic_hyperparameters: Dict[
            str, Tuple[str, Union[List[Union[str, int]], Callable[[], TrackableType]]]
        ] = None,
        static_hyperparameters: Dict[
            str, Union[TrackableType, Tuple[str, List[Union[str, int]]]]
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
                "Expecting to receive a mlrun.MLClientCtx context or a path to a directory to output the logging file "
                "but None were given."
            )

        # Replace the logger with an MLRunLogger:
        del self._logger
        self._logger = _PyTorchTensorboardLogger(
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

    def get_weights(self) -> Dict[str, Parameter]:
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
        Callable[[Union[Parameter, Tensor]], Union[float, Tensor]]
    ]:
        """
        Get the default list of statistics functions being applied on the tracked weights each epoch.

        :return: The default statistics functions list.
        """
        return [torch.mean, torch.std]

    def on_setup(
        self,
        model: Module = None,
        training_set: DataLoader = None,
        validation_set: DataLoader = None,
        loss_function: Module = None,
        optimizer: Optimizer = None,
        metric_functions: List[MetricFunctionType] = None,
        scheduler=None,
    ):
        """
        Storing all the given objects in the callback's objects dictionary and load the weights parameters as given in
        the callback's initialization.

        :param model:            The model to be stored in this callback.
        :param training_set:     The training set to be stored in this callback.
        :param validation_set:   The validation set to be stored in this callback.
        :param loss_function:    The loss function to be stored in this callback.
        :param optimizer:        The optimizer to be stored in this callback.
        :param metric_functions: The metric functions to be stored in this callback.
        :param scheduler:        The scheduler to be stored in this callback.
        """
        super(TensorboardLoggingCallback, self).on_setup(
            model=model,
            training_set=training_set,
            validation_set=validation_set,
            loss_function=loss_function,
            optimizer=optimizer,
            metric_functions=metric_functions,
            scheduler=scheduler,
        )

        # Start the tensorboard logger:
        self._logger.open()

        # Collect the weights for drawing histograms according to the stored configuration:
        if self._tracked_weights is False:
            return

        # Log the weights:
        for weight_name, weight_parameter in self._objects[
            self._ObjectKeys.MODEL
        ].named_parameters():
            collect = False
            if self._tracked_weights is True:  # Collect all weights
                collect = True
            else:
                for tag in self._tracked_weights:  # Collect by given name
                    if tag in weight_name:
                        collect = True
                        break
            if collect:
                self._logger.log_weight(
                    weight_name=weight_name, weight_holder=weight_parameter
                )

    def on_run_begin(self):
        """
        After the trainer / evaluator run begins, this method will be called to setup the weights, results and
        hyperparameters dictionaries for logging. Epoch 0 (pre-run state) will be logged here.
        """
        # Setup all the results and hyperparameters dictionaries:
        super(TensorboardLoggingCallback, self).on_run_begin()

        # Log the initial summary of the run:
        self._logger.log_run_start_text_to_tensorboard()

        # Log the model:
        self._logger.log_model_to_tensorboard(
            model=self._objects[self._ObjectKeys.MODEL],
            input_sample=next(self._objects[self._ObjectKeys.TRAINING_SET].__iter__())[
                0
            ],
        )

        # Log hyperparameters:
        self._logger.log_parameters_table_to_tensorboard()
        self._logger.log_dynamic_hyperparameters_to_tensorboard()

        # Log the initial weights:
        self._logger.log_weights_histograms_to_tensorboard()
        self._logger.log_weights_images_to_tensorboard()
        self._logger.log_weights_statistics()
        self._logger.log_statistics_to_tensorboard()

    def on_run_end(self):
        """
        Before the trainer / evaluator run ends, this method will be called to log the context summary.
        """
        # Log the final summary of the run:
        self._logger.log_run_end_text_to_tensorboard()

        super(TensorboardLoggingCallback, self).on_run_end()

    def on_epoch_end(self, epoch: int):
        """
        Before the trainer given epoch ends, this method will be called to log the dynamic hyperparameters as needed.
        All of the per epoch plots (loss and metrics summaries, dynamic hyperparameters, weights histograms and
        statistics) will log this epoch's tracked values.

        :param epoch: The epoch that has just ended.
        """
        super(TensorboardLoggingCallback, self).on_epoch_end(epoch=epoch)

        # Add this epoch text summary:
        self._logger.log_epoch_text_to_tensorboard()

        # Add this epoch loss and metrics averages to their graphs:
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

    def on_train_batch_end(self, batch: int, x: Tensor, y_true: Tensor, y_pred: Tensor):
        """
        Before the trainer training of the given batch ends, this method will be called to log the batch's loss and
        metrics results to their per iteration plots.

        :param batch:  The current batch iteration of when this method is called.
        :param x:      The input part of the current batch.
        :param y_true: The true value part of the current batch.
        :param y_pred: The prediction (output) of the model for this batch's input ('x').
        """
        super(TensorboardLoggingCallback, self).on_train_batch_end(
            batch=batch, x=x, y_true=y_true, y_pred=y_pred
        )

        # Add this batch loss and metrics results to their graphs:
        self._logger.log_training_results_to_tensorboard()

    def on_validation_batch_end(
        self, batch: int, x: Tensor, y_true: Tensor, y_pred: Tensor
    ):
        """
        Before the trainer / evaluator validation of the given batch ends, this method will be called to log the batch's
        loss and metrics results to their per iteration plots.

        :param batch:  The current batch iteration of when this method is called.
        :param x:      The input part of the current batch.
        :param y_true: The true value part of the current batch.
        :param y_pred: The prediction (output) of the model for this batch's input ('x').
        """
        super(TensorboardLoggingCallback, self).on_validation_batch_end(
            batch=batch, x=x, y_true=y_true, y_pred=y_pred
        )

        # Add this batch loss and metrics results to their graphs:
        self._logger.log_validation_results_to_tensorboard()
