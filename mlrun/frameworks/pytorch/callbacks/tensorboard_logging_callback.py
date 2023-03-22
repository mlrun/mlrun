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
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

import mlrun

from ..._dl_common.loggers import TensorboardLogger
from ..utils import PyTorchTypes
from .logging_callback import LoggingCallback


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
                                      to be slower. Default: 'epoch'.
        """
        super(_PyTorchTensorboardLogger, self).__init__(
            statistics_functions=statistics_functions,
            context=context,
            tensorboard_directory=tensorboard_directory,
            run_name=run_name,
            update_frequency=update_frequency,
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

    def write_model_to_tensorboard(self, model: Module, input_sample: Tensor):
        """
        Write the given model as a graph in tensorboard.

        :param model:        The model to write.
        :param input_sample: An input sample for writing the model.
        """
        self._summary_writer.add_graph(
            model=model,
            input_to_model=input_sample,
        )

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

        # Prepare the hyperparameters values:
        non_graph_parameters = {"Date": str(datetime.now()).split(".")[0]}
        for parameter, value in self._static_hyperparameters.items():
            non_graph_parameters[parameter] = value

        # Prepare the summaries values and the dynamic hyperparameters values:
        graph_parameters = {}
        for metric in self._training_summaries:
            graph_parameters[f"{self._Sections.SUMMARY}/training_{metric}"] = 0.0
        for metric in self._validation_summaries:
            graph_parameters[f"{self._Sections.SUMMARY}/validation_{metric}"] = 0.0
        for parameter, epochs in self._dynamic_hyperparameters.items():
            graph_parameters[f"{self._Sections.HYPERPARAMETERS}/{parameter}"] = epochs[
                -1
            ]

        # Write the hyperparameters and summaries table:
        self._summary_writer.add_hparams(non_graph_parameters, graph_parameters)

    def flush(self):
        """
        Make sure all values were written to the directory logs so it will be available live.
        """
        self._summary_writer.flush()

    def _write_text_to_tensorboard(self, tag: str, text: str, step: int):
        """
        Write text to tensorboard's text section. Summary information of this training / validation run will be logged
        to tensorboard using this method.

        :param tag:  The tag of the text (box it will be appearing under).
        :param text: The text to write.
        :param step: The iteration / epoch the text belongs to.
        """
        self._summary_writer.add_text(
            tag=tag,
            text_string=text,
            global_step=step,
        )

    def _write_scalar_to_tensorboard(self, name: str, value: float, step: int):
        """
        Write the scalar's value into its plot.

        :param name:  The plot's name.
        :param value: The value to add to the plot.
        :param step:  The iteration / epoch the value belongs to.
        """
        self._summary_writer.add_scalar(
            tag=name,
            scalar_value=value,
            global_step=step,
        )

    def _write_weight_histogram_to_tensorboard(
        self, name: str, weight: Parameter, step: int
    ):
        """
        Write the current state of the weights as histograms to tensorboard.

        :param name:   The weight's name.
        :param weight: The weight to write its histogram.
        :param step:   The iteration / epoch the weight's histogram state belongs to.
        """
        self._summary_writer.add_histogram(
            tag=name,
            values=weight,
            global_step=step,
        )

    def _write_weight_image_to_tensorboard(
        self, name: str, weight: Parameter, step: int
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
            Callable[[Union[Parameter, Tensor]], Union[float, Tensor]]
        ] = None,
        dynamic_hyperparameters: Dict[
            str,
            Tuple[
                str,
                Union[List[Union[str, int]], Callable[[], PyTorchTypes.TrackableType]],
            ],
        ] = None,
        static_hyperparameters: Dict[
            str, Union[PyTorchTypes.TrackableType, Tuple[str, List[Union[str, int]]]]
        ] = None,
        update_frequency: Union[int, str] = "epoch",
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
                                        Default: False.
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
                                            "lr": (HyperparametersKeys.OPTIMIZER, ["param_groups", 0, "lr"]),
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
        :param update_frequency:        Per how many iterations (batches) the callback should write the tracked values
                                        to tensorboard. Can be passed as a string equal to 'epoch' for per epoch and
                                        'batch' for per single batch, or as an integer specifying per how many
                                        iterations to update. Notice that writing to tensorboard too frequently may
                                        cause the training to be slower. Default: 'epoch'.
        :param auto_log:                Whether or not to enable auto logging, trying to track common static and dynamic
                                        hyperparameters.

        :raise MLRunInvalidArgumentError: In case both 'context' and 'tensorboard_directory' parameters were not given
                                          or the 'update_frequency' was incorrect.
        """
        super(TensorboardLoggingCallback, self).__init__(
            dynamic_hyperparameters=dynamic_hyperparameters,
            static_hyperparameters=static_hyperparameters,
            auto_log=auto_log,
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
            update_frequency=update_frequency,
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

    def get_weights_statistics(self) -> Dict[str, Dict[str, List[float]]]:
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
        metric_functions: List[PyTorchTypes.MetricFunctionType] = None,
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

        # Log the weights statistics:
        self._logger.log_weights_statistics()

    def on_run_begin(self):
        """
        After the run begins, this method will be called to setup the weights, results and hyperparameters dictionaries
        for logging. Epoch 0 (pre-run state) will be logged here.
        """
        # Setup all the results and hyperparameters dictionaries:
        super(TensorboardLoggingCallback, self).on_run_begin()

        # Log the initial summary of the run:
        self._logger.write_initial_summary_text()

        # Log the model:
        self._logger.write_model_to_tensorboard(
            model=self._objects[self._ObjectKeys.MODEL],
            input_sample=next(self._objects[self._ObjectKeys.TRAINING_SET].__iter__())[
                0
            ],
        )

        # Write the hyperparameters table:
        self._logger.write_parameters_table_to_tensorboard()

        # Write the initial dynamic hyperparameters values:
        self._logger.write_dynamic_hyperparameters()

        # Write the initial weights data:
        self._logger.write_weights_statistics()
        self._logger.write_weights_histograms()
        self._logger.write_weights_images()

    def on_run_end(self):
        """
        Before the run ends, this method will be called to log the context summary.
        """
        # Write the final summary of the run:
        self._logger.write_final_summary_text()

        super(TensorboardLoggingCallback, self).on_run_end()

    def on_validation_end(
        self, loss_value: PyTorchTypes.MetricValueType, metric_values: List[float]
    ):
        """
        Before the validation (in a training case it will be per epoch) ends, this method will be called to log the
        validation results summaries.

        :param loss_value:    The loss summary of this validation.
        :param metric_values: The metrics summaries of this validation.
        """
        super(TensorboardLoggingCallback, self).on_validation_end(
            loss_value=loss_value, metric_values=metric_values
        )

        # Check if this run was part of an evaluation:
        if not self._is_training:
            # Write the remaining epoch iterations results:
            self._logger.write_validation_results(ignore_update_frequency=True)
            # Write the epoch loss and metrics summaries to their graphs:
            self._logger.write_validation_summaries()
            # Make sure all values were written to the directory logs:
            self._logger.flush()

    def on_epoch_end(self, epoch: int):
        """
        Before the given epoch ends, this method will be called to log the dynamic hyperparameters as needed. All of the
        per epoch plots (loss and metrics summaries, dynamic hyperparameters, weights histograms and statistics) will
        log this epoch's tracked values.

        :param epoch: The epoch that has just ended.
        """
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

    def on_train_batch_end(self, batch: int, x: Tensor, y_true: Tensor, y_pred: Tensor):
        """
        Before the training of the given batch ends, this method will be called to log the batch's loss and metrics
        results to their per iteration plots.

        :param batch:  The current batch iteration of when this method is called.
        :param x:      The input part of the current batch.
        :param y_true: The true value part of the current batch.
        :param y_pred: The prediction (output) of the model for this batch's input ('x').
        """
        super(TensorboardLoggingCallback, self).on_train_batch_end(
            batch=batch, x=x, y_true=y_true, y_pred=y_pred
        )

        # Write the batch loss and metrics results to their graphs:
        self._logger.write_training_results()

    def on_validation_batch_end(
        self, batch: int, x: Tensor, y_true: Tensor, y_pred: Tensor
    ):
        """
        Before the validation of the given batch ends, this method will be called to log the batch's loss and metrics
        results to their per iteration plots.

        :param batch:  The current batch iteration of when this method is called.
        :param x:      The input part of the current batch.
        :param y_true: The true value part of the current batch.
        :param y_pred: The prediction (output) of the model for this batch's input ('x').
        """
        super(TensorboardLoggingCallback, self).on_validation_batch_end(
            batch=batch, x=x, y_true=y_true, y_pred=y_pred
        )

        # Write the batch loss and metrics results to their graphs:
        self._logger.write_validation_results()
