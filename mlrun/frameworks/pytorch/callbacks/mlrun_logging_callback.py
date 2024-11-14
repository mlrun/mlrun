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
from typing import Callable, Optional, Union

import torch
from torch import Tensor

import mlrun
from mlrun.artifacts import Artifact

from ..._common import LoggingMode
from ..._dl_common.loggers import MLRunLogger
from ..model_handler import PyTorchModelHandler
from ..utils import PyTorchTypes
from .logging_callback import LoggingCallback


class MLRunLoggingCallback(LoggingCallback):
    """
    Callback for logging data during training / validation via mlrun's context. Each tracked hyperparameter and metrics
    results will be logged per epoch and at the end of the run the model will be saved and logged as well. Some plots
    will be available as well. To summarize, the available data in mlrun will be:

    * For each epoch:

      * Tracking table: epoch, static hyperparameters, dynamic hyperparameters, training metrics, validation metrics.
      * Per iteration (batch) chart artifacts for the training and validation metrics.

    * At the end of the run:

      * Per epoch chart artifacts for the validation summaries and dynamic hyperparameters.
      * Model is logged with all of the files and artifacts.

    All the collected data will be available in this callback post the training / validation process and can be accessed
    via the 'training_results', 'validation_results', 'static_hyperparameters', 'dynamic_hyperparameters' and
    'summaries' properties.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx,
        model_handler: PyTorchModelHandler,
        log_model_tag: str = "",
        log_model_labels: Optional[dict[str, PyTorchTypes.TrackableType]] = None,
        log_model_parameters: Optional[dict[str, PyTorchTypes.TrackableType]] = None,
        log_model_extra_data: Optional[
            dict[str, Union[PyTorchTypes.TrackableType, Artifact]]
        ] = None,
        dynamic_hyperparameters: Optional[
            dict[
                str,
                tuple[
                    str,
                    Union[
                        list[Union[str, int]], Callable[[], PyTorchTypes.TrackableType]
                    ],
                ],
            ]
        ] = None,
        static_hyperparameters: Optional[
            dict[
                str,
                Union[PyTorchTypes.TrackableType, tuple[str, list[Union[str, int]]]],
            ]
        ] = None,
        auto_log: bool = False,
    ):
        """
        Initialize an mlrun logging callback with the given hyperparameters and logging configurations. Notice: In order
        to log the model, its class (torch.Module) must be in the custom objects map or the modules map.

        :param context:                  MLRun context to log to. Its parameters will be logged automatically  if
                                         'auto_log' is True.
        :param model_handler:            The model handler to use for logging the model at the end of the run with the
                                         collected logs.
        :param log_model_tag:            Version tag to give the logged model.
        :param log_model_labels:         Labels to log with the model.
        :param log_model_parameters:     Parameters to log with the model.
        :param log_model_extra_data:     Extra data to log with the model.
        :param dynamic_hyperparameters:  If needed to track a hyperparameter dynamically (sample it each epoch) it
                                         should be passed here. The parameter expects a dictionary where the keys are
                                         the hyperparameter chosen names and the values are tuples of object key and a
                                         list with the key chain. A key chain is a list of keys and indices to know how
                                         to access the needed hyperparameter. If the hyperparameter is not of accessible
                                         from any of the HyperparametersKeys, a custom callable method can be passed in
                                         the tuple instead of the key chain when providing the word
                                         HyperparametersKeys.CUSTOM. For example, to track the 'lr' attribute of
                                         an optimizer and a custom parameter, one should pass:
                                         {
                                             "lr": (HyperparametersKeys.OPTIMIZER, ["param_groups", 0, "lr"]),
                                             "custom parameter": (HyperparametersKeys.CUSTOM, get_custom_parameter)
                                         }
        :param static_hyperparameters:   If needed to track a hyperparameter one time per run it should be passed here.
                                         The parameter expects a dictionary where the keys are the
                                         hyperparameter chosen names and the values are the hyperparameter static value
                                         or a tuple of object key and a list with the key chain just like the dynamic
                                         hyperparameter. For example, to track the 'epochs' of an experiment run, one
                                         should pass:
                                         {
                                             "epochs": 7
                                         }
        :param auto_log:                 Whether or not to enable auto logging for logging the context parameters and
                                         trying to track common static and dynamic hyperparameters.
        """
        super().__init__(
            dynamic_hyperparameters=dynamic_hyperparameters,
            static_hyperparameters=static_hyperparameters,
            auto_log=auto_log,
        )

        # Replace the logger with an MLRunLogger:
        del self._logger
        self._logger = MLRunLogger(context=context)

        # Store the given handler:
        self._model_handler = model_handler

        # Store the attributes to log along the model:
        self._log_model_tag = log_model_tag
        self._log_model_labels = log_model_labels
        self._log_model_parameters = log_model_parameters
        self._log_model_extra_data = log_model_extra_data

        # Setup the additional PyTorchModelHandler parameters for logging the model later:
        self._input_sample = None  # type: PyTorchModelHandler.IOSample
        self._output_sample = None  # type: PyTorchModelHandler.IOSample

    def on_run_end(self):
        """
        Before the run ends, this method will be called to log the model and the run summaries charts.
        """
        # Check if the logger is in evaluation mode, if so, log the last epoch
        if self._logger.mode == LoggingMode.EVALUATION:
            self._logger.log_epoch_to_context(epoch=1)

        # Set the inputs and outputs:
        if self._model_handler.inputs is None:
            self._model_handler.set_inputs(from_sample=self._input_sample)
        if self._model_handler.outputs is None:
            self._model_handler.set_outputs(from_sample=self._output_sample)

        # End the run:
        self._logger.log_run(
            model_handler=self._model_handler,
            tag=self._log_model_tag,
            labels=self._log_model_labels,
            parameters=self._log_model_parameters,
            extra_data=self._log_model_extra_data,
        )

    def on_epoch_end(self, epoch: int):
        """
        Before the given epoch ends, this method will be called to log the dynamic hyperparameters and results of this
        epoch via the stored context.

        :param epoch: The epoch that has just ended.
        """
        super().on_epoch_end(epoch=epoch)

        # Create child context to hold the current epoch's results:
        self._logger.log_epoch_to_context(epoch=epoch)

    def on_inference_begin(self, x: Tensor):
        """
        Before the inference of the current batch sample into the model, this method will be called to save an input
        sample - a zeros tensor with the same properties of the 'x' input.

        :param x: The input of the current batch.
        """
        if self._input_sample is None:
            self._input_sample = torch.zeros(size=x.shape, dtype=x.dtype)

    def on_inference_end(self, y_pred: Tensor, y_true: Tensor):
        """
        After the inference of the current batch sample, this method will be called to save an output sample - a zeros
        tensor with the same properties of the 'y_pred' output.

        :param y_pred: The prediction (output) of the model for this batch's input ('x').
        :param y_true: The ground truth value of the current batch.
        """
        if self._output_sample is None:
            self._output_sample = torch.zeros(size=y_pred.shape, dtype=y_pred.dtype)
