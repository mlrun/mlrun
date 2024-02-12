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

import mlrun

from ..._common import LoggingMode
from ..utils import DLTypes


class Logger:
    """
    Logger for tracking hyperparamters and metrics results during training / evaluation of some framework.
    """

    def __init__(self, context: mlrun.MLClientCtx = None):
        """
        Initialize a generic logger for collecting training / validation runs data.

        :param context: MLRun context to log its parameters as static hyperparameters if needed.
        """
        # Save the context:
        self._context = context

        # Set up the logger's mode (default:  Training):
        self._mode = LoggingMode.TRAINING

        # Set up the results dictionaries - a dictionary of metrics for all the iteration results by their epochs:
        # [Metric: str] -> [Epoch: int] -> [Iteration: int] -> [value: float]
        self._training_results = {}  # type: Dict[str, List[List[float]]]
        self._validation_results = {}  # type: Dict[str, List[List[float]]]

        # Set up the metrics summary dictionaries - a dictionary of all metrics averages by epochs:
        # [Metric: str] -> [Epoch: int] -> [value: float]:
        self._training_summaries = {}  # type: Dict[str, List[float]]
        self._validation_summaries = {}  # type: Dict[str, List[float]]

        # Store the static hyperparameters given - a dictionary of parameters and their values to note:
        # [Parameter: str] -> [value: Union[str, bool, float, int]]
        self._static_hyperparameters = {}  # type: Dict[str, DLTypes.TrackableType]

        # Setup the dynamic hyperparameters dictionary - a dictionary of all tracked hyperparameters by epochs:
        # [Hyperparameter: str] -> [Epoch: int] -> [value: Union[str, bool, float, int]]
        self._dynamic_hyperparameters = {}  # type: Dict[str, List[DLTypes.TrackableType]]

        # Setup the iterations counter:
        self._epochs = 0
        self._training_iterations = 0
        self._validation_iterations = 0

    @property
    def context(self) -> mlrun.MLClientCtx:
        """
        Get the loggers context. If its not set, None will be returned.

        :return: The logger's context.
        """
        return self._context

    @property
    def mode(self) -> LoggingMode:
        """
        Get the logger's mode.

        :return: The logger's mode. One of Logger.LoggerMode.
        """
        return self._mode

    @property
    def training_results(self) -> dict[str, list[list[float]]]:
        """
        Get the training results logged. The results will be stored in a dictionary where each key is the metric name
        and the value is a list of lists of values. The first list is by epoch and the second list is by iteration
        (batch).

        :return: The training results.
        """
        return self._training_results

    @property
    def validation_results(self) -> dict[str, list[list[float]]]:
        """
        Get the validation results logged. The results will be stored in a dictionary where each key is the metric name
        and the value is a list of lists of values. The first list is by epoch and the second list is by iteration
        (batch).

        :return: The validation results.
        """
        return self._validation_results

    @property
    def training_summaries(self) -> dict[str, list[float]]:
        """
        Get the training summaries of the metrics results. The summaries will be stored in a dictionary where each key
        is the metric names and the value is a list of all the summary values per epoch.

        :return: The training summaries.
        """
        return self._training_summaries

    @property
    def validation_summaries(self) -> dict[str, list[float]]:
        """
        Get the validation summaries of the metrics results. The summaries will be stored in a dictionary where each key
        is the metric names and the value is a list of all the summary values per epoch.

        :return: The validation summaries.
        """
        return self._validation_summaries

    @property
    def static_hyperparameters(self) -> dict[str, DLTypes.TrackableType]:
        """
        Get the static hyperparameters logged. The hyperparameters will be stored in a dictionary where each key is the
        hyperparameter name and the value is his logged value.

        :return: The static hyperparameters.
        """
        return self._static_hyperparameters

    @property
    def dynamic_hyperparameters(self) -> dict[str, list[DLTypes.TrackableType]]:
        """
        Get the dynamic hyperparameters logged. The hyperparameters will be stored in a dictionary where each key is the
        hyperparameter name and the value is a list of his logged values per epoch.

        :return: The dynamic hyperparameters.
        """
        return self._dynamic_hyperparameters

    @property
    def epochs(self) -> int:
        """
        Get the overall epochs.

        :return: The overall epochs.
        """
        return self._epochs

    @property
    def training_iterations(self) -> int:
        """
        Get the overall training iterations.

        :return: The overall training iterations.
        """
        return self._training_iterations

    @property
    def validation_iterations(self) -> int:
        """
        Get the overall validation iterations.

        :return: The overall validation iterations.
        """
        return self._validation_iterations

    def set_mode(self, mode: LoggingMode):
        """
        Set the logger's mode.

        :param mode: The mode to set. One of Logger.LoggerMode.
        """
        self._mode = mode

    def log_epoch(self):
        """
        Log a new epoch, appending all the result with a new list for the new epoch.
        """
        # Count the new epoch:
        self._epochs += 1

        # Add a new epoch to each of the metrics in the results dictionary:
        for results_dictionary in [self._training_results, self._validation_results]:
            for metric in results_dictionary:
                results_dictionary[metric].append([])

    def log_training_iteration(self):
        """
        Log a new training iteration.
        """
        self._training_iterations += 1

    def log_validation_iteration(self):
        """
        Log a new validation iteration.
        """
        self._validation_iterations += 1

    def log_training_result(self, metric_name: str, result: float):
        """
        Log the given metric result in the training results dictionary at the current epoch.

        :param metric_name: The metric name as it was logged in 'log_metric'.
        :param result:      The metric result to log.
        """
        if metric_name not in self._training_results:
            self._training_results[metric_name] = [[]]
        self._training_results[metric_name][-1].append(result)

    def log_validation_result(self, metric_name: str, result: float):
        """
        Log the given metric result in the validation results dictionary at the current epoch.

        :param metric_name: The metric name as it was logged in 'log_metric'.
        :param result:      The metric result to log.
        """
        if metric_name not in self._validation_results:
            self._validation_results[metric_name] = [[]]
        self._validation_results[metric_name][-1].append(result)

    def log_training_summary(self, metric_name: str, result: float):
        """
        Log the given metric result in the training summaries results dictionary.

        :param metric_name: The metric name as it was logged in 'log_metric'.
        :param result:      The metric result to log.
        """
        if metric_name not in self._training_summaries:
            self._training_summaries[metric_name] = []
        self._training_summaries[metric_name].append(result)

    def log_validation_summary(self, metric_name: str, result: float):
        """
        Log the given metric result in the validation summaries results dictionary.

        :param metric_name: The metric name as it was logged in 'log_metric'.
        :param result:      The metric result to log.
        """
        if metric_name not in self._validation_summaries:
            self._validation_summaries[metric_name] = []
        self._validation_summaries[metric_name].append(result)

    def log_static_hyperparameter(
        self, parameter_name: str, value: DLTypes.TrackableType
    ):
        """
        Log the given parameter value in the static hyperparameters dictionary.

        :param parameter_name: The parameter name.
        :param value:          The parameter value to log.
        """
        self._static_hyperparameters[parameter_name] = value

    def log_dynamic_hyperparameter(
        self, parameter_name: str, value: DLTypes.TrackableType
    ):
        """
        Log the given parameter value in the dynamic hyperparameters dictionary at the current epoch (if its a new
        parameter it will be epoch 0). If the parameter appears in the static hyperparameters dictionary, it will be
        removed from there as it is now dynamic.

        :param parameter_name: The parameter name.
        :param value:          The parameter value to log.
        """
        # Check if its a new hyperparameter being tracked:
        if parameter_name not in self._dynamic_hyperparameters:
            # Look in the static hyperparameters:
            if parameter_name in self._static_hyperparameters:
                self._static_hyperparameters.pop(parameter_name)
            # Add it as a dynamic hyperparameter:
            self._dynamic_hyperparameters[parameter_name] = [value]
        else:
            self._dynamic_hyperparameters[parameter_name].append(value)

    # TODO: Move to MLRun logger
    def log_context_parameters(self):
        """
        Log the context given parameters as static hyperparameters. Should be called once as the context parameters do
        not change.
        """
        for parameter_name, parameter_value in self._context.parameters.items():
            # Check if the parameter is a trackable value:
            if isinstance(parameter_value, (str, bool, float, int)):
                self.log_static_hyperparameter(
                    parameter_name=parameter_name, value=parameter_value
                )
            else:
                # See if its string representation length is below the maximum value length:
                string_value = str(parameter_value)
                if (
                    len(string_value) < 30
                ):  # Temporary to no log to long variables into the UI.
                    # TODO: Make the user specify the parameters and take them all by default.
                    self.log_static_hyperparameter(
                        parameter_name=parameter_name, value=parameter_value
                    )
