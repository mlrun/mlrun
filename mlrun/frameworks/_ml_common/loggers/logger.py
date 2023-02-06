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
from typing import Dict, List

from ..utils import MLTypes


class Logger:
    """
    Logger for tracking hyperparamters and metrics results during training of some framework.
    """

    def __init__(self):
        """
        Initialize a generic logger for collecting training data.
        """
        # Set up the results dictionaries - a dictionary of metrics for all the iteration results by their epochs:
        # [Validation Set: str] -> [Metric: str] -> [Iteration: int] -> [value: float]
        self._results = {}  # type: Dict[str, Dict[str, List[float]]]

        # Store the static hyperparameters given - a dictionary of parameters and their values to note:
        # [Parameter: str] -> [value: Union[str, bool, float, int]]
        self._static_hyperparameters = {}  # type: Dict[str, MLTypes.TrackableType]

        # Set up the dynamic hyperparameters dictionary - a dictionary of all tracked hyperparameters by epochs:
        # [Hyperparameter: str] -> [Epoch: int] -> [value: Union[str, bool, float, int]]
        self._dynamic_hyperparameters = (
            {}
        )  # type: Dict[str, List[MLTypes.TrackableType]]

        # Set up the iterations counter:
        self._iterations = 0

    @property
    def results(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Get the results logged. The results will be stored in a dictionary where each key is the validation set name
        and the value is a dictionary of metrics to their list of iterations values.

        :return: The results.
        """
        return self._results

    @property
    def static_hyperparameters(self) -> Dict[str, MLTypes.TrackableType]:
        """
        Get the static hyperparameters logged. The hyperparameters will be stored in a dictionary where each key is the
        hyperparameter name and the value is his logged value.

        :return: The static hyperparameters.
        """
        return self._static_hyperparameters

    @property
    def dynamic_hyperparameters(self) -> Dict[str, List[MLTypes.TrackableType]]:
        """
        Get the dynamic hyperparameters logged. The hyperparameters will be stored in a dictionary where each key is the
        hyperparameter name and the value is a list of his logged values per epoch.

        :return: The dynamic hyperparameters.
        """
        return self._dynamic_hyperparameters

    @property
    def iterations(self) -> int:
        """
        Get the overall iterations.

        :return: The overall iterations.
        """
        return self._iterations

    def log_iteration(self):
        """
        Log a new iteration.
        """
        self._iterations += 1

    def log_result(self, validation_set_name: str, metric_name: str, result: float):
        """
        Log the given metric result in the training results dictionary at the current epoch.

        :param validation_set_name: Name of the validation set used.
        :param metric_name:         The metric name.
        :param result:              The metric result to log.
        """
        # Get the validation set's metrics (will set a new dictionary in case it's a new validation set):
        if validation_set_name not in self._results:
            self._results[validation_set_name] = {}
        validation_set_metrics = self._results[validation_set_name]

        # Get the metric's results list (will set a new list in case it's a new metric):
        if metric_name not in validation_set_metrics:
            validation_set_metrics[metric_name] = []
        metric_results = validation_set_metrics.setdefault(metric_name, [])

        # Log the metric's result:
        metric_results.append(result)

    def log_static_hyperparameter(
        self, parameter_name: str, value: MLTypes.TrackableType
    ):
        """
        Log the given parameter value in the static hyperparameters dictionary.

        :param parameter_name: The parameter name.
        :param value:          The parameter value to log.
        """
        self._static_hyperparameters[parameter_name] = value

    def log_dynamic_hyperparameter(
        self, parameter_name: str, value: MLTypes.TrackableType
    ):
        """
        Log the given parameter value in the dynamic hyperparameters dictionary at the current iteration (if it's a new
        parameter it will be iteration 0). If the parameter appears in the static hyperparameters dictionary, it will be
        removed from there as it is now dynamic.

        :param parameter_name: The parameter name.
        :param value:          The parameter value to log.
        """
        # Check if it's a new hyperparameter being tracked:
        if parameter_name not in self._dynamic_hyperparameters:
            # Look in the static hyperparameters:
            if parameter_name in self._static_hyperparameters:
                self._static_hyperparameters.pop(parameter_name)
            # Add it as a dynamic hyperparameter:
            self._dynamic_hyperparameters[parameter_name] = [value]
        else:
            self._dynamic_hyperparameters[parameter_name].append(value)
