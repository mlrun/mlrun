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
from typing import Optional, Union

import numpy as np
import pandas as pd

import mlrun

from .._common import LoggingMode
from .metric import Metric
from .utils import SKLearnUtils


class Estimator:
    """
    Class for handling metrics calculations during a run.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        metrics: Optional[list[Metric]] = None,
    ):
        """
        Initialize an estimator with the given metrics. The estimator will log the calculated results using the given
        context.

        :param context: The context to log with.
        :param metrics: The metrics
        """
        # Store the context and metrics:
        self._context = context
        self._metrics = metrics if metrics is not None else []

        # Setup the logger's mode (default:  Training):
        self._mode = LoggingMode.TRAINING

        # Prepare the dictionaries to hold the results. Once they are logged they will be moved from one to another:
        self._logged_results = {}  # type: Dict[str, float]
        self._not_logged_results = {}  # type: Dict[str, float]

    @property
    def context(self) -> mlrun.MLClientCtx:
        """
        Get the logger's MLRun context.

        :return: The logger's MLRun context.
        """
        return self._context

    @property
    def results(self) -> dict[str, float]:
        """
        Get the logged results.

        :return: The logged results.
        """
        return self._logged_results

    def set_mode(self, mode: LoggingMode):
        """
        Set the estimator's mode.

        :param mode: The mode to set.
        """
        self._mode = mode

    def set_context(self, context: mlrun.MLClientCtx):
        """
        Set the context this logger will log with.

        :param context: The to be set MLRun context.
        """
        self._context = context

    def set_metrics(self, metrics: list[Metric]):
        """
        Update the metrics of this logger to the given list of metrics here.

        :param metrics: The list of metrics to override the current one.
        """
        self._metrics = metrics

    def is_probabilities_required(self) -> bool:
        """
        Check if probabilities are required in order to calculate some of the metrics.

        :return: True if probabilities are required by at least one metric and False otherwise.
        """
        return any(metric.need_probabilities for metric in self._metrics)

    def estimate(
        self,
        y_true: Union[np.ndarray, pd.DataFrame, pd.Series],
        y_pred: Union[np.ndarray, pd.DataFrame, pd.Series],
        is_probabilities: bool = False,
    ):
        """
        Calculate the results according to the 'is_probabilities' flag and log them.

        :param y_true:           The ground truth values to send for the metrics functions.
        :param y_pred:           The predictions to send for the metrics functions.
        :param is_probabilities: True if the 'y_pred' is a prediction of probabilities (from 'predict_proba') and False
                                 if not. Default: False.
        """
        # Calculate the metrics results:
        self._calculate_results(
            y_true=y_true, y_pred=y_pred, is_probabilities=is_probabilities
        )

        # Log if a context is available:
        if self._context is not None:
            # Log the results in queue:
            self._log_results()
            # Commit:
            self._context.commit(completed=False)

    def _calculate_results(
        self,
        y_true: Union[np.ndarray, pd.DataFrame, pd.Series],
        y_pred: Union[np.ndarray, pd.DataFrame, pd.Series],
        is_probabilities: bool,
    ):
        """
        Calculate the results from all the metrics in the estimator.

        :param y_true:           The ground truth values to send for the metrics functions.
        :param y_pred:           The predictions to send for the metrics functions.
        :param is_probabilities: True if the 'y_pred' is a prediction of probabilities (from 'predict_proba') and False
                                 if not.
        """
        # Use squeeze to remove redundant dimensions:
        y_true = np.squeeze(SKLearnUtils.to_array(dataset=y_true))
        y_pred = np.squeeze(SKLearnUtils.to_array(dataset=y_pred))

        # Calculate the metrics:
        for metric in self._metrics:
            if metric.need_probabilities == is_probabilities:
                self._not_logged_results[metric.name] = metric(y_true, y_pred)

        # Add evaluation prefix if in Evaluation mode:
        if self._mode == LoggingMode.EVALUATION:
            self._not_logged_results = {
                f"evaluation_{key}": value
                for key, value in self._not_logged_results.items()
            }

    def _log_results(self):
        """
        Log the calculated metrics results using the logger's context.
        """
        # Use the context to log each metric result:
        self._context.log_results(self._not_logged_results)

        # Collect the logged results:
        self._logged_results = {**self._logged_results, **self._not_logged_results}

        # Clean the not logged results dictionary:
        self._not_logged_results = {}
