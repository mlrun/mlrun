from enum import Enum
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

import mlrun
from mlrun.artifacts import Artifact

from .model_handler import MLModelHandler
from .artifacts_library import MLPlanStages, MLPlan
from .metrics_library import Metric


class Logger:
    """
    Class for handling production of artifact plans and metrics calculations during a run.
    """

    def __init__(
        self, context: mlrun.MLClientCtx, plans: List[MLPlan], metrics: List[Metric]
    ):
        """
        Initialize a planner with the given plans. The planner will log the produced artifacts using the given context.

        :param context: The context to log with.
        :param plans:   The plans the planner will manage.
        :param metrics: The metrics
        """
        # Store the context and plans:
        self._context = context
        self._plans = plans
        self._metrics = metrics

        # Prepare the dictionaries to hold the artifacts. Once they are logged they will be moved from one to another:
        self._logged_artifacts = {}  # type: Dict[str, Artifact]
        self._not_logged_artifacts = {}  # type: Dict[str, Artifact]

        # Prepare the dictionaries to hold the results. Once they are logged they will be moved from one to another:
        self._logged_results = {}  # type: Dict[str, float]
        self._not_logged_results = {}  # type: Dict[str, float]

    @property
    def artifacts(self) -> Dict[str, Artifact]:
        """
        Get the logged artifacts.

        :return: The logged artifacts.
        """
        return {**self._logged_artifacts, **self._not_logged_artifacts}

    @property
    def results(self) -> Dict[str, float]:
        """
        Get the logged results.

        :return: The logged results.
        """
        return {**self._logged_results, **self._not_logged_results}

    def is_probabilities_required(self) -> bool:
        """
        Check if probabilities are required in order to produce and calculate some of the artifacts and metrics.

        :return: True if probabilities are required by at least one plan or metric and False otherwise.
        """
        probabilities_for_plans = any(plan.need_probabilities for plan in self._plans)
        probabilities_for_metrics = any(metric.need_probabilities for metric in self._metrics)
        return probabilities_for_plans or probabilities_for_metrics

    def produce_artifacts(self, stage: MLPlanStages, **kwargs):
        """
        Go through the plans and check if they are ready to be produced in the given stage of the run. If they are,
        the logger will pass all the arguments to the 'plan.produce' method and collect the returned artifact.

        :param stage: The stage to produce the artifact in.
        """
        for plan in self._plans:
            if plan.is_ready(stage=stage):
                self._not_logged_artifacts = {
                    **self._not_logged_artifacts,
                    **plan.produce(**kwargs),
                }

    def calculate_results(
        self,
        y_true: Union[np.ndarray, pd.DataFrame, pd.Series],
        y_pred: Union[np.ndarray, pd.DataFrame, pd.Series],
    ):
        """
        Calculate the results from all the metrics in the logger.

        :param y_true: The ground truth values to send for the metrics functions.
        :param y_pred: The predictions to send for the metrics functions.
        """
        for metric in self._metrics:
            self._not_logged_results[metric.name] = metric(y_true, y_pred)

    def log(self):
        """
        Use the logger's context to log the artifacts and results he collected.

        """
        # Log the artifacts in queue:
        self._log_artifacts()

        # Log the results in queue:
        self._log_results()

        # Commit:
        self._context.commit(completed=False)

    def _log_artifacts(self):
        """
        Log the produced plans artifacts using the logger's context.
        """
        # Use the context to log each artifact:
        for artifact in self._not_logged_artifacts.values():
            self._context.log_artifact(artifact)

        # Collect the logged artifacts:
        self._logged_artifacts = {
            **self._logged_artifacts,
            **self._not_logged_artifacts,
        }

        # Clean the not logged artifacts dictionary:
        self._not_logged_artifacts = {}

    def _log_results(self):
        """
        Log the calculated metrics results using the logger's context.
        """
        # Use the context to log each metric result:
        self._context.log_results(results=self._not_logged_results)

        # Collect the logged results:
        self._logged_results = {**self._logged_results, **self._not_logged_results}

        # Clean the not logged results dictionary:
        self._not_logged_results = {}
