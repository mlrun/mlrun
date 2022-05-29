from enum import Enum
from typing import Dict, List, Union

import numpy as np
import pandas as pd

import mlrun
from mlrun.artifacts import Artifact

from .metric import Metric
from .model_handler import MLModelHandler
from .plan import MLPlan, MLPlanStages
from .utils import to_array


class LoggerMode(Enum):
    """
    The logger's mode, can be training or evaluation.
    """

    TRAINING = "training"
    EVALUATION = "evaluation"


class Logger:
    """
    Class for handling production of artifact plans and metrics calculations during a run.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        plans: List[MLPlan] = None,
        metrics: List[Metric] = None,
    ):
        """
        Initialize a planner with the given plans. The planner will log the produced artifacts using the given context.

        :param context: The context to log with.
        :param plans:   The plans the planner will manage.
        :param metrics: The metrics
        """
        # Store the context and plans:
        self._context = context
        self._plans = plans if plans is not None else []
        self._metrics = metrics if metrics is not None else []

        # Setup the logger's mode (defaulted to Training):
        self._mode = LoggerMode.TRAINING

        # Prepare the dictionaries to hold the artifacts. Once they are logged they will be moved from one to another:
        self._logged_artifacts = {}  # type: Dict[str, Artifact]
        self._not_logged_artifacts = {}  # type: Dict[str, Artifact]

        # Prepare the dictionaries to hold the results. Once they are logged they will be moved from one to another:
        self._logged_results = {}  # type: Dict[str, float]
        self._not_logged_results = {}  # type: Dict[str, float]

    @property
    def mode(self) -> LoggerMode:
        """
        Get the logger's mode.

        :return: The logger mode.
        """
        return self._mode

    @property
    def context(self) -> mlrun.MLClientCtx:
        """
        Get the logger's MLRun context.

        :return: The logger's MLRun context.
        """
        return self._context

    @property
    def artifacts(self) -> Dict[str, Artifact]:
        """
        Get the logged artifacts.

        :return: The logged artifacts.
        """
        return self._logged_artifacts

    @property
    def results(self) -> Dict[str, float]:
        """
        Get the logged results.

        :return: The logged results.
        """
        return self._logged_results

    def set_mode(self, mode: LoggerMode):
        """
        Set the logger's mode.

        :param mode: The mode to set. One of Logger.LoggerMode options.
        """
        self._mode = mode

    def set_context(self, context: mlrun.MLClientCtx):
        """
        Set the context this logger will log with.

        :param context: The to be set MLRun context.
        """
        self._context = context

    def set_plans(self, plans: List[MLPlan]):
        """
        Update the plans of this logger to the given list of plans here.

        :param plans: The list of plans to override the current one.
        """
        self._plans = plans

    def set_metrics(self, metrics: List[Metric]):
        """
        Update the metrics of this logger to the given list of metrics here.

        :param metrics: The list of metrics to override the current one.
        """
        self._metrics = metrics

    def is_probabilities_required(self) -> bool:
        """
        Check if probabilities are required in order to produce and calculate some of the artifacts and metrics.

        :return: True if probabilities are required by at least one plan or metric and False otherwise.
        """
        probabilities_for_plans = any(plan.need_probabilities for plan in self._plans)
        probabilities_for_metrics = any(
            metric.need_probabilities for metric in self._metrics
        )
        return probabilities_for_plans or probabilities_for_metrics

    def log_stage(self, stage: MLPlanStages, is_probabilities: bool = False, **kwargs):
        """
        Produce the artifacts ready at the given stage and log them.

        :param stage:            The current stage to log at.
        :param is_probabilities: True if the 'y_pred' is a prediction of probabilities (from 'predict_proba') and False
                                 if not. Defaulted to False.
        :param kwargs:           All of the required produce arguments to pass onto the plans.
        """
        # Produce all the artifacts according to the given stage:
        self._produce_artifacts(
            stage=stage, is_probabilities=is_probabilities, **kwargs
        )

        # Log if a context is available:
        if self._context is not None:
            # Log the artifacts in queue:
            self._log_artifacts()
            # Commit:
            self._context.commit(completed=False)

    def log_results(
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
                                 if not. Defaulted to False.
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

    def log_run(
        self,
        model_handler: MLModelHandler,
    ):
        """
        End the logger's run, logging the collected artifacts and metrics results with the model. The model will be
        updated if the logger is in evaluation mode or logged as a new artifact if in training mode.

        :param model_handler: The model handler object holding the model to save and log.
        """
        # In case of training, log the model as a new model artifact and in case of evaluation - update the current
        # model artifact:
        if self._mode == LoggerMode.TRAINING:
            model_handler.log(
                metrics=self._logged_results,
                artifacts=self._logged_artifacts,
            )
        else:
            model_handler.update(
                metrics=self._logged_results,
                artifacts=self._logged_artifacts,
            )

        # Commit:
        self._context.commit(completed=False)

    def _produce_artifacts(
        self, stage: MLPlanStages, is_probabilities: bool = False, **kwargs
    ):
        """
        Go through the plans and check if they are ready to be produced in the given stage of the run. If they are,
        the logger will pass all the arguments to the 'plan.produce' method and collect the returned artifact.

        :param stage:            The stage to produce the artifact to check if its ready.
        :param is_probabilities: True if the 'y_pred' that will be sent to 'produce' is a prediction of probabilities
                                 (from 'predict_proba') and False if not. Defaulted to False.
        :param kwargs:           All of the required produce arguments to pass onto the plans.
        """
        # Initialize a new list of plans for all the plans that will still need to be produced:
        plans = []

        # Go ver the plans to produce their artifacts:
        for plan in self._plans:
            # Check if the plan is ready:
            if plan.is_ready(stage=stage, is_probabilities=is_probabilities):
                # Produce the artifact:
                self._not_logged_artifacts = {
                    **self._not_logged_artifacts,
                    **plan.produce(**kwargs),
                }
                # If the plan should not be produced again, continue to the next one so it won't be collected:
                if not plan.is_reproducible():
                    continue
            # Collect the plan to produce it later (or again if reproducible):
            plans.append(plan)

        # Clear the old plans:
        self._plans = plans

        # Add evaluation prefix if in Evaluation mode:
        if self._mode == LoggerMode.EVALUATION:
            self._not_logged_artifacts = {
                f"evaluation-{key}": value
                for key, value in self._not_logged_artifacts.items()
            }
            for artifact in self._not_logged_artifacts.values():
                artifact.key = f"evaluation-{artifact.key}"

    def _calculate_results(
        self,
        y_true: Union[np.ndarray, pd.DataFrame, pd.Series],
        y_pred: Union[np.ndarray, pd.DataFrame, pd.Series],
        is_probabilities: bool,
    ):
        """
        Calculate the results from all the metrics in the logger.

        :param y_true:           The ground truth values to send for the metrics functions.
        :param y_pred:           The predictions to send for the metrics functions.
        :param is_probabilities: True if the 'y_pred' is a prediction of probabilities (from 'predict_proba') and False
                                 if not.
        """
        # Use squeeze to remove redundant dimensions:
        y_true = np.squeeze(to_array(dataset=y_true))
        y_pred = np.squeeze(to_array(dataset=y_pred))

        # Calculate the metrics:
        for metric in self._metrics:
            if metric.need_probabilities == is_probabilities:
                self._not_logged_results[metric.name] = metric(y_true, y_pred)

        # Add evaluation prefix if in Evaluation mode:
        if self._mode == LoggerMode.EVALUATION:
            self._not_logged_results = {
                f"evaluation_{key}": value
                for key, value in self._not_logged_results.items()
            }

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
        self._context.log_results(self._not_logged_results)

        # Collect the logged results:
        self._logged_results = {**self._logged_results, **self._not_logged_results}

        # Clean the not logged results dictionary:
        self._not_logged_results = {}
