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
import itertools
from typing import Dict, List, Union

import numpy as np
import plotly.graph_objects as go

import mlrun
from mlrun.artifacts import Artifact, PlotlyArtifact

from ..._common import LoggingMode
from ..model_handler import DLModelHandler
from ..utils import DLTypes
from .logger import Logger


class MLRunLogger(Logger):
    """
    MLRun logger is logging the information collected during training / evaluation of the base logger and logging it to
    MLRun using a MLRun context. The logging includes:

    * For each epoch:

      * Tracking table: epoch, static hyperparameters, dynamic hyperparameters, training metrics, validation metrics.
      * Per iteration (batch) chart artifacts for the training and validation metrics.

    * At the end of the run:

      * Per epoch chart artifacts for the validation summaries and dynamic hyperparameters.
      * Model is logged with all the files and artifacts.
    """

    class _Loops:
        """
        The types of loops performed in a training / evaluation process. Used as names and prefixes to add to the
        metrics names when logging.
        """

        TRAINING = "training"
        VALIDATION = "validation"
        EVALUATION = "evaluation"

    def __init__(
        self,
        context: mlrun.MLClientCtx,
    ):
        """
        Initialize the MLRun logging interface to work with the given context.

        :param context:              MLRun context to log to. The context parameters can be logged as static
                                     hyperparameters.
        """
        super(MLRunLogger, self).__init__(context=context)

        # Prepare the artifacts collection:
        self._artifacts = {}  # type: Dict[str, Artifact]

    def log_epoch_to_context(
        self,
        epoch: int,
    ):
        """
        Log the last epoch. The last epoch information recorded in the given tracking dictionaries will be logged,
        meaning the epoch index will not be taken from the given 'epoch' parameter, but the '-1' index will be used in
        all the dictionaries. Each epoch will log the following information:

        * Results table:

          * Static hyperparameters.
          * Dynamic hyperparameters.
          * Last iteration recorded training results for loss and metrics.
          * Validation results summaries for loss and metrics.

        * Plot artifacts:

          * A chart for each of the metrics iteration results in training.
          * A chart for each of the metrics iteration results in validation.

        :param epoch: The epoch number that has just ended.
        """
        # Log the collected hyperparameters and values as results (the most recent value collected (-1 index)):
        for static_parameter, value in self._static_hyperparameters.items():
            self._context.log_result(static_parameter, value)
        if self._mode == LoggingMode.TRAINING:
            for dynamic_parameter, values in self._dynamic_hyperparameters.items():
                self._context.log_result(dynamic_parameter, values[-1])
            for metric, results in self._training_summaries.items():
                self._context.log_result(
                    f"{self._Loops.TRAINING}_{metric}", results[-1]
                )
        for metric, results in self._validation_summaries.items():
            self._context.log_result(
                f"{self._Loops.EVALUATION}_{metric}"
                if self._mode == LoggingMode.EVALUATION
                else f"{self._Loops.VALIDATION}_{metric}",
                results[-1],
            )

        # Log the epochs metrics results as chart artifacts:
        loops = (
            [self._Loops.EVALUATION]
            if self._mode == LoggingMode.EVALUATION
            else [self._Loops.TRAINING, self._Loops.VALIDATION]
        )
        metrics_dictionaries = (
            [self._validation_results]
            if self._mode == LoggingMode.EVALUATION
            else [self._training_results, self._validation_results]
        )
        for loop, metrics_dictionary in zip(loops, metrics_dictionaries):
            for metric_name in metrics_dictionary:
                # Create the plotly artifact:
                artifact = self._generate_metric_results_artifact(
                    loop=loop,
                    name=metric_name,
                    epochs_results=metrics_dictionary[metric_name],
                )
                # Log the artifact:
                self._context.log_artifact(
                    artifact,
                    local_path=artifact.key,
                    artifact_path=self._context.artifact_path,
                )
                # Collect it for later adding it to the model logging as extra data:
                self._artifacts[artifact.key.split(".")[0]] = artifact

        # Commit and commit children for MLRun flag bug:
        self._context.commit(completed=False)

    def log_run(
        self,
        model_handler: DLModelHandler,
        tag: str = "",
        labels: Dict[str, DLTypes.TrackableType] = None,
        parameters: Dict[str, DLTypes.TrackableType] = None,
        extra_data: Dict[str, Union[DLTypes.TrackableType, Artifact]] = None,
    ):
        """
        Log the run, summarizing the validation metrics and dynamic hyperparameters across all epochs. If 'update' is
        False, the collected logs will be updated to the model inside the given handler, otherwise the model will be
        saved and logged as a new artifact. The run log information will be the following:

        * Plot artifacts:

          * A chart for each of the metrics epochs results summaries across all the run (training and validation).
          * A chart for each of the dynamic hyperparameters epochs values across all the run.

        * Model artifact (only in training mode): The model will be saved and logged with all the collected artifacts
                                                  of this logger.

        :param model_handler: The model handler object holding the model to save and log.
        :param tag:           Version tag to give the logged model.
        :param labels:        Labels to log with the model.
        :param parameters:    Parameters to log with the model.
        :param extra_data:    Extra data to log with the model.
        """
        # If in training mode, log the summaries and hyperparameters artifacts:
        if self._mode == LoggingMode.TRAINING:
            # Create chart artifacts for summaries:
            for metric_name in self._training_summaries:
                # Create the plotly artifact:
                artifact = self._generate_summary_results_artifact(
                    name=metric_name,
                    training_results=self._training_summaries[metric_name],
                    validation_results=self._validation_summaries.get(
                        metric_name, None
                    ),
                )
                # Log the artifact:
                self._context.log_artifact(
                    artifact,
                    local_path=artifact.key,
                )
                # Collect it for later adding it to the model logging as extra data:
                self._artifacts[artifact.key.split(".")[0]] = artifact
            # Create chart artifacts for dynamic hyperparameters:
            for parameter_name in self._dynamic_hyperparameters:
                # Create the chart artifact:
                artifact = self._generate_dynamic_hyperparameter_values_artifact(
                    name=parameter_name,
                    values=self._dynamic_hyperparameters[parameter_name],
                )
                # Log the artifact:
                self._context.log_artifact(
                    artifact,
                    local_path=artifact.key,
                )
                # Collect it for later adding it to the model logging as extra data:
                self._artifacts[artifact.key.split(".")[0]] = artifact

        # Get the final metrics summary:
        metrics = self._generate_metrics_summary()

        # Log or update:
        model_handler.set_context(context=self._context)
        if self._mode == LoggingMode.EVALUATION:
            model_handler.update(
                labels=labels,
                parameters=parameters,
                metrics=metrics,
                extra_data=extra_data,
                artifacts=self._artifacts,
            )
        else:
            model_handler.log(
                tag=tag,
                labels=labels,
                parameters=parameters,
                metrics=metrics,
                extra_data=extra_data,
                artifacts=self._artifacts,
            )

        # Commit to update the changes, so they will be available in the MLRun UI:
        self._context.commit(completed=False)

    def _generate_metrics_summary(self) -> Dict[str, float]:
        """
        Generate a metrics summary to log along the model.

        :return: The metrics summary.
        """
        # If in training mode, return both training and validation metrics:
        if self._mode == LoggingMode.TRAINING:
            return {
                **{
                    f"{self._Loops.TRAINING}_{name}": values[-1]
                    for name, values in self._training_summaries.items()
                },
                **{
                    f"{self._Loops.VALIDATION}_{name}": values[-1]
                    for name, values in self._validation_summaries.items()
                },
            }

        # Return the evaluation metrics:
        return {
            f"{self._Loops.EVALUATION}_{name}": values[-1]
            for name, values in self._validation_summaries.items()
        }

    @staticmethod
    def _generate_metric_results_artifact(
        loop: str, name: str, epochs_results: List[List[float]]
    ) -> PlotlyArtifact:
        """
        Generate a plotly artifact for the results of the metric provided.

        :param loop:           The results loop, training or validation.
        :param name:           The metric name.
        :param epochs_results: The entire metric results across all logged epochs.

        :return: The generated plotly figure wrapped in MLRun artifact.
        """
        # Parse the artifact's name:
        artifact_name = f"{loop}_{name}"

        # Initialize a plotly figure:
        metric_figure = go.Figure()

        # Add titles:
        metric_figure.update_layout(
            title=f"{loop} {name} Results",
            xaxis_title="Batches",
            yaxis_title="Results",
        )

        # Prepare the results list:
        results = list(itertools.chain(*epochs_results))

        # Prepare the epochs list:
        epochs_indices = [
            i * len(epoch_results) for i, epoch_results in enumerate(epochs_results)
        ][1:]

        # Draw:
        metric_figure.add_trace(
            go.Scatter(x=list(np.arange(len(results))), y=results, mode="lines")
        )
        for epoch_index in epochs_indices:
            metric_figure.add_vline(x=epoch_index, line_dash="dash", line_color="grey")

        # Create the plotly artifact:
        artifact = PlotlyArtifact(key=f"{artifact_name}.html", figure=metric_figure)

        return artifact

    @staticmethod
    def _generate_summary_results_artifact(
        name: str, training_results: List[float], validation_results: List[float]
    ) -> PlotlyArtifact:
        """
        Generate a plotly artifact for the results summary across all the epochs of training.

        :param name:               The metric name.
        :param training_results:   The metric training results summaries across the epochs.
        :param validation_results: The metric validation results summaries across the epochs. If validation was not
                                   performed, None should be passed.

        :return: The generated plotly figure wrapped in MLRun artifact.
        """
        # Parse the artifact's name:
        artifact_name = f"{name}_summary"

        # Initialize a plotly figure:
        summary_figure = go.Figure()

        # Add titles:
        summary_figure.update_layout(
            title=f"{name} Summary",
            xaxis_title="Epochs",
            yaxis_title="Results",
        )

        # Draw the results:
        summary_figure.add_trace(
            go.Scatter(
                x=list(np.arange(1, len(training_results) + 1)),
                y=training_results,
                mode="lines+markers",
                name="Training",
            )
        )
        if validation_results is not None:
            summary_figure.add_trace(
                go.Scatter(
                    x=list(np.arange(1, len(validation_results) + 1)),
                    y=validation_results,
                    mode="lines+markers",
                    name="Validation",
                )
            )

        # Create the plotly artifact:
        artifact = PlotlyArtifact(key=f"{artifact_name}.html", figure=summary_figure)

        return artifact

    @staticmethod
    def _generate_dynamic_hyperparameter_values_artifact(
        name: str, values: List[float]
    ) -> PlotlyArtifact:
        """
        Generate a plotly artifact for the values of the hyperparameter provided.

        :param name:   The hyperparameter name.
        :param values: The hyperparameter values across the training.

        :return: The generated plotly figure wrapped in MLRun artifact.
        """
        # Parse the artifact's name:
        artifact_name = f"{name}_values"

        # Initialize a plotly figure:
        hyperparameter_figure = go.Figure()

        # Add titles:
        hyperparameter_figure.update_layout(
            title=name,
            xaxis_title="Epochs",
            yaxis_title="Values",
        )

        # Draw the values:
        hyperparameter_figure.add_trace(
            go.Scatter(x=list(np.arange(len(values))), y=values, mode="lines+markers")
        )

        # Create the plotly artifact:
        artifact = PlotlyArtifact(
            key=f"{artifact_name}.html", figure=hyperparameter_figure
        )

        return artifact
