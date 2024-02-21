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
import re

import numpy as np
import plotly.graph_objects as go

import mlrun
from mlrun.artifacts import Artifact, PlotlyArtifact

from .logger import Logger


class MLRunLogger(Logger):
    """
    MLRun logger is logging the information collected during training of the base logger and logging it to MLRun using
    an MLRun context.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx,
    ):
        """
        Initialize the MLRun logging interface to work with the given context.

        :param context: MLRun context to log to. The context parameters can be logged as static hyperparameters.
        """
        super().__init__()

        # An MLRun context to log to:
        self._context = context

        # Prepare the artifacts dictionary:
        self._artifacts = {}  # type: Dict[str, Artifact]

    def get_artifacts(self) -> dict[str, Artifact]:
        """
        Get the artifacts created by this logger.

        :return: The artifacts dictionary.
        """
        return self._artifacts

    def get_metrics(self) -> dict[str, float]:
        """
        Generate a metrics summary to log along the model.

        :return: The metrics summary.
        """
        return {
            f"{validation_set}_{metric_name}": results[-1]
            for validation_set, metrics in self._results.items()
            for metric_name, results in metrics.items()
        }

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

    def log_iteration_to_context(self):
        """
        Log the information of the last iteration and produce the updated artifacts. Each call will log the following
        information:

        * Results table:

          * Static hyperparameters.
          * Dynamic hyperparameters.
          * Metric results.

        * Plot artifacts:

          * A chart for each of the metrics iteration results.
          * A chart for each of the dynamic hyperparameters values.
        """
        # Log the collected hyperparameters:
        for static_parameter, value in self._static_hyperparameters.items():
            self._context.log_result(static_parameter, value)
        for dynamic_parameter, values in self._dynamic_hyperparameters.items():
            # Log as a result to the context (take the most recent result in the training history (-1 index):
            self._context.log_result(dynamic_parameter, values[-1])
            # Create the plotly artifact:
            artifact = self._produce_convergence_plot_artifact(
                name=dynamic_parameter,
                values=values,
            )
            # Log the artifact:
            self._context.log_artifact(artifact)
            # Collect it for later adding it to the model logging as extra data:
            self._artifacts[artifact.metadata.key] = artifact

        # Log the metrics:
        for metric_name, metric_results in {
            f"{validation_set}_{metric_name}": results
            for validation_set, metrics in self._results.items()
            for metric_name, results in metrics.items()
        }.items():
            # Log as a result to the context:
            self._context.log_result(metric_name, metric_results[-1])
            # Create the plotly artifact:
            artifact = self._produce_convergence_plot_artifact(
                name=f"{metric_name}_plot",
                values=metric_results,
            )
            # Log the artifact:
            self._context.log_artifact(artifact)
            # Collect it for later adding it to the model logging as extra data:
            self._artifacts[artifact.metadata.key] = artifact

        # Commit to update the changes, so they will be available in the MLRun UI:
        self._context.commit(completed=False)

    @staticmethod
    def _produce_convergence_plot_artifact(
        name: str, values: list[float]
    ) -> PlotlyArtifact:
        """
        Produce the convergences for the provided metric according.

        :param name:   The name of the metric / hyperparameter.
        :param values: The values per iteration of the metric / hyperparameter.

        :return: Plotly artifact of the convergence plot.
        """
        # Initialize a plotly figure:
        metric_figure = go.Figure()

        # Add titles:
        metric_figure.update_layout(
            title=f"{re.sub('_', ' ', name).capitalize()}",
            xaxis_title="Iterations",
            yaxis_title="Values",
        )

        # Draw:
        metric_figure.add_trace(
            go.Scatter(
                x=list(np.arange(len(values))),
                y=values,
                mode="lines",
            )
        )

        # Creating the artifact:
        return PlotlyArtifact(
            key=name,
            figure=metric_figure,
        )
