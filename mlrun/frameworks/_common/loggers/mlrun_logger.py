from typing import Dict, List, Union

import numpy as np
from bokeh.plotting import figure

import mlrun
from mlrun.artifacts import Artifact, BokehArtifact
from mlrun.frameworks._common.loggers.logger import Logger
from mlrun.frameworks._common.model_handler import ModelHandler

# All trackable values types:
TrackableType = Union[str, bool, float, int]


class MLRunLogger(Logger):
    """
    MLRun logger is logging the information collected during training / evaluation of the base logger and logging it to
    MLRun using a MLRun context. The logging includes:

    * For each epoch:

      * Tracking table: epoch, static hyperparameters, dynamic hyperparameters, training metrics, validation metrics.
      * Per iteration (batch) chart artifacts for the training and validation metrics.

    * At the end of the run:

      * Per epoch chart artifacts for the validation summaries and dynamic hyperparameters.
      * Model is logged with all of the files and artifacts.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx,
        log_model_labels: Dict[str, TrackableType],
        log_model_parameters: Dict[str, TrackableType],
        log_model_extra_data: Dict[str, Union[TrackableType, Artifact]],
    ):
        """
        Initialize the MLRun logging interface to work with the given context.

        :param context:              MLRun context to log to. The context parameters can be logged as static
                                     hyperparameters.
        :param log_model_labels:     Labels to log with the model.
        :param log_model_parameters: Parameters to log with the model.
        :param log_model_extra_data: Extra data to log with the model.
        """
        super(MLRunLogger, self).__init__(context=context)

        # Store the context:
        self._log_model_labels = (
            log_model_labels if log_model_labels is not None else {}
        )
        self._log_model_parameters = (
            log_model_parameters if log_model_parameters is not None else {}
        )
        self._log_model_extra_data = (
            log_model_extra_data if log_model_extra_data is not None else {}
        )

        # Prepare the artifacts collection:
        self._artifacts = {}  # type: Dict[str, Artifact]

    def log_epoch_to_context(
        self, epoch: int,
    ):
        """
        Log the last epoch as a child context of the main context. The last epoch information recorded in the given
        tracking dictionaries will be logged, meaning the epoch index will not be taken from the given 'epoch'
        parameter, but the '-1' index will be used in all of the dictionaries. Each epoch will log the following
        information:

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
        # Log the collected hyperparameters and values as results to the epoch's child context:
        for static_parameter, value in self._static_hyperparameters.items():
            self._context.log_result(static_parameter, value)
        for dynamic_parameter, values in self._dynamic_hyperparameters.items():
            self._context.log_result(dynamic_parameter, values[-1])
        for metric, results in self._training_summaries.items():
            self._context.log_result("training_{}".format(metric), results[-1])
        for metric, results in self._validation_summaries.items():
            self._context.log_result("validation_{}".format(metric), results[-1])

        # Log the epochs metrics results as chart artifacts:
        for loop, metrics_dictionary in zip(
            ["training", "validation"],
            [self._training_results, self._validation_results],
        ):
            for metric_name in metrics_dictionary:
                # Create the bokeh artifact:
                artifact = self._generate_metric_results_artifact(
                    epoch=len(metrics_dictionary[metric_name]),
                    loop=loop,
                    name=metric_name,
                    results=metrics_dictionary[metric_name][-1],
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
        self._context.commit()

    def log_run(self, model_handler: ModelHandler):
        """
        Log the run, summarizing the validation metrics and dynamic hyperparameters across all epochs and saving the
        model. The run log information will be the following:

        * Plot artifacts:

          * A chart for each of the metrics epochs results summaries across all the run (training and validation).
          * A chart for each of the dynamic hyperparameters epochs values across all the run.

        * Model artifact: The model will be saved and logged with all the collected artifacts of this logger.

        :param model_handler: The model handler object holding the model to save and log.
        """
        # Create chart artifacts for summaries:
        for metric_name in self._training_summaries:
            # Create the bokeh artifact:
            artifact = self._generate_summary_results_artifact(
                name=metric_name,
                training_results=self._training_summaries[metric_name],
                validation_results=self._validation_summaries.get(metric_name, None),
            )
            # Log the artifact:
            self._context.log_artifact(
                artifact, local_path=artifact.key,
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
                artifact, local_path=artifact.key,
            )
            # Collect it for later adding it to the model logging as extra data:
            self._artifacts[artifact.key.split(".")[0]] = artifact

        # Log the model:
        model_handler.set_context(context=self._context)
        model_handler.log(
            labels=self._log_model_labels,
            parameters=self._log_model_parameters,
            extra_data=self._log_model_extra_data,
            artifacts=self._artifacts,
        )

        # Commit:
        self._context.commit()

    @staticmethod
    def _generate_metric_results_artifact(
        epoch: int, loop: str, name: str, results: List[float]
    ) -> BokehArtifact:
        """
        Generate a bokeh artifact for the results of the metric provided.

        :param epoch:   The epoch of the recorded resutls.
        :param loop:    The results loop, training or validation.
        :param name:    The metric name.
        :param results: The metric results at the given epoch.

        :return: The generated bokeh figure wrapped in MLRun artifact.
        """
        # Parse the artifact's name:
        artifact_name = "{}_{}_epoch_{}".format(loop, name, epoch)

        # Initialize a bokeh figure:
        metric_figure = figure(
            title="{} Results for epoch {}".format(name, epoch),
            x_axis_label="Batches",
            y_axis_label="Results",
            x_axis_type="linear",
        )

        # Draw the results:
        metric_figure.line(x=list(np.arange(len(results))), y=results)

        # Create the bokeh artifact:
        artifact = BokehArtifact(
            key="{}.html".format(artifact_name), figure=metric_figure
        )

        return artifact

    @staticmethod
    def _generate_summary_results_artifact(
        name: str, training_results: List[float], validation_results: List[float]
    ) -> BokehArtifact:
        """
        Generate a bokeh artifact for the results summary across all the epochs of training.

        :param name:               The metric name.
        :param training_results:   The metric training results summaries across the epochs.
        :param validation_results: The metric validation results summaries across the epochs. If validation was not
                                   performed, None should be passed.

        :return: The generated bokeh figure wrapped in MLRun artifact.
        """
        # Parse the artifact's name:
        artifact_name = "{}_summary".format(name)

        # Initialize a bokeh figure:
        summary_figure = figure(
            title="{} Summary".format(name),
            x_axis_label="Epochs",
            y_axis_label="Results",
            x_axis_type="linear",
        )

        # Draw the results:
        summary_figure.line(
            x=list(np.arange(1, len(training_results) + 1)),
            y=training_results,
            legend_label="Training",
        )
        summary_figure.circle(
            x=list(np.arange(1, len(training_results) + 1)),
            y=training_results,
            legend_label="Training",
        )
        if validation_results is not None:
            summary_figure.line(
                x=list(np.arange(1, len(validation_results) + 1)),
                y=validation_results,
                legend_label="Validation",
                color="orangered",
            )
            summary_figure.circle(
                x=list(np.arange(1, len(validation_results) + 1)),
                y=validation_results,
                legend_label="Validation",
                color="orangered",
            )

        # Create the bokeh artifact:
        artifact = BokehArtifact(
            key="{}.html".format(artifact_name), figure=summary_figure
        )

        return artifact

    @staticmethod
    def _generate_dynamic_hyperparameter_values_artifact(
        name: str, values: List[float]
    ) -> BokehArtifact:
        """
        Generate a bokeh artifact for the values of the hyperparameter provided.

        :param name:   The hyperparameter name.
        :param values: The hyperparameter values across the training.

        :return: The generated bokeh figure wrapped in MLRun artifact.
        """
        # Parse the artifact's name:
        artifact_name = "{}.html".format(name)

        # Initialize a bokeh figure:
        hyperparameter_figure = figure(
            title=name,
            x_axis_label="Epochs",
            y_axis_label="Values",
            x_axis_type="linear",
        )

        # Draw the values:
        hyperparameter_figure.line(x=list(np.arange(len(values))), y=values)
        hyperparameter_figure.circle(x=list(np.arange(len(values))), y=values)

        # Create the bokeh artifact:
        artifact = BokehArtifact(
            key="{}.html".format(artifact_name), figure=hyperparameter_figure
        )

        return artifact
