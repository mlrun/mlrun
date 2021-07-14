from typing import Dict, Union

import numpy as np

from mlrun import MLClientCtx
from mlrun.artifacts import Artifact, ChartArtifact
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
        context: MLClientCtx,
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
        # Create child context to hold the current epoch's results:
        child_context = self._context.get_child_context()

        # Set the current iteration according to the epoch number:
        child_context._iteration = epoch + 1

        # Log the collected hyperparameters and values as results to the epoch's child context:
        for static_parameter, value in self._static_hyperparameters.items():
            child_context.log_result(static_parameter, value)
        for dynamic_parameter, values in self._dynamic_hyperparameters.items():
            child_context.log_result(dynamic_parameter, values[-1])
        for metric, results in self._training_summaries.items():
            child_context.log_result("training_{}".format(metric), results[-1])
        for metric, results in self._validation_summaries.items():
            child_context.log_result("validation_{}".format(metric), results[-1])

        # Update the last epoch to the main context:
        self._context._results = child_context.results

        # Log the epochs metrics results as chart artifacts:
        for metrics_prefix, metrics_dictionary in zip(
            ["training", "validation"],
            [self._training_results, self._validation_results],
        ):
            for metric_name, metric_epochs in metrics_dictionary.items():
                # Create the chart artifact:
                chart_name = "{}_{}_epoch_{}".format(
                    metrics_prefix, metric_name, len(metric_epochs)
                )
                chart_artifact = ChartArtifact(
                    key="{}.html".format(chart_name),
                    header=["iteration", "result"],
                    data=list(
                        np.array(
                            [list(np.arange(len(metric_epochs[-1]))), metric_epochs[-1]]
                        ).transpose()
                    ),
                )
                # Log the artifact:
                child_context.log_artifact(
                    chart_artifact,
                    local_path=chart_artifact.key,
                    artifact_path=child_context.artifact_path,
                )
                # Collect it for later adding it to the model logging as extra data:
                self._artifacts[chart_name] = chart_artifact

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
        for metric_name, metric_results in self._training_summaries.items():
            if metric_name in self._validation_summaries:
                header = ["epoch", "training_result", "validation_result"]
                data = list(
                    np.array(
                        [
                            list(np.arange(len(metric_results))),
                            metric_results,
                            self._validation_summaries[metric_name],
                        ]
                    ).transpose()
                )
            else:
                header = ["epoch", "training_result"]
                data = list(
                    np.array(
                        [list(np.arange(len(metric_results))), metric_results]
                    ).transpose()
                )
            # Create the chart artifact:
            chart_name = "{}_summary".format(metric_name)
            chart_artifact = ChartArtifact(
                key="{}.html".format(chart_name), header=header, data=data,
            )
            # Log the artifact:
            self._context.log_artifact(
                chart_artifact, local_path=chart_artifact.key,
            )
            # Collect it for later adding it to the model logging as extra data:
            self._artifacts[chart_name] = chart_artifact

        # Create chart artifacts for dynamic hyperparameters:
        for parameter_name, parameter_values in self._dynamic_hyperparameters.items():
            # Create the chart artifact:
            chart_artifact = ChartArtifact(
                key="{}.html".format(parameter_name),
                header=["epoch", "value"],
                data=list(
                    np.array(
                        [list(np.arange(len(parameter_values))), parameter_values]
                    ).transpose(),
                ),
            )
            # Log the artifact:
            self._context.log_artifact(
                chart_artifact, local_path=chart_artifact.key,
            )
            # Collect it for later adding it to the model logging as extra data:
            self._artifacts[parameter_name] = chart_artifact

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
