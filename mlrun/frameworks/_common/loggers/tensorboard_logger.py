from typing import Union, Dict, List, Any, Callable, TypeVar
from abc import abstractmethod
import os
from datetime import datetime
import yaml

import mlrun
from mlrun.config import config
from mlrun import MLClientCtx
from mlrun.frameworks._common.loggers.logger import Logger, TrackableType


# Define a type variable for the different tensor type objects of the supported frameworks:
Weight = TypeVar("Weight")


class TensorboardLogger(Logger):
    """
    An abstract tensorboard logger class for logging the information collected during training / evaluation of the base
    logger to tensorboard. Each framework has its own way of logging to tensorboard, but each must implement the entire
    features listed in this class. The logging includes:

    * Summary text of the run with a hyperlink to the MLRun log if it was done.
    * Hyperparameters tuning table: static hyperparameters, dynamic hyperparameters and epoch validation summaries.
    * Plots:

      * Per iteration (batch) plot for the training and validation metrics.
      * Per epoch plot for the dynamic hyperparameters and validation summaries results.
      * Per epoch weights statistics for each weight and statistic.

    * Histograms per epoch for each of the logged weights.
    * Distributions per epoch for each of the logged weights.
    * Images per epoch for each of the logged weights.
    * Model architecture graph.
    """

    # The default tensorboard directory to be used with a given context:
    _DEFAULT_TENSORBOARD_DIRECTORY = os.path.join(
        os.sep, "User", ".tensorboard", "{{project}}"
    )

    class _Sections:
        """
        Tensorboard split his plots to sections via a path like name <SECTION>/<PLOT_NAME>. These are the sections used
        in this callback for logging.
        """

        TRAINING = "Training"
        VALIDATION = "Validation"
        SUMMARY = "Summary"
        HYPERPARAMETERS = "Hyperparameters"
        WEIGHTS = "Weights"

    def __init__(
        self,
        statistics_functions: List[Callable[[Weight], Union[float, Weight]]],
        context: MLClientCtx = None,
        tensorboard_directory: str = None,
        run_name: str = None,
    ):
        """
        Initialize a tensorboard logger callback with the given configuration. At least one of 'context' and
        'tensorboard_directory' must be given.

        :param statistics_functions:  A list of statistics functions to calculate at the end of each epoch on the
                                      tracked weights. Only relevant if weights are being tracked. The functions in
                                      the list must accept one Weight and return a float (or float convertible) value.
        :param context:               A mlrun context to use for logging into the user's tensorboard directory.
        :param tensorboard_directory: If context is not given, or if wished to set the directory even with context,
                                      this will be the output for the event logs of tensorboard.
        :param run_name:              This experiment run name. Each run name will be indexed at the end of the name so
                                      each experiment will be numbered automatically. If a context was given, the
                                      context's uid will be added instead of an index. If a run name was not given the
                                      current time in the following format: 'YYYY-mm-dd_HH:MM:SS'.
        """
        super(TensorboardLogger, self).__init__()

        # Store the given parameters:
        self._statistics_functions = statistics_functions
        self._context = context
        self._tensorboard_directory = tensorboard_directory
        self._run_name = run_name

        # Setup the output path:
        self._output_path = None

        # Setup the weights dictionaries - a dictionary of all required weight parameters:
        # [Weight: str] -> [value: WeightType]
        self._weights = {}  # type: Dict[str, Weight]

        # Setup the statistics dictionaries - a dictionary of statistics for the required weights per epoch:
        # [Statistic: str] -> [Weight: str] -> [epoch: int] -> [value: float]
        self._weights_statistics = {}  # type: Dict[str, Dict[str, List[float]]]
        for statistic_function in self._statistics_functions:
            self._weights_statistics[
                statistic_function.__name__
            ] = {}  # type: Dict[str, List[float]]

    @property
    def weights(self) -> Dict[str, Weight]:
        """
        Get the logged weights dictionary. Each of the logged weight will be found by its name.

        :return: The weights dictionary.
        """
        return self._weights

    @property
    def weight_statistics(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Get the logged statistics for all the tracked weights. Each statistic has a dictionary of weights and their list
        of epochs values.

        :return: The statistics dictionary.
        """
        return self._weights_statistics

    def log_weight(self, weight_name: str, weight_holder: Weight):
        """
        Log the weight into the weights dictionary so it will be tracked and logged during the epochs. For each logged
        weight the key for it in the statistics logged will be initialized as well.

        :param weight_name:   The weight's name.
        :param weight_holder: The weight holder to track. Both Tensorflow (including Keras) and PyTorch (including
                              Lightning) keep the weights tensor in a holder object - 'Variable' for Tensorflow and
                              'Parameter' for PyTorch.
        """
        # Collect the given weight:
        self._weights[weight_name] = weight_holder

        # Insert the weight to all the statistics:
        for statistic in self._weights_statistics:
            self._weights_statistics[statistic][weight_name] = []

    def log_weights_statistics(self):
        """
        Calculate the statistics on the current weights and log the results.
        """
        for weight_name, weight_parameter in self._weights.items():
            for statistic_function in self._statistics_functions:
                self._weights_statistics[statistic_function.__name__][
                    weight_name
                ].append(float(statistic_function(weight_parameter)))

    @abstractmethod
    def log_run_start_text_to_tensorboard(self):
        """
        Log the initial information summary of this training / validation run to tensorboard.
        """
        pass

    @abstractmethod
    def log_epoch_text_to_tensorboard(self):
        """
        Log the last epoch summary of this training / validation run to tensorboard.
        """
        pass

    @abstractmethod
    def log_run_end_text_to_tensorboard(self):
        """
        Log the final information summary of this training / validation run to tensorboard.
        """
        pass

    @abstractmethod
    def log_parameters_table_to_tensorboard(self):
        """
        Log the validation summaries, static and dynamic hyperparameters to the 'HParams' table in tensorboard.
        """
        pass

    @abstractmethod
    def log_training_results_to_tensorboard(self):
        """
        Log the recent training iteration metrics results to tensorboard.
        """
        pass

    @abstractmethod
    def log_validation_results_to_tensorboard(self):
        """
        Log the recent validation iteration metrics results to tensorboard.
        """
        pass

    @abstractmethod
    def log_dynamic_hyperparameters_to_tensorboard(self):
        """
        Log the recent epoch dynamic hyperparameters values to tensorboard.
        """
        pass

    @abstractmethod
    def log_summaries_to_tensorboard(self):
        """
        Log the recent epoch summaries results to tensorboard.
        """
        pass

    @abstractmethod
    def log_weights_histograms_to_tensorboard(self):
        """
        Log the current state of the weights as histograms to tensorboard.
        """
        pass

    @abstractmethod
    def log_weights_images_to_tensorboard(self):
        """
        Log the current state of the weights as images to tensorboard.
        """
        pass

    @abstractmethod
    def log_statistics_to_tensorboard(self):
        """
        Log the last stored statistics values this logger collected to tensorboard.
        """
        pass

    @abstractmethod
    def log_model_to_tensorboard(self, *args, **kwargs):
        """
        Log the given model as a graph in tensorboard.
        """
        pass

    def _create_output_path(self):
        """
        Create the output path, indexing the given run name as needed.
        """
        # If a run name was not given, take the current timestamp as the run name in the format 'YYYY-mm-dd_HH:MM:SS':
        if self._run_name is None:
            self._run_name = (
                str(datetime.now()).split(".")[0].replace(" ", "_")
                if (self._context is None or self._context.name == "")
                else "{}-{}".format(self._context.name, self._context.uid)
            )

        # Check if a context is available:
        if self._tensorboard_directory is not None:
            # Create the main tensorboard directory:
            os.makedirs(self._tensorboard_directory, exist_ok=True)
            # Index the run name according to the tensorboard directory content:
            index = 1
            for run_directory in sorted(os.listdir(self._tensorboard_directory)):
                existing_run = run_directory.rsplit(
                    "_", 1
                )  # type: List[str] # [0] = name, [1] = index
                if self._run_name == existing_run[0]:
                    index += 1
            # Check if need to index the name:
            if index > 1:
                self._run_name = "{}_{}".format(self._run_name, index)
        else:
            # Try to get the 'tensorboard_dir' parameter:
            self._tensorboard_directory = self._context.get_param("tensorboard_dir")
            if self._tensorboard_directory is None:
                # The parameter was not given, set the directory to the default value:
                self._tensorboard_directory = self._DEFAULT_TENSORBOARD_DIRECTORY.replace(
                    "{{project}}", self._context.project
                )
                try:
                    os.makedirs(self._tensorboard_directory, exist_ok=True)
                except OSError:
                    # The tensorboard default directory is not writable, change to the artifact path:
                    self._tensorboard_directory = self._context.artifact_path

        # Create the output path:
        self._output_path = os.path.join(self._tensorboard_directory, self._run_name)
        os.makedirs(self._output_path, exist_ok=True)

    def _generate_run_start_text(self) -> str:
        """
        Generate the run start summary text. The callback configuration will be written along side the context
        information - a hyperlink for the job in MLRun and the context metadata as strings.

        :return: The generated text.
        """
        # Write the main header:
        text = "###Run Properties"

        # Add the callbacks properties:
        text += "\n####Callback configuration:"
        for property_name, property_value in zip(
            ["Output directory", "Run name", "Tracked hyperparameters"],
            [
                self._output_path,
                self._run_name,
                list(self._static_hyperparameters.keys())
                + list(self._dynamic_hyperparameters.keys()),
            ],
        ):
            text += "\n  * **{}**: {}".format(
                property_name.capitalize(),
                self._markdown_print(value=property_value, tabs=2),
            )

        # Add the context state:
        if self._context is not None:
            text += "\n####Context initial state: ({})".format(
                self._generate_context_link(context=self._context)
            )
            for property_name, property_value in self._extract_properties_from_context(
                context=self._context
            ).items():
                text += "\n  * **{}**: {}".format(
                    property_name.capitalize(),
                    self._markdown_print(value=property_value, tabs=2),
                )

        return text

    def _generate_epoch_text(self) -> str:
        """
        Generate the last epoch summary text. If MLRun context is available, the results and artifacts will be
        displayed. Otherwise, the epochs results will be simply written.

        :return: The generated text.
        """
        text = "####Epoch {} summary:".format(self._epochs)
        if self._context._children[-1] is not None:
            for property_name, property_value in self._extract_properties_from_context(
                context=self._context
            ).items():
                if property_name not in [
                    "project",
                    "uid",
                    "state",
                    "name",
                    "labels",
                    "inputs",
                    "parameters",
                ]:
                    text += "\n  * **{}**: {}".format(
                        property_name.capitalize(),
                        self._markdown_print(value=property_value, tabs=2),
                    )
        else:
            for property_name, property_value in self._extract_epoch_results().items():
                text += "\n  * **{}**: {}".format(
                    property_name.capitalize(),
                    self._markdown_print(value=property_value, tabs=2),
                )
        return text

    def _generate_run_end_text(self) -> str:
        """
        Generate the run end summary text, writing the final collected results and parameters values. If MLRun context
        is available the updated properties of the context will be written as well.

        :return: The generated text.
        """
        # Write the run summary:
        text = "\n####Run final summary - epoch {}:".format(self._epochs)
        for property_name, property_value in self._extract_epoch_results().items():
            text += "\n  * **{}**: {}".format(
                property_name.capitalize(),
                self._markdown_print(value=property_value, tabs=2),
            )

        # Add the context final state:
        if self._context is not None:
            text += "\n####Context final state: ({})".format(
                self._generate_context_link(context=self._context)
            )
            for property_name, property_value in self._extract_properties_from_context(
                context=self._context
            ).items():
                text += "\n  * **{}**: {}".format(
                    property_name.capitalize(),
                    self._markdown_print(value=property_value, tabs=2),
                )
        return text

    def _extract_epoch_results(
        self, epoch: int = -1
    ) -> Dict[str, Dict[str, TrackableType]]:
        """
        Extract the given epoch results from all the collected values and results.

        :param epoch: The epoch to get the results. Defaulted to the last epoch (-1).

        :return: A dictionary where the keys are the collected value and the values are the results.
        """
        return {
            "Static hyperparameters": self._static_hyperparameters,
            "Dynamic hyperparameters": {
                name: value[epoch]
                for name, value in self._dynamic_hyperparameters.items()
            },
            "Training results": {
                name: value[epoch] for name, value in self._training_summaries.items()
            },
            "Validation results": {
                name: value[epoch] for name, value in self._validation_summaries.items()
            },
        }

    @staticmethod
    def _generate_context_link(
        context: MLClientCtx, link_text: str = "view in MLRun"
    ) -> str:
        """
        Generate a hyperlink from the provided context to view in the MLRun web.

        :param context:   The context to generate his link.
        :param link_text: Text to present instead of the link.

        :return: The generated link.
        """
        return '<a href="{}/{}/{}/jobs/monitor/{}/overview" target="_blank">{}</a>'.format(
            config.resolve_ui_url(),
            config.ui.projects_prefix,
            context.project,
            context.uid,
            link_text,
        )

    @staticmethod
    def _extract_properties_from_context(context: MLClientCtx) -> Dict[str, Any]:
        """
        Extract the properties of the run this context belongs to.

        :param context: The context to get his properties.

        :return: The properties as a dictionary where each key is the property name.
        """
        run = mlrun.RunObject.from_dict(context.to_dict())
        runs = mlrun.lists.RunList([run.to_dict()])
        info = {}
        for property_name, property_value in list(zip(*runs.to_rows())):
            info[property_name] = property_value
        return info

    @staticmethod
    def _markdown_print(value: Any, tabs: int = 0):
        """
        Convert the given value into a markdown styled string to print in tensorboard. List and dictionaries will both
        printed as yaml with minor differences.

        :param value: The value to print.
        :param tabs:  Indent to add at the beginning of every line.

        :return: The markdown styled string.
        """
        if isinstance(value, list):
            if len(value) == 0:
                return ""
            text = "\n" + yaml.dump(value)
            text = "  \n".join(["  " * tabs + line for line in text.splitlines()])
            return text
        if isinstance(value, dict):
            if len(value) == 0:
                return ""
            text = yaml.dump(value)
            text = "  \n".join(
                ["  " * tabs + "- " + line for line in text.splitlines()]
            )
            text = "  \n" + text
            return text
        return str(value)
