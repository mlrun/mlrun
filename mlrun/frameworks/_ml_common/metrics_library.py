from abc import ABC
from typing import Dict, List, Type, Union

import sklearn
from sklearn.preprocessing import LabelBinarizer

import mlrun.errors

from .._common.utils import ModelType
from .metric import Metric
from .utils import AlgorithmFunctionality, DatasetType, MetricEntry


class MetricsLibrary(ABC):
    """
    Static class of a collection of metrics to use in training and evaluation of machine learning frameworks.
    """

    _NEED_PROBABILITIES_KEYWORD = "need_probabilities"

    @staticmethod
    def from_list(metrics_list: List[Union[Metric, MetricEntry]]) -> List[Metric]:
        """
        Collect the given metrics configurations from a list. The metrics names will be chosen by the following rules:

        * If a function was given, the function's name.
        * If a callable object was given, the object's class name.
        * If a module path was given, the text after the last dot.
        * Duplicates will be counted and concatenated with their number of duplicates.

        :param metrics_list: The list of metrics. A metric can be a function, callable object, name of an imported
                             function or a module path to import the function. Arguments can be passed as a tuple:
                             (metric, arguments). To specify the metric expects probabilities (calling
                             model.predict_proba instead of model.predict), pass within the arguments:
                             { "need_probabilities": True }.

        :return: A list of metrics objects.
        """
        return [
            metric
            if isinstance(metric, Metric)
            else MetricsLibrary._to_metric_class(metric_entry=metric)
            for metric in metrics_list
        ]

    @staticmethod
    def from_dict(metrics_dictionary: Dict[str, MetricEntry]) -> List[Metric]:
        """
        Collect the given metrics configurations from a dictionary.

        :param metrics_dictionary: A dictionary of metrics where each key is the metric name to be logged with and its
                                   value is the metric itself. A metric can be a function, callable object, name of an
                                   imported function or a module path to import the function. Arguments can be passed
                                   along as a tuple: (metric, arguments). To specify the metric expects probabilities
                                   (calling model.predict_proba instead of model.predict), pass within the arguments:
                                   { "need_probabilities": True }.

        :return: A list of metrics objects.
        """
        return [
            MetricsLibrary._to_metric_class(
                metric_entry=metric, metric_name=metric_name
            )
            for metric_name, metric in metrics_dictionary.items()
        ]

    @classmethod
    def default(cls, model: ModelType, y: DatasetType = None, **kwargs) -> List[Metric]:
        """
        Get the default metrics list of this framework's library.

        :param model: The model to check if its a regression model or a classification model.
        :param y:     The ground truth values to check if its multiclass and / or multi output.

        :return: The default metrics list.
        """
        # Discover the algorithm functionality of the provided model:
        algorithm_functionality = AlgorithmFunctionality.get_algorithm_functionality(
            model=model, y=y
        )

        # Initialize the metrics list:
        metrics = []  # type: List[Metric]

        # Add classification metrics:
        if algorithm_functionality.is_classification():
            metrics += [Metric(name="accuracy", metric=sklearn.metrics.accuracy_score)]
            if (
                algorithm_functionality.is_binary_classification()
                and algorithm_functionality.is_single_output()
            ):
                metrics += [
                    Metric(metric=sklearn.metrics.f1_score),
                    Metric(metric=sklearn.metrics.precision_score),
                    Metric(metric=sklearn.metrics.recall_score),
                ]
            if algorithm_functionality.is_multiclass_classification():
                metrics += [
                    Metric(
                        metric=sklearn.metrics.f1_score,
                        additional_arguments={"average": "macro"},
                    ),
                    Metric(
                        metric=sklearn.metrics.precision_score,
                        additional_arguments={"average": "macro"},
                    ),
                    Metric(
                        metric=sklearn.metrics.recall_score,
                        additional_arguments={"average": "macro"},
                    ),
                ]
                if algorithm_functionality.is_single_output():
                    metrics += [
                        Metric(
                            name="auc-micro",
                            metric=lambda y_true, y_pred: sklearn.metrics.roc_auc_score(
                                LabelBinarizer().fit_transform(y_true),
                                y_pred,
                                multi_class="ovo",
                                average="micro",
                            ),
                            need_probabilities=True,
                        ),
                        Metric(
                            name="auc-macro",
                            metric=sklearn.metrics.roc_auc_score,
                            additional_arguments={
                                "multi_class": "ovo",
                                "average": "macro",
                            },
                            need_probabilities=True,
                        ),
                        Metric(
                            name="auc-weighted",
                            metric=sklearn.metrics.roc_auc_score,
                            additional_arguments={
                                "multi_class": "ovo",
                                "average": "weighted",
                            },
                            need_probabilities=True,
                        ),
                    ]

        # Add regression metrics:
        if algorithm_functionality.is_regression():
            metrics += [
                Metric(metric=sklearn.metrics.mean_absolute_error),
                Metric(metric=sklearn.metrics.r2_score),
                Metric(
                    name="root_mean_squared_error",
                    metric=sklearn.metrics.mean_squared_error,
                    additional_arguments={"squared": False},
                ),
                Metric(metric=sklearn.metrics.mean_squared_error),
            ]

        # Filter out the metrics by probabilities requirement:
        if not hasattr(model, "predict_proba"):
            metrics = [metric for metric in metrics if not metric.need_probabilities]

        return metrics

    @staticmethod
    def _to_metric_class(
        metric_entry: MetricEntry,
        metric_name: str = None,
    ) -> Metric:
        """
        Create a Metric instance from a user given metric entry.

        :param metric_entry: Metric entry as passed inside of a list or a dictionary.
        :param metric_name:  The metric name to use (if passed from a dictionary).

        :return: The metric class instance of this entry.
        """
        # If its a tuple, unpack it to get the additional arguments:
        if isinstance(metric_entry, tuple):
            metric, arguments = metric_entry
        else:
            metric = metric_entry
            arguments = {}

        # Check if the 'need_probabilities' attribute is given:
        if MetricsLibrary._NEED_PROBABILITIES_KEYWORD in arguments:
            need_probabilities = arguments[MetricsLibrary._NEED_PROBABILITIES_KEYWORD]
            arguments.pop(MetricsLibrary._NEED_PROBABILITIES_KEYWORD)
        else:
            need_probabilities = False

        # Initialize the Metric with the collected information and return:
        return Metric(
            metric=metric,
            name=metric_name,
            additional_arguments=arguments,
            need_probabilities=need_probabilities,
        )


# A constant name for the context parameter to use for passing a plans configuration:
METRICS_CONTEXT_PARAMETER = "_metrics"


def get_metrics(
    metrics_library: Type[MetricsLibrary],
    metrics: Union[List[Metric], List[MetricEntry], Dict[str, MetricEntry]] = None,
    context: mlrun.MLClientCtx = None,
    include_default: bool = True,
    **default_kwargs,
) -> List[Metric]:
    """
    Get metrics for a run. The metrics will be taken from the provided metrics / configuration via code, from provided
    configuration via MLRun context and if the 'include_default' is True, from the framework metric library's defaults.

    :param metrics_library: The framework's metrics library class to get its defaults.
    :param metrics:         The metrics parameter passed to the function. Can be passed as a dictionary or a list of
                            metrics.
    :param context:         A context to look in if the configuration was passed as a parameter.
    :param include_default: Whether to include the default in addition to the provided metrics. Defaulted to True.
    :param default_kwargs:  Additional key word arguments to pass to the 'default' method of the given metrics library
                            class.

    :return: The metrics list.

    :raise MLRunInvalidArgumentError: If the metrics were not passed in a list or a dictionary.
    """
    # Setup the plans list:
    parsed_metrics = []  # type: List[Metric]

    # Get the user input metrics:
    metrics_from_context = None
    if context is not None:
        metrics_from_context = context.parameters.get(METRICS_CONTEXT_PARAMETER, None)
    for user_input in [metrics, metrics_from_context]:
        if user_input is not None:
            if isinstance(user_input, dict):
                parsed_metrics += metrics_library.from_dict(
                    metrics_dictionary=user_input
                )
            elif isinstance(user_input, list):
                parsed_metrics += metrics_library.from_list(metrics_list=user_input)
            else:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"The metrics are expected to be in a list or a dictionary. Received: {type(user_input)}. A metric "
                    f"can be a function, callable object, name of an imported function or a module path to import the "
                    f"function. Arguments can be passed as a tuple: in the following form: (metric, arguments). If "
                    f"used in a dictionary, each key will be the name to use for logging the metric."
                )

    # Get the library's default:
    if include_default:
        parsed_metrics += metrics_library.default(**default_kwargs)

    return parsed_metrics
