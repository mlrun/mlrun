import importlib
import sys
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Callable, Dict, List, Tuple, Type, Union

import mlrun.errors

# Type for a metric entry, can be passed as the metric function itself, as a callable object, a string of the name of
# the function and the full module path to the function to import. Arguments to use when calling the metric can be
# joined by wrapping it as a tuple:
MetricEntry = Union[Tuple[Union[Callable, str], dict], Callable, str]


class Metric:
    """
    A metric handling class to call a metric with additional keyword arguments later during a run and log the results to
    MLRun.
    """

    def __init__(
        self,
        metric: Union[Callable, str],
        name: str = None,
        additional_arguments: dict = None,
        need_probabilities: bool = False,
    ):
        """
        Initialize a metric object to be used with the MLRun logger.

        :param metric:               The metric to use. Can be passed as a string of an imported function or a full
                                     module path to import from.
        :param name:                 The metric name to use for logging it to MLRun.
        :param additional_arguments: Additional arguments to pass for the metric function when calculating it.
        :param need_probabilities:   Whether this metric expects 'y_pred' to be from the 'predict_proba' method or from
                                     'predict'.
        """
        self._metric = (
            self._from_string(metric=metric) if isinstance(metric, str) else metric
        )
        self._arguments = {} if additional_arguments is None else additional_arguments
        self._need_probabilities = need_probabilities
        self._name = name if name is not None else self._get_default_name()

    def __call__(self, y_true, y_pred) -> float:
        """
        Call the metric function on the provided y_true and y_pred values using the stored additional arguments.

        :param y_true: The ground truth values.
        :param y_pred: The model predictions.

        :return: The metric result.
        """
        return self._metric(y_true, y_pred, **self._arguments)

    @property
    def name(self) -> str:
        """
        Get the name of the metric.

        :return: The name of the metric.
        """
        return self._name

    @property
    def need_probabilities(self) -> bool:
        """
        Return whether this metric expects 'y_pred' to be from the 'model.predict_proba' method or from 'model.predict'.

        :return: True if probabilities are required and False if not.
        """
        return self._need_probabilities

    def _get_default_name(self) -> str:
        """
        Get the default name for this metric by the following rules:

        * If metric is a function, the function's name.
        * If metric is a callable object, the object's class name.

        :return: The metric default name.
        """
        # Function object have __name__ where classes instances have __class__.__name__:
        name = getattr(self._metric, "__name__", None)
        return name if name is not None else self._metric.__class__.__name__

    @staticmethod
    def _from_string(metric: str) -> Callable:
        """
        Look for the metric by name in the globally imported objects. If the given metric is a full module path, it will
        be imported from the path.

        :param metric: The metric name or a full module path to import the metric.

        :return: The imported metric function.

        :raise MLRunInvalidArgumentError: If the metric name was not found within the global imports.
        """
        # Check if the metric is inside a module path:
        module = None  # type: Union[ModuleType, str, None]
        if "." in metric:
            module, metric = metric.rsplit(".", 1)

        # Look for the metric in the globals dictionary (it was imported before):
        if metric in globals():
            return globals()[metric]

        # Import the metric from the given module:
        if module is not None:
            # Check if the module is already imported:
            if module in sys.modules:
                # It is already imported:
                module = sys.modules[module]
            else:
                # Import the module:
                module = importlib.import_module(module)
            imported_metric = getattr(module, metric)
            globals().update({metric: imported_metric})
            return imported_metric

        # Metric string was not provided properly:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"The metric {metric} was not found in the global imports dictionary meaning it was not "
            f"imported. In order to import it during the run, please provide the full module path to the"
            f"metric. For example: 'module.sub_module.metric' will be parsed as "
            f"from module.sub_module import metric."
        )


class MetricsLibrary(ABC):
    """
    Static class of a collection of metrics to use in training and evaluation of machine learning frameworks.
    """

    _NEED_PROBABILITIES_KEYWORD = "need_probabilities"

    @staticmethod
    def from_list(metrics_list: List[MetricEntry]) -> List[Metric]:
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
            MetricsLibrary._to_metric_class(metric_entry=metric)
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
    @abstractmethod
    def default(cls, **kwargs) -> List[Metric]:
        """
        Get the default artifacts plans list of this framework's library.

        :return: The default metrics list.
        """
        pass

    @staticmethod
    def _to_metric_class(metric_entry: MetricEntry, metric_name: str = None,) -> Metric:
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
METRICS_CONTEXT_PARAMETER = "metrics"


def get_metrics(
    metrics_library: Type[MetricsLibrary],
    metrics: Union[List[MetricEntry], Dict[str, MetricEntry]] = None,
    context: mlrun.MLClientCtx = None,
    **default_kwargs,
) -> List[Metric]:
    """
    Get metrics for a run by the following priority:

    1. Provided metrics via code.
    2. Provided configuration via MLRun context.
    3. The framework metrics library's defaults.

    :param metrics_library: The framework's metrics library class to get its defaults.
    :param metrics:         The metrics parameter passed to the function. Can be passed as a dictionary or a list of
                            metrics.
    :param context:         A context to look in if the configuration was passed as a parameter.
    :param default_kwargs:  Additional key word arguments to pass to the 'default' method of the given metrics library
                            class.

    :return: The metrics list by the priority mentioned above.

    :raise MLRunInvalidArgumentError: If the metrics were not passed in a list or a dictionary.
    """
    # Look for available metrics prioritizing provided argument to the function over parameter passed via the context:
    if metrics is None and context is not None:
        context_parameters = context.parameters
        if METRICS_CONTEXT_PARAMETER in context_parameters:
            metrics = context.parameters[METRICS_CONTEXT_PARAMETER]

    # Parse the metrics if available:
    if metrics is not None:
        if isinstance(metrics, dict):
            return metrics_library.from_dict(metrics_dictionary=metrics)
        elif isinstance(metrics, list):
            return metrics_library.from_list(metrics_list=metrics)
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"The metrics are expected to be in a list or a dictionary. Received: {type(metrics)}. A metric can be "
                f"a function, callable object, name of an imported function or a module path to import the function. "
                f"Arguments can be passed as a tuple: in the following form: (metric, arguments). If used in a "
                f"dictionary, each key will be the name to use for logging the metric."
            )

    # Return the library's default:
    return metrics_library.default(**default_kwargs)
