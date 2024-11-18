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
from abc import ABC
from typing import Optional, Union

import sklearn
from sklearn.preprocessing import LabelBinarizer

import mlrun.errors

from .._ml_common import MLUtils
from .metric import Metric
from .utils import SKLearnTypes


class MetricsLibrary(ABC):
    """
    Static class for getting and parsing metrics to use in training and evaluation of SciKit-Learn.
    """

    # A constant name for the context parameter to use for passing a metrics configuration:
    CONTEXT_PARAMETER = "_metrics"

    # A keyword to add in case the metric is based on predictions probabilities (not final predictions):
    _NEED_PROBABILITIES_KEYWORD = "need_probabilities"

    @classmethod
    def get_metrics(
        cls,
        metrics: Optional[
            Union[
                list[Metric],
                list[SKLearnTypes.MetricEntryType],
                dict[str, SKLearnTypes.MetricEntryType],
            ]
        ] = None,
        context: mlrun.MLClientCtx = None,
        include_default: bool = True,
        **default_kwargs,
    ) -> list[Metric]:
        """
        Get metrics for a run. The metrics will be taken from the provided metrics / configuration via code, from
        provided configuration via MLRun context and if the 'include_default' is True, from the metric library's
        defaults as well.

        :param metrics:         The metrics parameter passed to the function. Can be passed as a dictionary or a list of
                                metrics.
        :param context:         A context to look in if the configuration was passed as a parameter.
        :param include_default: Whether to include the default in addition to the provided metrics. Default: True.
        :param default_kwargs:  Additional key word arguments to pass to the 'default' method of the given metrics
                                library class.

        :return: The metrics list.

        :raise MLRunInvalidArgumentError: If the metrics were not passed in a list or a dictionary.
        """
        # Set up the plans list:
        parsed_metrics = []  # type: List[Metric]

        # Get the metrics passed via context:
        if context is not None and cls.CONTEXT_PARAMETER in context.parameters:
            parsed_metrics += cls._parse(
                metrics=context.parameters.get(cls.CONTEXT_PARAMETER, None)
            )

        # Get the user's set metrics:
        if metrics is not None:
            parsed_metrics += cls._parse(metrics=metrics)

        # Get the library's default:
        if include_default:
            parsed_metrics += cls._default(**default_kwargs)

        return parsed_metrics

    @classmethod
    def _parse(
        cls,
        metrics: Union[
            list[Metric],
            list[SKLearnTypes.MetricEntryType],
            dict[str, SKLearnTypes.MetricEntryType],
        ],
    ) -> list[Metric]:
        """
        Parse the given metrics by the possible rules of the framework implementing.

        :param metrics: A collection of metrics to parse.

        :return: The parsed metrics to use in training / evaluation.
        """
        # Parse from dictionary:
        if isinstance(metrics, dict):
            return cls._from_dict(metrics_dictionary=metrics)

        # Parse from list:
        if isinstance(metrics, list):
            return cls._from_list(metrics_list=metrics)

        raise mlrun.errors.MLRunInvalidArgumentError(
            f"The metrics are expected to be in a list or a dictionary. Received: {type(metrics)}. A metric can be a "
            f"function, callable object, name of an imported function or a module path to import the function. "
            f"Arguments can be passed as a tuple: in the following form: (metric, arguments). If used in a dictionary, "
            f"each key will be the name to use for logging the metric."
        )

    @classmethod
    def _from_list(
        cls, metrics_list: list[Union[Metric, SKLearnTypes.MetricEntryType]]
    ) -> list[Metric]:
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
            else cls._to_metric_class(metric_entry=metric)
            for metric in metrics_list
        ]

    @classmethod
    def _from_dict(
        cls, metrics_dictionary: dict[str, SKLearnTypes.MetricEntryType]
    ) -> list[Metric]:
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
            cls._to_metric_class(metric_entry=metric, metric_name=metric_name)
            for metric_name, metric in metrics_dictionary.items()
        ]

    @classmethod
    def _default(
        cls, model: SKLearnTypes.ModelType, y: SKLearnTypes.DatasetType = None
    ) -> list[Metric]:
        """
        Get the default metrics list according to the algorithm functionality.

        :param model: The model to check if its a regression model or a classification model.
        :param y:     The ground truth values to check if its multiclass and / or multi output.

        :return: The default metrics list.
        """
        # Discover the algorithm functionality of the provided model:
        algorithm_functionality = MLUtils.get_algorithm_functionality(model=model, y=y)

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

    @classmethod
    def _to_metric_class(
        cls,
        metric_entry: SKLearnTypes.MetricEntryType,
        metric_name: Optional[str] = None,
    ) -> Metric:
        """
        Create a Metric instance from a user given metric entry.

        :param metric_entry: Metric entry as passed inside a list or a dictionary.
        :param metric_name:  The metric name to use (if passed from a dictionary).

        :return: The metric class instance of this entry.
        """
        # If it's a tuple, unpack it to get the additional arguments:
        if isinstance(metric_entry, tuple):
            metric, arguments = metric_entry
        else:
            metric = metric_entry
            arguments = {}

        # Check if the 'need_probabilities' attribute is given:
        if cls._NEED_PROBABILITIES_KEYWORD in arguments:
            need_probabilities = arguments[cls._NEED_PROBABILITIES_KEYWORD]
            arguments.pop(cls._NEED_PROBABILITIES_KEYWORD)
        else:
            need_probabilities = False

        # Initialize the Metric with the collected information and return:
        return Metric(
            metric=metric,
            name=metric_name,
            additional_arguments=arguments,
            need_probabilities=need_probabilities,
        )
