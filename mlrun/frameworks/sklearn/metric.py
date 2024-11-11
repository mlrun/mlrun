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
import importlib
import json
import sys
from typing import Callable, Optional, Union

import mlrun.errors

from .utils import SKLearnTypes


class Metric:
    """
    A metric handling class to call a metric with additional keyword arguments later during a run and log the results to
    MLRun.
    """

    def __init__(
        self,
        metric: Union[Callable, str],
        name: Optional[str] = None,
        additional_arguments: Optional[dict] = None,
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
        self._result = None  # type: Union[float, None]

    def __call__(
        self,
        y_true: SKLearnTypes.DatasetType,
        y_pred: SKLearnTypes.DatasetType = None,
        model: SKLearnTypes.ModelType = None,
        x: SKLearnTypes.DatasetType = None,
    ) -> float:
        """
        Call the metric function on the provided y_true and y_pred values using the stored additional arguments.

        :param y_true: The ground truth values.
        :param y_pred: The model predictions.

        :return: The metric result.
        """
        # Run a prediction if 'y_pred' was not given:
        if y_pred is None:
            if model is None or x is None:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "Calculating a metric requires the model's predictions / probabilities (y_pred) or the model "
                    "itself and an input (x) to run 'predict' / 'predict_proba'."
                )
            y_pred = (
                model.predict_proba(x) if self._need_probabilities else model.predict(x)
            )

        # Calculate the result and return:
        self._result = self._metric(y_true, y_pred, **self._arguments)
        return self._result

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

    @property
    def result(self) -> Union[float, None]:
        """
        Get the metric result. If the metric was not calculated, None will be returned.

        :return: The metric result.
        """
        return self._result

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

    def display(self, full: bool = False):
        """
        Display the metric and its result.

        :param full: Whether to print a full display of the metric, including the metric arguments. Default: False.
        """
        result = self._result if self._result is not None else "?"
        print(f"{self._name} = {result}")
        if full:
            print(f"Arguments: {json.dumps(self._arguments, indent=4)}")

    def _repr_pretty_(self, p, cycle: bool):
        """
        A pretty representation of the metric. Will be called by the IPython kernel. This method will call the metric
        display method.

        :param p:     A RepresentationPrinter instance.
        :param cycle: If a cycle is detected to prevent infinite loop.
        """
        self.display()

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
            f"imported. In order to import it during the run, please provide the full module path to the "
            f"metric. For example: 'module.sub_module.metric' will be parsed as "
            f"from module.sub_module import metric."
        )
