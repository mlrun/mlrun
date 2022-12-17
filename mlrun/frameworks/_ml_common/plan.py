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
import json
from abc import ABC, abstractmethod
from enum import Enum

from IPython.core.display import HTML, display

import mlrun

from .._common import Plan
from .utils import MLTypes


class MLPlanStages(Enum):
    """
    Stages for a machine learning plan to be produced.
    """

    # SciKit-Learn's API:
    PRE_FIT = "pre_fit"
    POST_FIT = "post_fit"
    PRE_PREDICT = "pre_predict"
    POST_PREDICT = "post_predict"

    # Boosting API:
    PRE_TRAIN = "pre_train"
    POST_TRAIN = "post_train"
    PRE_ITERATION = "pre_iteration"
    POST_ITERATION = "post_iteration"


class MLPlan(Plan, ABC):
    """
    An abstract class for describing a ML plan. A ML plan is used to produce artifact manually or in a given time using
    the MLLogger.
    """

    def __init__(self, need_probabilities: bool = False):
        """
        Initialize a new ML plan.

        :param need_probabilities: Whether this plan will need the predictions return from 'model.predict()' or
                                   'model.predict_proba()'. True means predict_proba and False predict. Default:
                                   False.
        """
        self._need_probabilities = need_probabilities
        super(MLPlan, self).__init__()

    @property
    def need_probabilities(self) -> bool:
        """
        Whether this plan require predictions returned from 'model.predict()' or 'model.predict_proba()'.

        :return: True if predict_proba and False if predict.
        """
        return self._need_probabilities

    @abstractmethod
    def is_ready(self, stage: MLPlanStages, is_probabilities: bool) -> bool:
        """
        Check whether or not the plan is fit for production by the given stage and prediction probabilities.

        :param stage:            The stage to check if the plan is ready.
        :param is_probabilities: True if the 'y_pred' that will be sent to 'produce' is a prediction of probabilities
                                 (from 'predict_proba') and False if not.

        :return: True if the plan is producible and False otherwise.
        """
        pass


class MLPlotPlan(MLPlan, ABC):
    """
    An abstract class for describing a ML plan for plots. A ML plan is used to produce artifact manually or in a given
    time using the MLLogger.
    """

    def _cli_display(self):
        """
        Print the logged artifacts names and their URIs.
        """
        print(
            json.dumps(
                {name: artifact.uri for name, artifact in self._artifacts.items()}
            )
        )

    def _gui_display(self):
        """
        Plot the plot.
        """
        for artifact in self._artifacts.values():
            if artifact.kind == "plotly":
                display(HTML(artifact.get_body()))

    def _calculate_predictions(
        self,
        y_pred: MLTypes.DatasetType = None,
        model: MLTypes.ModelType = None,
        x: MLTypes.DatasetType = None,
    ) -> MLTypes.DatasetType:
        """
        Calculate the predictions using the model and input dataset only if the predictions (y_pred) were not provided.

        :param y_pred: Predictions if available to not run predict / predict_proba.
        :param model:  The model to predict with.
        :param x:      The input dataset to predict on.

        :return: The predictions.

        :raise MLRunInvalidArgumentError: If model or x were not provided to calculate the predictions.
        """
        if y_pred is None:
            if model is None or x is None:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "In order to produce a confusion matrix artifact, the predictions must be passed. If predictions "
                    "are not available, the model and a dataset (x) must be given to produce them."
                )
            if self._need_probabilities:
                y_pred = (
                    model.original_predict_proba(x)
                    if hasattr(model, "original_predict_proba")
                    else model.predict_proba(x)
                )
            else:
                y_pred = (
                    model.original_predict(x)
                    if hasattr(model, "original_predict")
                    else model.predict(x)
                )

        return y_pred
