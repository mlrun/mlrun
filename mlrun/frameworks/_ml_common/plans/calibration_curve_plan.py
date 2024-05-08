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

import plotly.graph_objects as go
from sklearn.calibration import calibration_curve

from mlrun.artifacts import Artifact, PlotlyArtifact

from ..plan import MLPlanStages, MLPlotPlan
from ..utils import MLTypes


class CalibrationCurvePlan(MLPlotPlan):
    """
    Plan for producing a calibration curve - computed true and predicted probabilities for a calibration curve. The
    method assumes the inputs come from a binary classifier, and discretize the [0, 1] interval into bins.
    """

    _ARTIFACT_NAME = "calibration-curve"

    def __init__(
        self,
        n_bins: int = 5,
        strategy: str = "uniform",
    ):
        """
        Initialize a calibration curve plan with the given configuration.

        To read more about the parameters, head to the SciKit-Learn docs at:
        https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html

        :param n_bins:    Number of bins to discretize the [0, 1] interval.
        :param strategy:  Strategy used to define the widths of the bins. Can be on of {‘uniform’, ‘quantile’}.
                          Default: "uniform".
        """
        # Store the parameters:
        self._n_bins = n_bins
        self._strategy = strategy

        # Continue the initialization for the MLPlan:
        super().__init__(need_probabilities=True)

    def is_ready(self, stage: MLPlanStages, is_probabilities: bool) -> bool:
        """
        Check whether or not the plan is fit for production by the given stage and prediction probabilities. The
        calibration curve is ready only post prediction.

        :param stage:            The stage to check if the plan is ready.
        :param is_probabilities: True if the 'y_pred' that will be sent to 'produce' is a prediction of probabilities
                                 (from 'predict_proba') and False if not.

        :return: True if the plan is producible and False otherwise.
        """
        return stage == MLPlanStages.POST_PREDICT and is_probabilities

    def produce(
        self,
        y: MLTypes.DatasetType,
        y_pred: MLTypes.DatasetType = None,
        model: MLTypes.ModelType = None,
        x: MLTypes.DatasetType = None,
        **kwargs,
    ) -> dict[str, Artifact]:
        """
        Produce the calibration curve according to the ground truth (y) and predictions (y_pred) values. If predictions
        are not available, the model and a dataset can be given to produce them.

        :param y:      The ground truth values.
        :param y_pred: The predictions values.
        :param model:  Model to produce the predictions.
        :param x:      Input dataset to produce the predictions.

        :return: The produced calibration curve artifact in an artifacts dictionary.
        """
        # Calculate the calibration curve:
        prob_true, prob_pred = calibration_curve(
            y,
            y_pred[:, -1],  # Take only the second class probabilities (1, not 0).
            n_bins=self._n_bins,
            strategy=self._strategy,
        )

        # Create the figure:
        fig = go.Figure(
            data=[go.Scatter(x=prob_true, y=prob_pred)],
            layout={"title": {"text": "Calibration Curve"}},
        )

        # Add custom x-axis title:
        fig.add_annotation(
            {
                "font": {"color": "black", "size": 14},
                "x": 0.5,
                "y": -0.15,
                "showarrow": False,
                "text": "prob_true",
                "xref": "paper",
                "yref": "paper",
            }
        )

        # Add custom y-axis title:
        fig.add_annotation(
            {
                "font": {"color": "black", "size": 14},
                "x": -0.1,
                "y": 0.5,
                "showarrow": False,
                "text": "prob_pred",
                "textangle": -90,
                "xref": "paper",
                "yref": "paper",
            }
        )

        # Adjust margins to make room for yaxis title
        fig.update_layout(margin={"t": 100, "l": 100}, width=800, height=500)

        # Creating the artifact:
        self._artifacts[self._ARTIFACT_NAME] = PlotlyArtifact(
            key=self._ARTIFACT_NAME,
            figure=fig,
        )

        return self._artifacts
