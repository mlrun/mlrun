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

from typing import Optional

import numpy as np
import pandas as pd
from plotly.figure_factory import create_annotated_heatmap
from sklearn.metrics import confusion_matrix

from mlrun.artifacts import Artifact, PlotlyArtifact

from ..plan import MLPlanStages, MLPlotPlan
from ..utils import MLTypes, MLUtils


class ConfusionMatrixPlan(MLPlotPlan):
    """
    Plan for producing a confusion matrix.
    """

    _ARTIFACT_NAME = "confusion-matrix"

    def __init__(
        self,
        labels: np.ndarray = None,
        sample_weight: np.ndarray = None,
        normalize: Optional[str] = None,
    ):
        """
        Initialize a confusion matrix plan with the given configuration.

        To read more about the parameters, head to the SciKit-Learn docs at:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

        :param labels:        Array of labels to index the matrix. This may be used to reorder or select a subset of the
                              labels.
        :param sample_weight: Sample weights to apply.
        :param normalize:     One of {'true', 'pred', 'all'} to normalize the confusion matrix over the true values
                              (rows), the predicted values (columns) or all. If None, confusion matrix will not be
                              normalized. Default: None.
        """
        # Store the parameters:
        self._labels = labels
        self._sample_weight = sample_weight
        self._normalize = normalize

        # Continue the initialization for the MLPlan:
        super().__init__()

    def is_ready(self, stage: MLPlanStages, is_probabilities: bool) -> bool:
        """
        Check whether the plan is fit for production by the given stage and prediction probabilities. The confusion
        matrix is ready only post prediction.

        :param stage:            The stage to check if the plan is ready.
        :param is_probabilities: True if the 'y_pred' that will be sent to 'produce' is a prediction of probabilities
                                 (from 'predict_proba') and False if not.

        :return: True if the plan is producible and False otherwise.
        """
        return stage == MLPlanStages.POST_PREDICT and not is_probabilities

    def produce(
        self,
        y: MLTypes.DatasetType,
        y_pred: MLTypes.DatasetType = None,
        model: MLTypes.ModelType = None,
        x: MLTypes.DatasetType = None,
        **kwargs,
    ) -> dict[str, Artifact]:
        """
        Produce the confusion matrix according to the ground truth (y) and predictions (y_pred) values. If predictions
        are not available, the model and a dataset can be given to produce them.

        :param y:      The ground truth values.
        :param y_pred: The predictions values.
        :param model:  Model to produce the predictions.
        :param x:      Input dataset to produce the predictions.

        :return: The produced confusion matrix artifact in an artifacts dictionary.
        """
        # Calculate the predictions if needed:
        y_pred = self._calculate_predictions(y_pred=y_pred, model=model, x=x)

        # Convert to DataFrame:
        y = MLUtils.to_dataframe(dataset=y)
        y_pred = MLUtils.to_dataframe(dataset=y_pred)

        # Set the labels array it not set:
        if self._labels is None:
            labels = pd.concat((y[y.columns[0]], y_pred[y_pred.columns[0]]))
            self._labels = np.sort(labels.unique()).tolist()

        # Calculate the confusion matrix:
        cm = confusion_matrix(
            y,
            y_pred,
            labels=self._labels,
            sample_weight=self._sample_weight,
            normalize=self._normalize,
        )

        # Initialize a plotly figure according to the targets (classes):
        figure = create_annotated_heatmap(
            cm,
            x=self._labels,
            y=self._labels,
            annotation_text=cm.astype(str),
            colorscale="Blues",
        )

        # Add title:
        figure.update_layout(
            title_text="Confusion matrix",
        )

        # Add custom x-axis title:
        figure.add_annotation(
            dict(
                font=dict(color="black", size=14),
                x=0.5,
                y=-0.1,
                showarrow=False,
                text="Predicted value",
                xref="paper",
                yref="paper",
            )
        )
        figure.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

        # Add custom y-axis title:
        figure.add_annotation(
            dict(
                font=dict(color="black", size=14),
                x=-0.2,
                y=0.5,
                showarrow=False,
                text="Real value",
                textangle=-90,
                xref="paper",
                yref="paper",
            )
        )
        figure.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

        # Adjust margins to make room for the y-axis title:
        figure.update_layout(margin=dict(t=100, l=100), width=500, height=500)

        # Add a color bar:
        figure["data"][0]["showscale"] = True
        figure["layout"]["yaxis"]["autorange"] = "reversed"

        # Create the plot's artifact:
        self._artifacts[self._ARTIFACT_NAME] = PlotlyArtifact(
            key=self._ARTIFACT_NAME,
            figure=figure,
        )

        return self._artifacts
