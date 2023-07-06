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
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score, roc_curve

from mlrun.artifacts import Artifact, PlotlyArtifact

from ..plan import MLPlanStages, MLPlotPlan
from ..utils import MLTypes, MLUtils


class ROCCurvePlan(MLPlotPlan):
    """
    Plan for producing a receiver operating characteristic (ROC) - a plot that shows the connection / trade-off between
    clinical sensitivity and specificity for every possible cut-off for a test or a combination of tests.
    """

    _ARTIFACT_NAME = "roc-curves"

    def __init__(
        self,
        pos_label: Union[str, int] = None,
        sample_weight: np.ndarray = None,
        drop_intermediate: bool = True,
        average: str = "macro",
        max_fpr: float = None,
        multi_class: str = "raise",
        labels: List[str] = None,
    ):
        """
        Initialize a receiver operating characteristic plan with the given configuration.

        To read more about the parameters, head to the SciKit-Learn docs at:

        * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        * https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

        :param pos_label:         The label of the positive class. When None, if 'y_true' (y) is in {-1, 1} or {0, 1},
                                  'pos_label' is set to 1, otherwise an error will be raised. Default: None.
        :param sample_weight:     Sample weights to apply.
        :param drop_intermediate: Whether to drop some suboptimal thresholds which would not appear on a plotted ROC
                                  curve. Default: True.
        :param average:           Determines the type of averaging performed on the data. If None, the scores for each
                                  class are returned. Default: "macro".
        :param max_fpr:           For multiclass it should be equal to 1 or None. If not None, the standardized partial
                                  AUC [2] over the range [0, max_fpr] is returned. Default: None.
        :param multi_class:       Only used for multiclass targets. Determines the type of configuration to use. Can be
                                  one of {'raise', 'ovr', 'ovo'}. Default: "raise".
        :param labels:            Only used for multiclass targets. List of labels that index the classes in 'y_pred'.
                                  If None, the labels found in 'y_true' (y) will be used. Default: None.
        """
        # Store the parameters:
        self._pos_label = pos_label
        self._sample_weight = sample_weight
        self._drop_intermediate = drop_intermediate
        self._average = average
        self._max_fpr = max_fpr
        self._multi_class = multi_class
        self._labels = labels

        # Continue the initialization for the MLPlan:
        super(ROCCurvePlan, self).__init__(need_probabilities=True)

    def is_ready(self, stage: MLPlanStages, is_probabilities: bool) -> bool:
        """
        Check whether or not the plan is fit for production by the given stage and prediction probabilities. The
        roc curve is ready only post prediction probabilities.

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
    ) -> Dict[str, Artifact]:
        """
        Produce the roc curve according to the ground truth (y) and predictions (y_pred) values. If predictions are not
        available, the model and a dataset can be given to produce them.

        :param y:      The ground truth values.
        :param y_pred: The predictions values.
        :param model:  Model to produce the predictions.
        :param x:      Input dataset to produce the predictions.

        :return: The produced roc curve artifact in an artifacts dictionary.
        """
        # Calculate the predictions if needed:
        y_pred = self._calculate_predictions(y_pred=y_pred, model=model, x=x)

        # Convert to DataFrame:
        y = MLUtils.to_dataframe(dataset=y)
        y_pred = MLUtils.to_dataframe(dataset=y_pred)

        # One hot encode the labels in order to plot them
        y_one_hot = pd.get_dummies(y, columns=y.columns.to_list())

        # Create an empty figure:
        fig = go.Figure()
        fig.add_shape(type="line", line={"dash": "dash"}, x0=0, x1=1, y0=0, y1=1)

        # Iteratively add new lines every time we compute a new class:
        for i in range(y_pred.shape[1]):
            class_i_true = y_one_hot.iloc[:, i]
            class_i_pred = y_pred.iloc[:, i]
            fpr, tpr, _ = roc_curve(
                class_i_true,
                class_i_pred,
                pos_label=self._pos_label,
                sample_weight=self._sample_weight,
                drop_intermediate=self._drop_intermediate,
            )
            auc_score = roc_auc_score(
                class_i_true,
                class_i_pred,
                average=self._average,
                sample_weight=self._sample_weight,
                max_fpr=self._max_fpr,
                multi_class=self._multi_class,
                labels=self._labels,
            )
            name = f"{y_one_hot.columns[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))

        # Configure the layout:
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            yaxis={"scaleanchor": "x", "scaleratio": 1},
            xaxis={"constrain": "domain"},
            width=700,
            height=500,
        )

        # Creating the plot artifact:
        self._artifacts[self._ARTIFACT_NAME] = PlotlyArtifact(
            key=self._ARTIFACT_NAME,
            figure=fig,
        )

        return self._artifacts
