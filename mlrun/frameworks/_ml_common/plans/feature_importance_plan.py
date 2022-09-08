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
from typing import Dict

import pandas as pd
import plotly.graph_objects as go

import mlrun
from mlrun.artifacts import Artifact, PlotlyArtifact

from ..plan import MLPlanStages, MLPlotPlan
from ..utils import MLTypes, MLUtils


class FeatureImportancePlan(MLPlotPlan):
    """
    Plan for producing a feature importance.
    """

    _ARTIFACT_NAME = "feature-importance"

    def __init__(self):
        """
        Initialize a feature importance plan.

        An example of use can be seen at the Scikit-Learn docs here:
        https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
        """
        super(FeatureImportancePlan, self).__init__()

    def is_ready(self, stage: MLPlanStages, is_probabilities: bool) -> bool:
        """
        Check whether the plan is fit for production by the given stage and prediction probabilities. The feature
        importance is ready post training.

        :param stage:            The stage to check if the plan is ready.
        :param is_probabilities: True if the 'y_pred' that will be sent to 'produce' is a prediction of probabilities
                                 (from 'predict_proba') and False if not.

        :return: True if the plan is producible and False otherwise.
        """
        return stage == MLPlanStages.POST_FIT

    def produce(
        self, model: MLTypes.ModelType, x: MLTypes.DatasetType, **kwargs
    ) -> Dict[str, Artifact]:
        """
        Produce the feature importance according to the given model and dataset ('x').

        :param model:  Model to get its 'feature_importances_' or 'coef_[0]' fields.
        :param x:      Input dataset the model trained on for the column labels.

        :return: The produced feature importance artifact in an artifacts dictionary.
        """
        # Get the importance score:
        if hasattr(model, "feature_importances_"):  # Tree-based feature importance
            importance_score = model.feature_importances_
        elif hasattr(model, "feature_importance"):  # Booster feature importance
            importance_score = model.feature_importance()
        elif hasattr(model, "coef_"):  # Coefficient-based importance
            importance_score = model.coef_[0]
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "This model cannot be used for Feature Importance plotting."
            )

        # Create a table of features and their importance:
        df = pd.DataFrame(
            {
                "features": MLUtils.to_dataframe(dataset=x).columns,
                "feature_importance": importance_score,
            }
        ).sort_values(by="feature_importance", ascending=False)

        # Create the figure:
        fig = go.Figure(
            [go.Bar(x=df["feature_importance"], y=df["features"], orientation="h")]
        )

        # Creating the artifact:
        self._artifacts[self._ARTIFACT_NAME] = PlotlyArtifact(
            key=self._ARTIFACT_NAME,
            figure=fig,
        )

        return self._artifacts
