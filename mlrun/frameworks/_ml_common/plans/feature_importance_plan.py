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
from typing import Union

import numpy as np
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
        super().__init__()

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
    ) -> dict[str, Artifact]:
        """
        Produce the feature importance according to the given model and dataset ('x').

        :param model: Model to get its 'feature_importances_' or 'coef_' fields.
        :param x:     Input dataset the model trained on for the column labels.

        :return: The produced feature importance artifact in an artifacts dictionary.
        """
        # Get the importance score:
        importance_score = self._get_importance_score(model=model)
        if importance_score is None:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "This model cannot be used for Feature Importance plotting."
            )

        # Create a table of features and their importance along the assigned class:
        features = MLUtils.to_dataframe(dataset=x).columns
        targets = model.classes_ if hasattr(model, "classes_") else [0]
        if len(targets) != len(importance_score):
            targets = [0]

        # Create the figure:
        fig = go.Figure(
            [
                go.Bar(
                    x=class_importance_score,
                    y=[str(f) for f in features],
                    orientation="h",
                    name=str(target),
                )
                for class_importance_score, target in zip(importance_score, targets)
            ]
        )

        # Creating the artifact:
        self._artifacts[self._ARTIFACT_NAME] = PlotlyArtifact(
            key=self._ARTIFACT_NAME,
            figure=fig,
        )

        return self._artifacts

    @staticmethod
    def _get_importance_score(model: MLTypes.ModelType) -> Union[np.ndarray, None]:
        """
        Get the features importance score of the model. If the model do not hold the scores, None will be returned.

        :param model: Model to get its 'feature_importances_' or 'coef_' fields.

        :return: The feature importance scores if available and None otherwise.
        """
        # Set the score variable:
        importance_score: np.ndarray = None

        # Look for the importance score data:
        if hasattr(model, "feature_importances_"):
            # Tree-based feature importance:
            importance_score = model.feature_importances_
        elif hasattr(model, "feature_importance"):
            # Booster feature importance:
            importance_score = model.feature_importance()
        elif hasattr(model, "coef_"):
            # Coefficient-based importance:
            importance_score = model.coef_

        # If found, return it in two dimensions (first one for the classes amount):
        if importance_score is not None:
            if len(importance_score.shape) == 1:
                return [importance_score]
            return importance_score

        # If not found, check for inner estimator/s:
        if hasattr(model, "estimator_"):
            return FeatureImportancePlan._get_importance_score(model=model.estimator_)
        elif hasattr(model, "estimators_"):
            return np.vstack(
                [
                    FeatureImportancePlan._get_importance_score(model=estimator)
                    for estimator in model.estimators_
                ]
            )

        # No feature importance or coefficients are available for the given model:
        return None
