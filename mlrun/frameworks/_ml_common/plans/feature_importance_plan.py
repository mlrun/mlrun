import pandas as pd
import plotly.graph_objects as go

from mlrun.artifacts import Artifact

from ..plan import MLPlotPlan


class FeatureImportance(MLPlotPlan):
    """
    Plot Feature Importances within a dataset.
    """

    _ARTIFACT_NAME = "feature_importance"

    def __init__(
            self,
            model=None,
            X_train=None,
    ):
        """
        :param model: any model pre-fit or post-fit.
        :param X_train: train dataset.
        """

        super(FeatureImportance, self).__init__(model=model, X_train=X_train)

    def is_ready(self, stage: MLPlanStages, is_probabilities: bool) -> bool:
        """
        Check whether or not the plan is fit for production by the given stage and prediction probabilities. The
        confusion matrix is ready only post prediction.
        :param stage:            The stage to check if the plan is ready.
        :param is_probabilities: True if the 'y_pred' that will be sent to 'produce' is a prediction of probabilities
                                 (from 'predict_proba') and False if not.
        :return: True if the plan is producible and False otherwise.
        """
        return stage == MLPlanStages.PRE_FIT and not is_probabilities

    def produce(self, model, X_train, **kwargs) -> Dict[str, PlotlyArtifact]:
        """
        Produce the artifact according to this plan.
        :return: The produced artifact.
        """
        validate_numerical(X_train)
        if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):

            # Tree-based feature importance
            if hasattr(model, "feature_importances_"):
                importance_score = model.feature_importances_

            else:
                # Coefficient-based importance
                importance_score = model.coef_[0]

            df = pd.DataFrame(
                {
                    "features": X_train.columns,
                    "feature_importance": importance_score,
                }
            ).sort_values(by="feature_importance", ascending=False)

            fig = go.Figure(
                [go.Bar(x=df["feature_importance"], y=df["features"], orientation="h")]
            )

            # Creating an html rendering of the plot
            self._artifacts[self._ARTIFACT_NAME] = PlotlyArtifact(
                figure=fig, key=self._ARTIFACT_NAME
            )
            return self._artifacts
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "This model cannot be used for Feature Importance plotting."
            )