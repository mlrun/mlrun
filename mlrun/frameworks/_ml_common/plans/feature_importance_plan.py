import pandas as pd
import plotly.graph_objects as go

from mlrun.artifacts import Artifact

from ..plan import MLPlotPlan


class FeatureImportancePlan(MLPlotPlan):
    """
    """

    def is_ready(self, stage: ProductionStages) -> bool:
        return stage == ProductionStages.POST_FIT

    def produce(self, model, apply_args, *args, **kwargs) -> Dict[str, Artifact]:
        """
        Produce the artifact according to this plan.
        :return: The produced artifact.
        """

        # Tree-based feature importance
        if hasattr(model, "feature_importances_"):
            importance_score = model.feature_importances_

        # Coefficient-based importance|
        elif hasattr(model, "coef_"):
            importance_score = model.coef_[0]

        df = pd.DataFrame(
            {
                "features": apply_args["X_train"].columns,
                "feature_importance": importance_score,
            }
        ).sort_values(by="feature_importance", ascending=False)

        fig = go.Figure(
            [go.Bar(x=df["feature_importance"], y=df["features"], orientation="h")]
        )

        # Creating an html rendering of the plot
        plot_as_html = fig.to_html()
        display(HTML(plot_as_html))
