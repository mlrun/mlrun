from abc import abstractmethod

import pandas as pd
import plotly.graph_objects as go
from IPython.core.display import HTML, display

from mlrun.artifacts import Artifact

from ..._common.plan import Plan


class FeatureImportancePlan(MLPlan):
    """
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._extra_data = {}

    @abstractmethod
    def validate(self, *args, **kwargs):
        """
        Validate this plan has the required data to produce its artifact.
        :raise ValueError: In case this plan is missing information in order to produce its artifact.
        """
        pass

    @abstractmethod
    def is_ready(self, stage: ProductionStages) -> bool:
        return stage == ProductionStages.POST_FIT

    @abstractmethod
    def produce(self, model, apply_args, *args, **kwargs) -> Artifact:
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
