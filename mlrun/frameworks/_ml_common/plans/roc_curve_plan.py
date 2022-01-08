from abc import abstractmethod

import pandas as pd
import plotly.graph_objects as go
from IPython.core.display import HTML, display
from sklearn.metrics import roc_auc_score, roc_curve


class ROCCurvePlan(MLPlan):
    """
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._extra_data = {}

    @abstractmethod
    def validate(self, *args, **kwargs):
        pass

    @abstractmethod
    def is_ready(self, stage: ProductionStages) -> bool:
        return stage == ProductionStages.POST_EVALUATION

    @abstractmethod
    def produce(self, model, context, apply_args, plots_artifact_path="", **kwargs):
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(apply_args["X_test"])

        # One hot encode the labels in order to plot them
        y_onehot = pd.get_dummies(apply_args["y_test"], columns=model.classes_)

        # Create an empty figure, and iteratively add new lines
        # every time we compute a new class
        fig = go.Figure()
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

        for i in range(y_scores.shape[1]):
            y_true = y_onehot.iloc[:, i]
            y_score = y_scores[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)

            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))

        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain="domain"),
            width=700,
            height=500,
        )

        # Creating an html rendering of the plot
        plot_as_html = fig.to_html()
        display(HTML(plot_as_html))
