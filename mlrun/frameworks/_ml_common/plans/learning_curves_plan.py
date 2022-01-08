from abc import abstractmethod

import numpy as np
import plotly.graph_objects as go
from IPython.core.display import HTML, display
from sklearn.model_selection import learning_curve

from mlrun.artifacts import Artifact


class LearningCurvesPlan(MLPlan):
    """
    SkLearn Learning Curves
    """

    def __init__(self, cv=3, **kwargs):

        # Learning Curves parameters
        self._cv = cv

        # Other
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

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            model,
            apply_args["X_train"],
            apply_args["y_train"].values.ravel(),
            cv=self._cv,
            return_times=True,
        )

        fig = go.Figure(
            data=[go.Scatter(x=train_sizes.tolist(), y=np.mean(train_scores, axis=1))],
            layout=dict(title=dict(text="Learning Curves")),
        )

        # add custom xaxis title
        fig.add_annotation(
            dict(
                font=dict(color="black", size=14),
                x=0.5,
                y=-0.15,
                showarrow=False,
                text="Train Size",
                xref="paper",
                yref="paper",
            )
        )

        # add custom yaxis title
        fig.add_annotation(
            dict(
                font=dict(color="black", size=14),
                x=-0.1,
                y=0.5,
                showarrow=False,
                text="Score",
                textangle=-90,
                xref="paper",
                yref="paper",
            )
        )

        # adjust margins to make room for yaxis title
        fig.update_layout(margin=dict(t=100, l=100), width=800, height=500)

        # Creating an html rendering of the plot
        plot_as_html = fig.to_html()
        display(HTML(plot_as_html))
