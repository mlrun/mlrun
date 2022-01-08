from typing import Dict

import numpy as np
from IPython.core.display import HTML, display
from plotly.figure_factory import create_annotated_heatmap
from sklearn.metrics import confusion_matrix

from mlrun.artifacts import PlotlyArtifact


class ConfusionMatrixPlan(MLPlan):

    _ARTIFACT_NAME = "confusion_matrix"

    def __init__(
        self, labels=None, sample_weight=None, normalize=None, y_test=None, y_pred=None
    ):
        # confusion_matrix() parameters
        self._labels = labels
        self._sample_weight = sample_weight
        self._normalize = normalize
        super(ConfusionMatrixPlan, self).__init__(y_test=y_test, y_pred=y_pred)

    def is_ready(self, stage: PlanStages) -> bool:
        return stage == PlanStages.POST_EVALUATION

    def produce(self, y_test, y_pred, **kwargs) -> Dict[str, PlotlyArtifact]:
        """
        Produce the artifact according to this plan.
        :return: The produced artifact.
        """
        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=self._labels,
            sample_weight=self._sample_weight,
            normalize=self._normalize,
        )

        x = np.sort(y_test[y_test.columns[0]].unique()).tolist()

        # set up figure
        figure = create_annotated_heatmap(
            cm, x=x, y=x, annotation_text=cm.astype(str), colorscale="Blues"
        )

        # add title
        figure.update_layout(title_text="Confusion matrix",)

        # add custom xaxis title
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

        # add custom yaxis title
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

        figure.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
        figure.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

        # adjust margins to make room for yaxis title
        figure.update_layout(margin=dict(t=100, l=100), width=500, height=500)

        # add colorbar
        figure["data"][0]["showscale"] = True
        figure["layout"]["yaxis"]["autorange"] = "reversed"

        # Creating an html rendering of the plot
        self._artifacts[self._ARTIFACT_NAME] = PlotlyArtifact(
            figure=figure, key=self._ARTIFACT_NAME
        )
        return self._artifacts

    def display(self):
        if self._artifacts:
            display(HTML(self._artifacts[self._ARTIFACT_NAME].get_body()))
        else:
            super(ConfusionMatrixPlan, self).display()
