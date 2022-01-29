from abc import abstractmethod

import pandas as pd
import plotly.graph_objects as go
from IPython.core.display import HTML, display
from sklearn.metrics import roc_auc_score, roc_curve


class ROCCurves(MLPlotPlan):
    """
    Plot Receiver operating characteristic (ROC). Shows in a graphical way the connection/trade-off between clinical
    sensitivity and specificity for every possible cut-off for a test or a combination of tests.
    """

    _ARTIFACT_NAME = "roc_curves"

    def __init__(
        self,
        model=None,
        X_test=None,
        y_test=None,
        pos_label=None,
        sample_weight=None,
        drop_intermediate: bool = True,
        average: str = "macro",
        max_fpr=None,
        multi_class: str = "raise",
        labels=None,
    ):
        """

        :param model: a fitted model
        :param X_test: train dataset used to verified a fitted model.
        :param y_test: target dataset.
        :param pos_label: The label of the positive class. When pos_label=None, if y_true is in {-1, 1} or {0, 1}, pos_label is set to 1, otherwise an error will be raised.
        :param sample_weight: Sample weights.
        :param drop_intermediate: Whether to drop some suboptimal thresholds which would not appear on a plotted ROC curve.
        :param average: If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data
        :param max_fpr: If not None, the standardized partial AUC [2] over the range [0, max_fpr] is returned.
        :param multi_class: Only used for multiclass targets. Determines the type of configuration to use.
        :param labels: Only used for multiclass targets. List of labels that index the classes in y_score
        """

        self._pos_label = pos_label
        self.sample_weight = sample_weight
        self.drop_intermediate = drop_intermediate
        self.average = average
        self.max_fpr = max_fpr
        self.multi_class = multi_class
        self.labels = labels

        super(ROCCurves, self).__init__(model=model, X_test=X_test, y_test=y_test)

    def is_ready(self, stage: MLPlanStages, is_probabilities: bool) -> bool:
        """
        Check whether or not the plan is fit for production by the given stage and prediction probabilities. The
        confusion matrix is ready only post prediction.
        :param stage:            The stage to check if the plan is ready.
        :param is_probabilities: True if the 'y_pred' that will be sent to 'produce' is a prediction of probabilities
                                 (from 'predict_proba') and False if not.
        :return: True if the plan is producible and False otherwise.
        """
        return stage == MLPlanStages.POST_FIT and not is_probabilities

    def produce(
        self, model, X_test, y_test, y_prob, **kwargs
    ) -> Dict[str, PlotlyArtifact]:
        """
        Produce the artifact according to this plan.
        :return: The produced artifact.
        """
        validate_numerical(X_test)
        validate_numerical(y_test)

        # One hot encode the labels in order to plot them
        y_onehot = pd.get_dummies(y_test, columns=y_test.columns.to_list())

        # Create an empty figure, and iteratively add new lines
        # every time we compute a new class
        fig = go.Figure()
        fig.add_shape(type="line", line={"dash": "dash"}, x0=0, x1=1, y0=0, y1=1)

        for i in range(y_prob.shape[1]):
            y_true = y_onehot.iloc[:, i]
            y_score = y_prob[:, i]

            fpr, tpr, _ = roc_curve(
                y_true,
                y_score,
                pos_label=self._pos_label,
                sample_weight=self._sample_weight,
                drop_intermediate=self._drop_intermediate,
            )

            auc_score = roc_auc_score(
                y_true,
                y_score,
                average=self._average,
                sample_weight=self._sample_weight,
                max_fpr=self._max_fpr,
                multi_class=self._multi_class,
                labels=self._labels,
            )

            name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))

        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            yaxis={"scaleanchor": "x", "scaleratio": 1},
            xaxis={"constrain": "domain"},
            width=700,
            height=500,
        )

        # Creating an html rendering of the plot
        self._artifacts[self._ARTIFACT_NAME] = PlotlyArtifact(
            figure=fig, key=self._ARTIFACT_NAME
        )
        return self._artifacts
