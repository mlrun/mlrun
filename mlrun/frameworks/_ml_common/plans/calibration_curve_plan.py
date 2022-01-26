from typing import Any, Dict
from enum import Enum
from abc import ABC, abstractmethod
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve

import inspect
import plotly.graph_objects as go
import pandas as pd
import mlrun
import plotly

from typing import Dict, List

import numpy as np
from plotly.figure_factory import create_annotated_heatmap
from sklearn.metrics import confusion_matrix

from mlrun.artifacts import Artifact, PlotlyArtifact

from ..._common import ModelType
from ..plan import MLPlanStages, MLPlotPlan
from ..utils import DatasetType


class CalibrationCurve(MLPlotPlan):
    """
    Compute true and predicted probabilities for a calibration curve. The method assumes the inputs come from a binary
    classifier, and discretize the [0, 1] interval into bins.
    """

    _ARTIFACT_NAME = "calibration_curve"

    def __init__(
        self, normalize: bool = False, n_bins: int = 5, strategy: str = "uniform",
    ):
        """
        https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html

        :param normalize: Whether the probabilities needs to be normalized into the [0, 1] interval, i.e. is not a
                          proper probability.
        :param n_bins:    Number of bins to discretize the [0, 1] interval.
        :param strategy:  Strategy used to define the widths of the bins. Can be on of {‘uniform’, ‘quantile’}.
                          Defaulted to "uniform".
        """

        # calibration_curve() parameters
        self._normalize = normalize
        self._n_bins = n_bins
        self._strategy = strategy

        super(CalibrationCurve, self).__init__(need_probabilities=True)

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
        prob_true, prob_pred = calibration_curve(
            y_test,
            y_prob,
            n_bins=self._n_bins,
            normalize=self._normalize,
            strategy=self._strategy,
        )

        fig = go.Figure(
            data=[go.Scatter(x=prob_true, y=prob_pred)],
            layout={"title": {"text": "Calibration Curve"}},
        )

        # add custom xaxis title
        fig.add_annotation(
            {
                "font": {"color": "black", "size": 14},
                "x": 0.5,
                "y": -0.15,
                "showarrow": False,
                "text": "prob_true",
                "xref": "paper",
                "yref": "paper",
            }
        )

        # add custom yaxis title
        fig.add_annotation(
            {
                "font": {"color": "black", "size": 14},
                "x": -0.1,
                "y": 0.5,
                "showarrow": False,
                "text": "prob_pred",
                "textangle": -90,
                "xref": "paper",
                "yref": "paper",
            }
        )

        # adjust margins to make room for yaxis title
        fig.update_layout(margin={"t": 100, "l": 100}, width=800, height=500)

        # Creating an html rendering of the plot
        self._artifacts[self._ARTIFACT_NAME] = PlotlyArtifact(
            figure=fig, key=self._ARTIFACT_NAME
        )
        return self._artifacts
