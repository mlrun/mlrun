from abc import ABC

from ..sklearn import SKLearnMLRunInterface


class XGBModelMLRunInterface(SKLearnMLRunInterface, ABC):
    """
    Interface for adding MLRun features for XGBoost models (SciKit-Learn API models).
    """

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-xgboost"
