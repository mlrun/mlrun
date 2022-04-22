from abc import ABC

from .._ml_common import MLMLRunInterface


class XGBModelMLRunInterface(MLMLRunInterface, ABC):
    """
    Interface for adding MLRun features for XGBoost models (SciKit-Learn API models).
    """

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-xgboost"
