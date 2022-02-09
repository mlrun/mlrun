from abc import ABC

from .._ml_common import MLMLRunInterface


class LGBMModelMLRunInterface(MLMLRunInterface, ABC):
    """
    Interface for adding MLRun features for LightGBM models (SciKit-Learn API models).
    """

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-lgbm"
