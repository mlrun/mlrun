from abc import ABC

from .._ml_common import MLMLRunInterface


class SKLearnMLRunInterface(MLMLRunInterface, ABC):
    """
    Interface for adding MLRun features for SciKit-Learn API.
    """

    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-sklearn"
