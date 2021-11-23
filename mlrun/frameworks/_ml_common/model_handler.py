from abc import ABC

from mlrun.frameworks._common import ModelHandler


class MLModelHandler(ModelHandler, ABC):
    """
    Abstract class for a machine learning framework model handling, enabling loading, saving and logging it during runs.
    """

    pass
