from abc import ABC

from ...sklearn import SKLearnMLRunInterface


class LGBMModelMLRunInterface(SKLearnMLRunInterface, ABC):
    """
    Interface for adding MLRun features for LightGBM models (SciKit-Learn API).
    """

    # TODO: Should be changed from SKLearn's interface to its own, it has the same `params` and callbacks passed to
    #       `train`.
    # TODO: Add to `apply_mlrun` a "use_dask": bool = None argument. A boolean value that will replace the object of a
    #       SciKit-Learn API `LGBMModel` to its Dask version (`LGBMClassifier` to `DaskLGBMClassifier`). None will look
    #       for dask parameters in the given context and turn on and off accordingly.
    # MLRun's context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-lgbm"
