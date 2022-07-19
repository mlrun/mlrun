from typing import Union

import sklearn.base

from .._ml_common import MLTypes, MLUtils


class SKLearnTypes(MLTypes):
    """
    Typing hints for the SciKit-Learn framework.
    """

    # A union of all SciKitLearn model base classes:
    ModelType = Union[
        sklearn.base.BaseEstimator,
        sklearn.base.BiclusterMixin,
        sklearn.base.ClassifierMixin,
        sklearn.base.ClusterMixin,
        sklearn.base.DensityMixin,
        sklearn.base.RegressorMixin,
        sklearn.base.TransformerMixin,
    ]


class SKLearnUtils(MLUtils):
    """
    Utilities functions for the SciKit-Learn framework.
    """

    pass
