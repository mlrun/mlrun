from typing import Union

import sklearn.base

# A union of all SciKitLearn model base classes:
SKLearnModelType = Union[
    sklearn.base.BaseEstimator,
    sklearn.base.BiclusterMixin,
    sklearn.base.ClassifierMixin,
    sklearn.base.ClusterMixin,
    sklearn.base.DensityMixin,
    sklearn.base.RegressorMixin,
    sklearn.base.TransformerMixin,
]
