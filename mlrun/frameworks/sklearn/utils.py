# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
