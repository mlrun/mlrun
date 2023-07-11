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
