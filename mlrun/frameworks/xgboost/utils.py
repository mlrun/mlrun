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

from __future__ import annotations

from typing import Union

import xgboost as xgb

from .._ml_common import MLTypes, MLUtils


class XGBoostTypes(MLTypes):
    """
    Typing hints for the XGBoost framework.
    """

    # Union (not |) so this is not evaluated at import time during docs build:
    ModelType = Union[xgb.XGBModel, xgb.Booster]  # noqa: UP007
    DatasetType = Union[MLTypes.DatasetType, xgb.DMatrix]  # noqa: UP007


class XGBoostUtils(MLUtils):
    """
    Utilities functions for the XGBoost framework.
    """

    pass
