from typing import Union

import xgboost as xgb

from .._ml_common import MLTypes, MLUtils


class XGBoostTypes(MLTypes):
    """
    Typing hints for the XGBoost framework.
    """

    # A union of all XGBoost model base classes:
    ModelType = Union[xgb.XGBModel, xgb.Booster]

    # A type for all the supported dataset types:
    DatasetType = Union[MLTypes.DatasetType, xgb.DMatrix]


class XGBoostUtils(MLUtils):
    """
    Utilities functions for the XGBoost framework.
    """

    pass
