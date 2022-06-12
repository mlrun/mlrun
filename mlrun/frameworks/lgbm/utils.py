from typing import Union

import lightgbm as lgb

from .._ml_common import MLUtils, MLTypes


class LGBMTypes(MLTypes):
    """
    Typing hints for the LightGBM framework.
    """

    # A union of all LightGBM model base classes:
    ModelType = Union[lgb.LGBMModel, lgb.Booster]

    # A type for all the supported dataset types:
    DatasetType = Union[MLTypes.DatasetType, lgb.Dataset]


class LGBMUtils(MLUtils):
    """
    Utilities functions for the LightGBM framework.
    """

    pass
