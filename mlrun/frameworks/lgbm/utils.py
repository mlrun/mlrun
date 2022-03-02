from typing import Union

import lightgbm as lgb

from .._ml_common import DatasetType as MLDatasetType


ModelType = Union[lgb.LGBMModel, lgb.Booster]

# A type for all the supported dataset types:
DatasetType = Union[MLDatasetType, lgb.Dataset]
