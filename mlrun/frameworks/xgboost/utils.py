from typing import Union

import xgboost as xgb

from .._ml_common import DatasetType as MLDatasetType

# A type for all the supported dataset types:
DatasetType = Union[MLDatasetType, xgb.DMatrix]
