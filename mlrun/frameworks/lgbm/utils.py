from typing import List, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd

import mlrun.errors

from .._ml_common import MLTypes, MLUtils


class LGBMTypes(MLTypes):
    """
    Typing hints for the LightGBM framework.
    """

    # A union of all LightGBM model base classes:
    ModelType = Union[lgb.LGBMModel, lgb.Booster]

    # A type for all the supported dataset types:
    DatasetType = Union[MLTypes.DatasetType, lgb.Dataset]

    # An evaluation result as packaged by the training in LightGBM:
    EvaluationResultType = Union[
        Tuple[str, str, float, bool],  # As packaged in `lightgbm.train`
        Tuple[str, str, float, bool, float],  # As packaged in `lightgbm.cv`
    ]

    # Detailed type for the named tuple `CallbackEnv` passed during LightGBM's training for the callbacks:
    CallbackEnvType = Tuple[
        lgb.Booster, dict, int, int, int, List[EvaluationResultType]
    ]


class LGBMUtils(MLUtils):
    """
    Utilities functions for the LightGBM framework.
    """

    @staticmethod
    def to_array(dataset: LGBMTypes.DatasetType) -> np.ndarray:
        """
        Convert the given dataset to np.ndarray.

        :param dataset: The dataset to convert. Must be one of {lgb.Dataset, pd.DataFrame, pd.Series,
                        scipy.sparse.base.spmatrix, list, tuple, dict}.

        :return: The dataset as a ndarray.

        :raise MLRunInvalidArgumentError: If the dataset type is not supported.
        """
        if isinstance(dataset, lgb.Dataset):
            x = LGBMUtils.to_array(dataset=dataset.data)
            if dataset.label is None:
                return x
            y = LGBMUtils.to_array(dataset=dataset.label)
            return LGBMUtils.to_array(LGBMUtils.concatenate_x_y(x=x, y=y)[0])
        try:
            return MLUtils.to_array(dataset=dataset)
        except mlrun.errors.MLRunInvalidArgumentError:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Could not convert the given dataset into a numpy ndarray. Supporting conversion from: "
                f"{LGBMUtils.get_union_typehint_string(LGBMTypes.DatasetType)}. The given dataset was of type: "
                f"'{type(dataset)}'"
            )

    @staticmethod
    def to_dataframe(dataset: LGBMTypes.DatasetType) -> pd.DataFrame:
        """
        Convert the given dataset to pd.DataFrame.

        :param dataset: The dataset to convert. Must be one of {lgb.Dataset, np.ndarray, pd.Series,
                        scipy.sparse.base.spmatrix, list, tuple, dict}.

        :return: The dataset as a DataFrame.

        :raise MLRunInvalidArgumentError: If the dataset type is not supported.
        """
        if isinstance(dataset, lgb.Dataset):
            x = LGBMUtils.to_dataframe(dataset=dataset.data)
            if dataset.label is None:
                return x
            y = LGBMUtils.to_dataframe(dataset=dataset.label)
            return LGBMUtils.concatenate_x_y(x=x, y=y)[0]
        try:
            return MLUtils.to_dataframe(dataset=dataset)
        except mlrun.errors.MLRunInvalidArgumentError:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Could not convert the given dataset into a pandas DataFrame. Supporting conversion from: "
                f"{LGBMUtils.get_union_typehint_string(LGBMTypes.DatasetType)}. The given dataset was of type: "
                f"'{type(dataset)}'"
            )
