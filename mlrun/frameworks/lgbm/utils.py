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
from typing import Optional, Union

import lightgbm as lgb
import numpy as np
import pandas as pd

import mlrun.errors

from .._ml_common import AlgorithmFunctionality, MLTypes, MLUtils


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
        tuple[str, str, float, bool],  # As packaged in `lightgbm.train`
        tuple[str, str, float, bool, float],  # As packaged in `lightgbm.cv`
    ]

    # Detailed type for the named tuple `CallbackEnv` passed during LightGBM's training for the callbacks:
    CallbackEnvType = tuple[
        lgb.Booster, dict, int, int, int, list[EvaluationResultType]
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

    @staticmethod
    def get_algorithm_functionality(
        model: MLTypes.ModelType = None,
        y: MLTypes.DatasetType = None,
        objective: Optional[str] = None,
    ) -> AlgorithmFunctionality:
        """
        Get the algorithm functionality of the LightGBM model. If SciKit-Learn API is used, pass the LGBBMModel and a y
        sample. Otherwise, training API is used, so pass the objective of the params dictionary.

        The objectives here are taken from the official docs of LightGBBM at:
        https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters

        :param model:     The model to check if its a regression model or a classification model (SciKit-Learn API).
        :param y:         The ground truth values to check if its multiclass and / or multi output (SciKit-Learn API).
        :param objective: The objective string (Training API).

        :return: The objective's algorithm functionality.
        """
        # Check if LightGBM is being used with SciKit-Learn API:
        if objective is None:
            return super().get_algorithm_functionality(model=model, y=y)

        # Declare the conversion map according to the LightGBM docs:
        objective_to_algorithm_functionality_map = {
            # regression application:
            "regression": AlgorithmFunctionality.REGRESSION,
            "regression_l2": AlgorithmFunctionality.REGRESSION,
            "l2": AlgorithmFunctionality.REGRESSION,
            "mean_squared_error": AlgorithmFunctionality.REGRESSION,
            "mse": AlgorithmFunctionality.REGRESSION,
            "l2_root": AlgorithmFunctionality.REGRESSION,
            "root_mean_squared_error": AlgorithmFunctionality.REGRESSION,
            "rmse": AlgorithmFunctionality.REGRESSION,
            "regression_l1": AlgorithmFunctionality.REGRESSION,
            "l1": AlgorithmFunctionality.REGRESSION,
            "mean_absolute_error": AlgorithmFunctionality.REGRESSION,
            "mae": AlgorithmFunctionality.REGRESSION,
            "huber": AlgorithmFunctionality.REGRESSION,
            "fair": AlgorithmFunctionality.REGRESSION,
            "poisson": AlgorithmFunctionality.REGRESSION,
            "quantile": AlgorithmFunctionality.REGRESSION,
            "mape": AlgorithmFunctionality.REGRESSION,
            "mean_absolute_percentage_error": AlgorithmFunctionality.REGRESSION,
            "gamma": AlgorithmFunctionality.REGRESSION,
            "tweedie": AlgorithmFunctionality.REGRESSION,
            # binary classification application:
            "binary": AlgorithmFunctionality.BINARY_CLASSIFICATION,
            # multi-class classification application:
            "multiclass": AlgorithmFunctionality.MULTICLASS_CLASSIFICATION,
            "softmax": AlgorithmFunctionality.MULTICLASS_CLASSIFICATION,
            "multiclassova": AlgorithmFunctionality.MULTICLASS_CLASSIFICATION,
            "multiclass_ova": AlgorithmFunctionality.MULTICLASS_CLASSIFICATION,
            "ova": AlgorithmFunctionality.MULTICLASS_CLASSIFICATION,
            "ovr": AlgorithmFunctionality.MULTICLASS_CLASSIFICATION,
            # cross-entropy application
            "cross_entropy": AlgorithmFunctionality.BINARY_CLASSIFICATION,
            "xentropy": AlgorithmFunctionality.BINARY_CLASSIFICATION,
            "cross_entropy_lambda": AlgorithmFunctionality.BINARY_CLASSIFICATION,
            "xentlambda": AlgorithmFunctionality.BINARY_CLASSIFICATION,
            # ranking application
            "lambdarank": AlgorithmFunctionality.MULTICLASS_CLASSIFICATION,
            "rank_xendcg": AlgorithmFunctionality.MULTICLASS_CLASSIFICATION,
            "xendcg": AlgorithmFunctionality.MULTICLASS_CLASSIFICATION,
            "xe_ndcg": AlgorithmFunctionality.MULTICLASS_CLASSIFICATION,
            "xe_ndcg_mart": AlgorithmFunctionality.MULTICLASS_CLASSIFICATION,
            "xendcg_mart": AlgorithmFunctionality.MULTICLASS_CLASSIFICATION,
        }

        # Return unknown if the objective is not in the map and otherwise return its functionality:
        if objective not in objective_to_algorithm_functionality_map:
            raise AlgorithmFunctionality.UNKNOWN
        return objective_to_algorithm_functionality_map[objective]
