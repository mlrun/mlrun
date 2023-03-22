# Copyright 2018 Iguazio
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
from enum import Enum
from typing import Callable, List, Tuple, Union

import pandas as pd
from sklearn.base import is_classifier, is_regressor

from .._common import CommonTypes, CommonUtils


class AlgorithmFunctionality(Enum):
    """
    An enum for the type of a machine learning algorithm. The algorithm types are based on the model's type and the
    predictions it will need to do. The types are chosen by the following table:

    __________________________________________________________________________________
    |                | Algorithm Functionality Name           | Outputs | Classes    |
    |________________|________________________________________|_________|____________|
    | Classification | Binary Classification                  | 1       | 2          |
    |                | Multiclass Classification              | 1       | >2         |
    |                | Multi Output Classification            | >1      | 2          |
    |                | Multi Output Multiclass Classification | >1      | >2         |
    |________________|________________________________________|_________|____________|
    | Regression     | Regression                             | 1       | Continuous |
    |                | Multi Output Regression                | >1      | Continuous |
    |________________________________________________________________________________|
    """

    # A default value in case of unrecognized algorithm functionalities:
    UNKNOWN = "Unknown"

    # Classification:
    BINARY_CLASSIFICATION = "Binary Classification"
    MULTICLASS_CLASSIFICATION = "Multiclass Classification"
    MULTI_OUTPUT_CLASSIFICATION = "Multi Output Classification"
    MULTI_OUTPUT_MULTICLASS_CLASSIFICATION = "Multi Output Multiclass Classification"
    UNKNOWN_CLASSIFICATION = "Unknown Classification"

    # Regression:
    REGRESSION = "Regression"
    MULTI_OUTPUT_REGRESSION = "Multi Output Regression"
    UNKNOWN_REGRESSION = "Unknown Regression"

    def is_classification(self) -> bool:
        """
        Check whether this algorithm functionality is of a classification model.

        :return: True if classification and False if not.
        """
        return (
            self == AlgorithmFunctionality.BINARY_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTICLASS_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_MULTICLASS_CLASSIFICATION
            or self == AlgorithmFunctionality.UNKNOWN_CLASSIFICATION
        )

    def is_regression(self) -> bool:
        """
        Check whether this algorithm functionality is of a regression model.

        :return: True if regression and False if not.
        """
        return (
            self == AlgorithmFunctionality.REGRESSION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_REGRESSION
            or self == AlgorithmFunctionality.UNKNOWN_REGRESSION
        )

    def is_binary_classification(self) -> bool:
        """
        Check whether this algorithm functionality is of a binary classification model.

        :return: True if binary classification and False if not.
        """
        return (
            self == AlgorithmFunctionality.BINARY_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_CLASSIFICATION
        )

    def is_multiclass_classification(self) -> bool:
        """
        Check whether this algorithm functionality is of a multiclass classification model.

        :return: True if multiclass classification and False if not.
        """
        return (
            self == AlgorithmFunctionality.MULTICLASS_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_MULTICLASS_CLASSIFICATION
        )

    def is_single_output(self) -> bool:
        """
        Check whether this algorithm functionality is predicting a single output.

        :return: True if predicting a single output and False if not.
        """
        return (
            self == AlgorithmFunctionality.BINARY_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTICLASS_CLASSIFICATION
            or self == AlgorithmFunctionality.REGRESSION
        )

    def is_multi_output(self) -> bool:
        """
        Check whether this algorithm functionality is predicting multiple outputs.

        :return: True if predicting multiple outputs and False if not.
        """
        return (
            self == AlgorithmFunctionality.MULTI_OUTPUT_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_MULTICLASS_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_REGRESSION
        )


class MLTypes(CommonTypes, ABC):
    """
    Machine learning frameworks type hints.
    """

    # Type for a metric entry, can be passed as the metric function itself, as a callable object, a string of the name
    # of the function and the full module path to the function to import. Arguments to use when calling the metric can
    # be joined by wrapping it as a tuple:
    # TODO: will be moved to SKLearn's framework once LightGBM and XGBoost are fully supported.
    MetricEntryType = Union[Tuple[Union[Callable, str], dict], Callable, str]

    # Type for the target column name - a list of indices or column names that are the ground truth (y) of a dataset.
    TargetColumnsNamesType = Union[List[str], List[int]]


class MLUtils(CommonUtils, ABC):
    """
    Machine learning frameworks utilities.
    """

    @staticmethod
    def concatenate_x_y(
        x: CommonTypes.DatasetType,
        y: CommonTypes.DatasetType = None,
        target_columns_names: MLTypes.TargetColumnsNamesType = None,
        default_target_column_prefix: str = "y_",
    ) -> Tuple[pd.DataFrame, Union[MLTypes.TargetColumnsNamesType, None]]:
        """
        Concatenating the provided x and y data into a single pd.DataFrame, casting from np.ndarray and renaming y's
        original columns if 'y_columns' was not provided. The concatenated dataset index level will be reset to 0
        (multi-level indexes will be dropped using pandas 'reset_index' method).

        :param x:                            A collection of inputs to a model.
        :param y:                            A collection of ground truth labels corresponding to the inputs.
        :param target_columns_names:         List of names or indices to give the columns of the ground truth labels.
        :param default_target_column_prefix: A default value to join the y columns in case one of them is found in x (so
                                             there won't be any duplicates). Default: "y_".

        :return: A tuple of:
                 [0] = The concatenated x and y as a single DataFrame.
                 [1] = The target columns names / indices.
        """
        # Cast x to a DataFrame (from np.ndarray and pd.Series):
        x = MLUtils.to_dataframe(dataset=x)
        if y is None:
            # Reset the indices levels:
            x = x.reset_index(drop=True)
            return x, None

        # Cast y to a DataFrame (from np.ndarray and pd.Series):
        y = MLUtils.to_dataframe(dataset=y)

        # Check if y's columns are given, if not set the default avoiding duplicates with x's columns:
        if target_columns_names is None:
            target_columns_names = [
                column
                if column not in x.columns
                else f"{default_target_column_prefix}{column}"
                for column in list(y.columns)
            ]

        # Override the columns with the names the user provided:
        y.columns = target_columns_names

        # Concatenate the x and y into a single dataset:
        dataset = pd.concat([x, y], axis=1)

        # Reset the indices levels:
        dataset.reset_index(drop=True, inplace=True)

        return dataset, target_columns_names

    @staticmethod
    def get_algorithm_functionality(
        model: MLTypes.ModelType, y: MLTypes.DatasetType = None
    ) -> AlgorithmFunctionality:
        """
        Get the algorithm functionality according to the provided model and ground truth labels.

        :param model: The model to check if its a regression model or a classification model.
        :param y:     The ground truth values to check if its multiclass and / or multi output.

        :return: The algorithm functionality enum value.
        """
        # Convert the provided ground truths to DataFrame:
        if y is not None:
            y = MLUtils.to_dataframe(dataset=y)

        # Check for classification:
        if is_classifier(model):
            # Check if y is provided:
            if y is None:
                return AlgorithmFunctionality.UNKNOWN_CLASSIFICATION
            # Check amount of columns:
            if len(y.columns) == 1:
                # Check amount of classes:
                if len(pd.unique(y.to_numpy().flatten())) <= 2:
                    return AlgorithmFunctionality.BINARY_CLASSIFICATION
                return AlgorithmFunctionality.MULTICLASS_CLASSIFICATION
            # More than one column, check amount of classes (2 columns means binary - 1 column for each class):
            if len(y.columns) == 2:
                return AlgorithmFunctionality.MULTI_OUTPUT_CLASSIFICATION
            return AlgorithmFunctionality.MULTI_OUTPUT_MULTICLASS_CLASSIFICATION

        # Check for regression:
        if is_regressor(model):
            # Check if y is provided:
            if y is None:
                return AlgorithmFunctionality.UNKNOWN_REGRESSION
            # Check amount of columns:
            if len(y.columns) == 1:
                return AlgorithmFunctionality.REGRESSION
            return AlgorithmFunctionality.MULTI_OUTPUT_REGRESSION

        # Unrecognized model:
        return AlgorithmFunctionality.UNKNOWN
