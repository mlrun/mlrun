from enum import Enum
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse.base
from sklearn.base import is_classifier, is_regressor

import mlrun

from .._common import ModelType

# A type for all the supported dataset types:
DatasetType = Union[
    list, dict, np.ndarray, pd.DataFrame, pd.Series, scipy.sparse.base.spmatrix
]

# Type for a metric entry, can be passed as the metric function itself, as a callable object, a string of the name of
# the function and the full module path to the function to import. Arguments to use when calling the metric can be
# joined by wrapping it as a tuple:
MetricEntry = Union[Tuple[Union[Callable, str], dict], Callable, str]


def to_dataframe(dataset: DatasetType) -> pd.DataFrame:
    """
    Convert the given dataset to pd.DataFrame.

    :param dataset: The dataset to convert. Must be one of {np.ndarray, pd.Series, scipy.sparse.base.spmatrix}.

    :return: The dataset as a DataFrame.

    :raise MLRunInvalidArgumentError: If the dataset type is not supported.
    """
    if isinstance(dataset, pd.DataFrame):
        return dataset
    if isinstance(dataset, (np.ndarray, pd.Series, list, dict)):
        return pd.DataFrame(dataset)
    if isinstance(dataset, scipy.sparse.base.spmatrix):
        return pd.DataFrame.sparse.from_spmatrix(dataset)
    raise mlrun.errors.MLRunInvalidArgumentError(
        f"Could not convert the given dataset into a pandas DataFrame. Supporting conversion from: "
        f"'numpy.ndarray', 'pandas.Series' and 'scipy.sparse.base.spmatrix'. The given dataset was of type: "
        f"'{type(dataset)}'"
    )


def concatenate_x_y(
    x: DatasetType,
    y: DatasetType = None,
    y_columns: Union[List[str], List[int]] = None,
) -> Tuple[pd.DataFrame, Union[List[str], List[int], None]]:
    """
    Concatenating the provided x and y data into a single pd.DataFrame, casting from np.ndarray and renaming y's
    columns if provided (y_columns). The concatenated dataset index level will be reset to 0 (multi-level indexes
    will be dropped using pandas 'reset_index' method).

    :param x:         A collection of inputs to a model.
    :param y:         A collection of ground truth labels corresponding to the inputs.
    :param y_columns: List of names or indices to give the columns of the ground truth labels.

    :return: A tuple of:
             [0] = The concatenated x and y as a single DataFrame.
             [1] = The y columns names / indices.
    """
    # Cast x to a DataFrame (from np.ndarray and pd.Series):
    x = to_dataframe(dataset=x)
    if y is None:
        # Reset the indices levels:
        x = x.reset_index(drop=True)
        return x, None

    # Cast y to a DataFrame (from np.ndarray and pd.Series):
    y = to_dataframe(dataset=y)

    # Check if y's columns are given:
    if y_columns is not None:
        # Override the columns with the names the user provided:
        y.columns = y_columns
    else:
        # Get the y columns:
        y_columns = list(y.columns)

    # Concatenate the x and y into a single dataset:
    dataset = pd.concat([x, y], axis=1)

    # Reset the indices levels:
    dataset.reset_index(drop=True, inplace=True)

    return dataset, y_columns


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

    # Classification:
    BINARY_CLASSIFICATION = "Binary Classification"
    MULTICLASS_CLASSIFICATION = "Multiclass Classification"
    MULTI_OUTPUT_CLASSIFICATION = "Multi Output Classification"
    MULTI_OUTPUT_MULTICLASS_CLASSIFICATION = "Multi Output Multiclass Classification"

    # Regression:
    REGRESSION = "Regression"
    MULTI_OUTPUT_REGRESSION = "Multi Output Regression"

    @classmethod
    def get_algorithm_functionality(
        cls, model: ModelType, y: DatasetType
    ) -> "AlgorithmFunctionality":
        """
        Get the algorithm functionality according to the provided model and ground truth labels.

        :param model: The model to check if its a regression model or a classification model.
        :param y:     The ground truth values to check if its multiclass and / or multi output.

        :return: The algorithm functionality enum value.

        :raise MLRunInvalidArgumentError: If the model was not recognized to be a classifier or regressor.
        """
        # Convert the provided ground truths to DataFrame:
        y = to_dataframe(dataset=y)

        # Check for classification:
        if is_classifier(model):
            # Check amount of columns:
            if len(y.columns) == 1:
                # Check amount of classes:
                if len(pd.unique(y)) <= 2:
                    return cls.BINARY_CLASSIFICATION
                return cls.MULTICLASS_CLASSIFICATION
            # More than one column, check amount of classes:
            if len(pd.unique(y.to_numpy().flatten())) <= 2:
                return cls.MULTI_OUTPUT_CLASSIFICATION
            return cls.MULTI_OUTPUT_MULTICLASS_CLASSIFICATION

        # Check for regression:
        if is_regressor(model):
            # Check amount of columns:
            if len(y.columns) == 1:
                return cls.REGRESSION
            return cls.MULTI_OUTPUT_REGRESSION

        # Unrecognized model:
        raise mlrun.errors.MLRunInvalidArgumentError(
            f"Could not figure out if the given model '{type(model)}' is a classifier or regressor. Please contact us "
            f"on GitHub at https://github.com/mlrun/mlrun with the type of model that failed being recognized. You can "
            f"also use an explicit list of desired artifacts instead of calling the default method."
        )

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
        )

    def is_regression(self) -> bool:
        """
        Check whether this algorithm functionality is of a regression model.

        :return: True if regression and False if not.
        """
        return (
            self == AlgorithmFunctionality.REGRESSION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_REGRESSION
        )

    def is_binary_classification(self) -> bool:
        """
        Check whether this algorithm functionality is of a binary classification model.

        :return: True if binary classification and False if not.
        """
        if self.is_regression():
            return False
        return (
            self == AlgorithmFunctionality.BINARY_CLASSIFICATION
            or self == AlgorithmFunctionality.MULTI_OUTPUT_CLASSIFICATION
        )

    def is_multiclass_classification(self) -> bool:
        """
        Check whether this algorithm functionality is of a multiclass classification model.

        :return: True if multiclass classification and False if not.
        """
        if self.is_regression():
            return False
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
